from typing import Dict, List, Union
from pathlib import Path
import re
import datetime
import seyfert.utils.formatters as fm
import seyfert.utils.general_utils as gu
from seyfert.utils.workspace import WorkSpace


SUCCESS_STRING = "SUCCESS"
FAILURE_STRING = "FAILURE"
UNDEFINED_STRING = "UNDEFINED"
NOT_EXECUTED_STRING = "NOT_EXECUTED"
NOT_COMPLETED_STRING = "NOT_COMPLETED"


class JobReport:
    err_file: Path
    out_file: Path
    log_file: Path
    output_file: Path

    def __init__(self, jobdir: Path = None, output_pattern: str = None):
        self.jobdir = jobdir
        self.job_id = None
        self.status = None
        self.err_file = None
        self.out_file = None
        self.log_file = None
        self.output_file = None
        self.output_file_pattern = output_pattern
        self.output_size_MB = 0
        self.execution_time_secs = 0
        self.exec_hosts = []
        self.regex_dict = {'job_id': re.compile(r'Job\s+([0-9]+)'),
                           'exec_host': re.compile(r'<([a-z0-9*]+)\.ge\.infn\.it>'),
                           'cpu_time': re.compile(r'CPU time\s*:\s+([0-9.]+)'),
                           'core_dump': re.compile(r'[cC]ore\s+[dD]ump'),
                           'core_dump_file_pattern': re.compile(r'core\.[0-9]+')}
        self.core_dumped = False

    def setUp(self) -> None:
        self.getFiles()
        self.evaluateJobStatus()
        if self.out_file is not None:
            self.parseOutFile()

    def getReport(self) -> str:
        job_report_string = f'## {self.jobdir.name}\n'
        job_report_string += f'- JobID: {self.job_id}\n'
        job_report_string += f'- Status: {self.status}\n'
        job_report_string += f'- Execution Hosts: {" ".join(self.exec_hosts)}\n'
        job_report_string += f'- Execution Time: {fm.string_time_format(self.execution_time_secs)}\n'
        if self.status == SUCCESS_STRING:
            job_report_string += f'- Output file:\n'
            job_report_string += f'  - Name: {self.output_file.name}\n'
            job_report_string += f'  - Size: {self.output_size_MB} MB\n'
        else:
            if self.status != NOT_EXECUTED_STRING:
                job_report_string += f'- Core Dump: {self.core_dumped}\n'
                job_report_string += f'- Errfile Path: {self.err_file}\n'
                err_content = self.err_file.read_text()
                if err_content:
                    job_report_string += f'- Errfile Content: \n{err_content}\n'
        job_report_string += '\n'
        return job_report_string

    def getFiles(self) -> None:
        self.err_file = self.getFilePathFromJobdirWithPattern('job*.err')
        self.out_file = self.getFilePathFromJobdirWithPattern('job*.out')
        self.log_file = self.getFilePathFromJobdirWithPattern('*.log')
        self.output_file = self.getFilePathFromJobdirWithPattern(self.output_file_pattern)

    def evaluateJobStatus(self) -> None:
        if self.err_file is None:
            if self.out_file is None:
                self.status = NOT_EXECUTED_STRING
            else:
                self.status = UNDEFINED_STRING
        else:
            err_content = self.err_file.read_text()
            if len(err_content) > 0:
                self.status = FAILURE_STRING
                core_dump_files = [x for x in self.jobdir.iterdir()
                                   if self.regex_dict["core_dump_file_pattern"].match(x.name)]
                self.core_dumped = bool(self.regex_dict['core_dump'].search(err_content)) or len(core_dump_files) > 0
            else:
                if self.output_file is not None and self.output_file.exists():
                    self.status = SUCCESS_STRING
                    outsize_bytes = self.output_file.stat().st_size
                    self.output_size_MB = outsize_bytes / 1e6
                else:
                    self.status = UNDEFINED_STRING

    def parseOutFile(self) -> None:
        out_content = self.out_file.read_text()
        self.execution_time_secs = self.getExecutionTimeFromOutfile(out_content)
        job_id_match = self.regex_dict['job_id'].search(out_content)
        self.job_id = job_id_match.groups()[0] if job_id_match is not None else None
        self.exec_hosts = [x for x in self.regex_dict['exec_host'].findall(out_content)
                           if not x.startswith('farmui')]

    def getExecutionTimeFromOutfile(self, out_content: str) -> float:
        start_time_regex = re.compile('Started at ([a-zA-Z0-9\s:]+)')
        end_time_regex = re.compile('Results reported on ([a-zA-Z0-9\s:]+)')
        times = {}
        for line in out_content.splitlines():
            start_time_match = start_time_regex.search(line)
            end_time_match = end_time_regex.search(line)
            if start_time_match:
                ti_string = start_time_match.groups()[0]
                ti_string = self.convertDateTimeString(ti_string)
                times['ti'] = datetime.datetime.strptime(ti_string, '%Y %b %d %H:%M:%S')
            elif end_time_match:
                tf_string = end_time_match.groups()[0]
                tf_string = self.convertDateTimeString(tf_string)
                times['tf'] = datetime.datetime.strptime(tf_string, '%Y %b %d %H:%M:%S')
        try:
            execution_time_secs = (times['tf'] - times['ti']).seconds
        except KeyError:
            execution_time_secs = 0
        return execution_time_secs

    @staticmethod
    def convertDateTimeString(s: "str") -> "str":
        month, day, time, year = s.split()[1:]
        return f'{year} {month} {int(day):02} {time}'

    def getFilePathFromJobdirWithPattern(self, pattern: str) -> "Union[Path, None]":
        file_candidates = list(self.jobdir.glob(f'{pattern}'))
        file_path = file_candidates[0] if len(file_candidates) > 0 else None
        return file_path


class TaskReportMaker:
    workdir: Path
    job_reports: List[JobReport]
    is_task_successful: bool
    summary: str
    hosts: Dict[str, set]

    def __init__(self, workdir: "Union[str, Path]"):
        self.workdir = Path(workdir)
        self.job_reports = None
        self.n_jobs = None
        self.n_failed_jobs = 0
        self.n_core_dumps = 0
        self.n_succeeded_jobs = 0
        self.n_not_executed_jobs = 0
        self.n_undefined_jobs = 0
        self.mean_exec_time = None
        self.output_file_pattern = '*.csv' if self.workdir.name == 'Fisher' else '*.h5'
        self.total_output_size_MB = None
        self.summary = None
        self.status = None
        self.hosts = None

    def buildJobsReports(self):
        self.job_reports = []
        jobdirs = [x for x in self.workdir.iterdir() if x.is_dir()]
        self.mean_exec_time = 0
        self.total_output_size_MB = 0
        for jobdir in jobdirs:
            job_report = JobReport(jobdir=jobdir, output_pattern=self.output_file_pattern)
            job_report.setUp()
            self.job_reports.append(job_report)
            self.mean_exec_time += job_report.execution_time_secs
            self.total_output_size_MB += job_report.output_size_MB
        self.n_jobs = len(self.job_reports)
        self.mean_exec_time /= self.n_jobs
        if all([x.job_id is not None for x in self.job_reports]):
            self.job_reports.sort(key=lambda x: x.job_id)

    def writeToFiles(self, report_outfile=None, summary_outfile=None) -> None:
        summary_outfile.write(self.summary)
        report_outfile.write(self.summary)
        for job_report in self.job_reports:
            job_report_string = job_report.getReport()
            report_outfile.write(job_report_string)

    def evaluateStatus(self) -> None:
        good_hosts = set()
        bad_hosts = set()
        for job_report in self.job_reports:
            if job_report.status == FAILURE_STRING:
                self.n_failed_jobs += 1
                for host in job_report.exec_hosts:
                    bad_hosts.add(host)
            elif job_report.status == SUCCESS_STRING:
                self.n_succeeded_jobs += 1
                for host in job_report.exec_hosts:
                    good_hosts.add(host)
            elif job_report.status == NOT_EXECUTED_STRING:
                self.n_not_executed_jobs += 1
            elif job_report.status == UNDEFINED_STRING:
                self.n_undefined_jobs += 1
            if job_report.core_dumped:
                self.n_core_dumps += 1

        self.hosts = {'good': good_hosts, 'bad': bad_hosts}

        if self.n_failed_jobs > 0:
            self.status = FAILURE_STRING
        elif self.n_succeeded_jobs == self.n_jobs:
            self.status = SUCCESS_STRING
        elif self.n_not_executed_jobs == self.n_jobs:
            self.status = NOT_EXECUTED_STRING
        elif all([job.status == SUCCESS_STRING or job.status == NOT_EXECUTED_STRING
                  for job in self.job_reports]):
            self.status = NOT_COMPLETED_STRING
        else:
            self.status = UNDEFINED_STRING

    def buildReportSummary(self):
        summary = f'########### WORKDIR: {self.workdir.name} {self.status} ###########\n'
        if self.status != NOT_EXECUTED_STRING:
            summary += f'Workdir: {self.workdir}\n'
            summary += f'{self.n_succeeded_jobs}/{self.n_jobs} jobs succeeded\n'
            summary += f'{self.n_failed_jobs}/{self.n_jobs} jobs failed\n'
            summary += f'{self.n_core_dumps}/{self.n_failed_jobs} failed jobs core dumped\n'
            summary += f'Mean Execution Time: {fm.string_time_format(self.mean_exec_time)}\n'
            summary += f'Total output size: {self.total_output_size_MB:.3f} MB\n'
            summary += f'Hosts on which jobs failed: {", ".join(self.hosts["bad"])}\n'
            summary += f'Hosts on which jobs succeeded: {", ".join(self.hosts["good"])}\n'
        summary += '\n'
        self.summary = summary


class RunReportMaker:
    rundir: "Path"
    workdirs: "List[Path]"
    workspace: "WorkSpace"
    task_report_makers: "List[TaskReportMaker]"
    is_run_successful: bool
    run_summary: str

    def __init__(self, rundir: "Union[str, Path]"):
        self.rundir = Path(rundir)
        self.workspace = WorkSpace(self.rundir)
        self.workdirs = None
        self.report_outfile = None
        self.summary_outfile = None
        self.task_report_makers = None
        self.run_summary = None
        self.status = None

        self.workdirs = [
            self.workspace.pmm_dir,
            self.workspace.cl_dir,
            self.workspace.der_dir,
            self.workspace.fish_elems_dir
        ]

    def writeReportFiles(self):
        run_name = self.rundir.name
        self.report_outfile = self.rundir / f'report_{run_name}.log'
        self.summary_outfile = self.rundir / f'summary_{run_name}.log'
        if all([trm.status == SUCCESS_STRING for trm in self.task_report_makers]):
            self.status = SUCCESS_STRING
        elif any([trm.status == FAILURE_STRING for trm in self.task_report_makers]):
            self.status = FAILURE_STRING
        elif all([trm.status == SUCCESS_STRING or trm.status == NOT_EXECUTED_STRING
                  for trm in self.task_report_makers]):
            self.status = NOT_COMPLETED_STRING
        else:
            self.status = UNDEFINED_STRING
        with open(self.report_outfile, 'w') as rf, open(self.summary_outfile, 'w') as sf:
            run_status_string = f'Run {run_name}: {self.status}\n'
            rf.write(run_status_string)
            sf.write(run_status_string)
            self.run_summary = ''
            self.run_summary += run_status_string
            for trm in self.task_report_makers:
                trm.writeToFiles(report_outfile=rf, summary_outfile=sf)
                self.run_summary += trm.summary

    def buildTaskReportMakers(self):
        self.task_report_makers = []
        for workdir in self.workdirs:
            if workdir.exists() and not workdir.is_symlink():
                trm = TaskReportMaker(workdir=workdir)
                trm.buildJobsReports()
                trm.evaluateStatus()
                trm.buildReportSummary()
                self.task_report_makers.append(trm)
