from typing import List, ByteString, Union, Tuple
import logging
import datetime
import re
import pandas as pd
import subprocess as sp
import time
from pathlib import Path

from seyfert.utils import formatters as fm

logger = logging.getLogger(__name__)
bsub_stdout_regex = re.compile(r"Job <([0-9]+)> is submitted to")


class ExitedBatchJobError(Exception):
    pass


def count_down(t):
    while t:
        time.sleep(1)
        t -= 1


class BsubInterface:
    _bjobs_out_regex_dict = {
        "JOBID": r"([0-9]+)",
        "USER": r"([a-z]+)",
        "STAT": r"([A-Z]+)",
        "QUEUE": r"([a-z]+)",
        "FROM_HOST": r"([a-z0-9\.]+)",
        "EXEC_HOST": r"([a-z0-9\.\*]+)",
        "JOB_NAME": r"(.*\-)",
        "SUBMIT_TIME": r"([a-zA-Z]{3}\s+[0-9]+\s+[0-9]+:[0-9]+)"
    }
    _bsub_out_regex = re.compile(r"Job <([0-9]+)> is submitted to")
    _basic_bjobs_out_regex = re.compile(r"([0-9]+)\s+([a-z]+)\s+([A-Z]+)", re.MULTILINE)

    def __init__(self):
        self.regex_bjobs_out = re.compile(r"\s+".join(self._bjobs_out_regex_dict.values()), re.MULTILINE)
        self.bsub_opts = {
            "-P": "c7",
            "-q": "medium",
            "-M": 2048
        }

    @property
    def req_memory_MB(self):
        return self.bsub_opts['-M']

    @staticmethod
    def decodeIfNotString(s) -> "str":
        if not isinstance(s, str):
            s = s.decode('utf-8')

        return s

    def buildResourceSpecString(self) -> "str":
        return f'-R"select[avx && mem>{self.req_memory_MB}] rusage[mem={self.req_memory_MB}]"'

    def setOptions(self, queue: "str", n_cores: "int", memory_MB: "int"):
        n_cores = str(n_cores)
        memory_MB = str(memory_MB)
        self.bsub_opts.update({
            "-q": queue, "-n": n_cores, "-M": memory_MB
        })

    def parseBjobsOutput(self, out: "Union[str, ByteString]", basic_regex=True) -> "pd.DataFrame":
        out = self.decodeIfNotString(out)

        if basic_regex:
            rows = self._basic_bjobs_out_regex.findall(out)
            columns = list(self._bjobs_out_regex_dict.keys())[0:3]
        else:
            rows = self.regex_bjobs_out.findall(out)
            columns = list(self._bjobs_out_regex_dict.keys())

        return pd.DataFrame(rows, columns=columns)

    def buildBsubOptsString(self, err, out):
        cli_args = {}
        cli_args.update(self.bsub_opts)
        cli_args.update({"-e": err, "-o": out})

        R_spec = self.buildResourceSpecString()
        opts_str = ' '.join([f"{key} {value}" for key, value in cli_args.items()]) + f' {R_spec}'

        return opts_str
    
    def buildBsubCommand(self, cmd, err, out) -> "str":
        opts_str = self.buildBsubOptsString(err=err, out=out)

        return f"bsub {opts_str} {cmd}"

    def submitJob(self, cmd_to_execute: "str", add_datetime_to_logs=True, logs_path=".",
                  test=False, logs_start_str="job", silent=False, separate_err_out_dirs=False) -> "str":
        out_name = f"out_{logs_start_str}"
        err_name = f"err_{logs_start_str}"
        if add_datetime_to_logs:
            now = fm.datetime_str_format(datetime.datetime.now())
            out_name += f"_{now}.out"
            err_name += f"_{now}.err"

        if separate_err_out_dirs:
            out_path = Path(logs_path) / "out" / out_name
            err_path = Path(logs_path) / "err" / err_name
        else:
            out_path = Path(logs_path) / out_name
            err_path = Path(logs_path) / err_name

        out_path.parent.mkdir(exist_ok=True, parents=True)
        err_path.parent.mkdir(exist_ok=True, parents=True)

        bsub_cmd_string = self.buildBsubCommand(cmd_to_execute, err=err_path, out=out_path)

        if not silent:
            logger.info(f"Running command: \n{bsub_cmd_string}")
        if not test:
            proc = sp.run(bsub_cmd_string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
            if proc.stdout and not silent:
                logger.info(proc.stdout.decode('utf-8').strip())
            job_id = self.getJobID(proc.stdout)
        else:
            logger.info(f"test mode: skipping")
            job_id = None

        return job_id

    def getJobID(self, bsub_std_out: "Union[str, ByteString]"):
        bsub_std_out = self.decodeIfNotString(bsub_std_out)

        return self._bsub_out_regex.search(bsub_std_out).groups()[0]

    def getBjobsTable(self, job_ids: "List" = None, bjobs_opts="-w", basic_regex=True):
        if job_ids is not None:
            bjobs_cmd = sp.run(f"bjobs {bjobs_opts} {' '.join([str(job_id) for job_id in job_ids])}",
                               shell=True, stdout=sp.PIPE)
        else:
            bjobs_cmd = sp.run(f"bjobs {bjobs_opts}", shell=True, stdout=sp.PIPE)

        return self.parseBjobsOutput(bjobs_cmd.stdout, basic_regex=basic_regex)

    def waitForJobsToComplete(self, job_ids: "List", bjobs_opts="-w",
                              check_time_resolution_secs=30.0, max_time_wait_h=3, wait_overhead_secs=5):
        logger.info(f"Waiting for jobs {job_ids[0]}-{job_ids[-1]} to execute")
        logger.info(f"Add overhead of {wait_overhead_secs} s to make sure bjobs returns something...")
        time.sleep(wait_overhead_secs)
        check_counter = 0
        num_checks_lim_log = 10
        max_time_wait_s = max_time_wait_h * 3600
        t0 = time.time()
        while True:
            if check_counter < num_checks_lim_log:
                logger.info(f"Waiting for {check_time_resolution_secs} s")

            count_down(check_time_resolution_secs)

            if check_counter < num_checks_lim_log:
                logger.info(f"Checking jobs status, check number: {check_counter}")

            status_df = self.getBjobsTable(job_ids, bjobs_opts=bjobs_opts)
            if all(status_df["STAT"] == 'DONE'):
                logger.info(f"Jobs {job_ids[0]}-{job_ids[-1]} have finished")
                break
            else:
                stat_counts = status_df["STAT"].value_counts()
                if check_counter < num_checks_lim_log:
                    stat_counts_str = ", ".join(f"{status} {num_status}"
                                                for status, num_status in sorted(stat_counts.items()))
                    logger.info(f"jobs situation: {stat_counts_str}")
                if "EXIT" in stat_counts and stat_counts["EXIT"] > 0:
                    raise ExitedBatchJobError(f"{stat_counts['EXIT']} jobs failed")

            check_counter += 1

            tf = time.time()
            if check_counter < num_checks_lim_log:
                logger.info(f"Elapsed time up to now: {fm.string_time_format(tf - t0)}")

            if (tf - t0) > max_time_wait_s:
                logger.info(f"The checks have been lasted for {max_time_wait_h} hours, going on")
                break

    def getResourceUsageTable(self, out_files_dir: "Path"):
        outname_regex = re.compile(r"^out_(.*)_202[0-1]")
        regexps = {
            "cpu_time_s": re.compile(r"CPU time\s*:\s+([0-9\.]+)"),
            "max_mem_MB": re.compile(r"Max Memory\s*:\s+([0-9\.]+)"),
            "avg_mem_MB": re.compile(r"Average Memory\s*:\s+([0-9\.]+)"),
            "req_mem_MB": re.compile(r"Total Requested Memory\s*:\s+([0-9\.]+)"),
            "delta_mem_MB": re.compile(r"Delta Memory\s*:\s+([0-9\.]+)"),
        }

        rows = []
        for file in out_files_dir.glob("*.out"):
            text = file.read_text()
            fisher = outname_regex.match(file.stem).groups()[0]
            start_time, end_time = self.extractStartEndTimesFromOutText(text)
            row = {
                "fisher": fisher,
                "start_time": start_time, "end_time": end_time,
                "duration_s": (end_time - start_time).seconds
            }
            row.update({key: float(pattern.search(text).groups()[0]) for key, pattern in regexps.items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        df["duration_h_m_s_ms"] = df["duration_s"].apply(fm.string_time_format)

        return df

    def extractStartEndTimesFromOutText(self, text: "str") -> "Tuple[datetime.datetime, datetime.datetime]":
        start_time = self.extractDateTimeFromOut(text, prefix="Started at")
        end_time = self.extractDateTimeFromOut(text, prefix="Results reported on")

        return start_time, end_time

    @staticmethod
    def extractDateTimeFromOut(text: "str", prefix: "str") -> "datetime.datetime":
        date_pattern = r"[A-Za-z]+ ([A-Za-z]{3})\s+([0-9]+)\s+([0-9]{2}:[0-9]{2}:[0-9]{2})\s+([0-9]{4})"
        date_time_regex = re.compile(r"%s %s" % (prefix, date_pattern))

        month, day, h_m_s, year = date_time_regex.search(text).groups()

        return datetime.datetime.strptime(f"{year} {month} {day.zfill(2)} {h_m_s}", "%Y %b %d %H:%M:%S")
