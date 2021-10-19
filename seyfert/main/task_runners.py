import json
import sys
import os
import datetime
import time
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Union
import logging
import subprocess as sp
from collections import namedtuple

import seyfert.utils.filesystem_utils as fsu
from seyfert.main.bsub_utils import BsubInterface
from seyfert.utils import general_utils as gu
from seyfert import VERSION
from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.cosmology.cosmology import Cosmology
from seyfert.utils.workspace import WorkSpace
from seyfert.config.cmd_line import JobSubmitterCommandLineArgs
from seyfert.config import forecast_config as fcfg, main_config as mcfg
from seyfert.utils import formatters as fm
from seyfert.fisher import fisher_utils
from seyfert.fisher.fisher_matrix import FisherMatrix
from seyfert.utils.type_helpers import TPathLike
from seyfert.main import cl_core

logger = logging.getLogger(__name__)
this_module = sys.modules[__name__]

Task = namedtuple("Task", ["name", "is_present", "input_file", "external_dir"])


class JobFailError(Exception):
    pass


class TaskRunner(ABC):
    rundir: "Path"
    outdir: "Path"
    forecast_config: "fcfg.ForecastConfig"
    config: "mcfg.MainConfig"
    checktime_resolution_secs: "float"
    err_files_path: "List[Path]"
    outfile_pattern: "str"

    # noinspection PyTypeChecker
    def __init__(self, workspace: "WorkSpace" = None, args: "JobSubmitterCommandLineArgs" = None,
                 json_input: "Union[str, Path]" = None, forecast_config: "fcfg.ForecastConfig" = None):
        self.workspace = workspace
        self.args = args
        self.json_input = json_input
        self.kind = self.__class__.__name__
        self.forecast_config = forecast_config
        self.config = None
        self.specific_config_filename = Path(self.json_input).name
        self.shell_scripts_paths = []
        self.checktime_resolution_secs = None
        self.main_script_parameters = {}
        self.main_script_name = None
        self.err_files_paths = []
        self.out_files_paths = []
        self.resubmitted_jobs = 0
        self.outfile_pattern = None
        self.venv_path = os.getenv("VIRTUAL_ENV")
        self.bsub = BsubInterface()
        self.task_name = None

    @property
    def rundir(self) -> "Path":
        return self.workspace.run_dir

    @property
    def verbose(self) -> "bool":
        return self.args.verbose

    @property
    def outdir(self) -> "Path":
        return self.workspace.task_dirs[self.task_name]

    @property
    def phys_pars(self) -> "PhysicalParametersCollection":
        return self.forecast_config.phys_pars

    @abstractmethod
    def createJobDirStructure(self):
        self.outdir.mkdir(exist_ok=True)

    def fillJobDir(self, jobdir_path=None) -> None:
        try:
            jobdir_path = Path(jobdir_path)
        except TypeError:
            raise TypeError(f"cannot cast {jobdir_path} as Path")
        sh_path = self.writeJobShellScript(jobdir_path=jobdir_path)
        self.shell_scripts_paths.append(sh_path)
        self.config.writeToJSON(jobdir_path / self.specific_config_filename)

    def writeJobShellScript(self, jobdir_path=None):
        jobdir_name = jobdir_path.name
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        sh_path = jobdir_path / f"job_{jobdir_name}_{now}.sh"
        self.main_script_parameters["workdir"] = jobdir_path
        self.main_script_parameters["logfile"] = f"logfile_{self.main_script_name}.log"
        self.main_script_parameters["main_config"] = self.specific_config_filename
        self.main_script_parameters["rundir"] = self.rundir
        with sh_path.open(mode="w") as sh:
            sh.write(f"#!/bin/bash\n\n")
            sh.write(f'cd {jobdir_path}\n')
            if self.venv_path is not None:
                sh.write(f'source {Path(self.venv_path) / "bin/activate"}\n')
            run_cmd = f'{self.main_script_name}'
            for key, value in self.main_script_parameters.items():
                run_cmd += f' --{key} {value}'
            sh.write(f'{run_cmd}\n')
        sh_path.chmod(0o755)
        return sh_path

    @abstractmethod
    def getJobDirName(self):
        pass

    @abstractmethod
    def createAndFillJobDir(self):
        pass

    def collectShellScripts(self):
        self.shell_scripts_paths = []
        self.shell_scripts_paths += list(self.outdir.glob('*/*.sh'))

    def run(self):
        if self.args.execution != 'no':
            logger.info(f'Running task {self.kind}')
            t0 = time.time()
            if self.args.execution == 'batch':
                self.bsub.setOptions(queue=self.args.queue, memory_MB=self.args.memory,
                                     n_cores=self.args.n_cores_per_job)
                self.submitBatchJobs()
            elif self.args.execution == 'interactive':
                self.executeJobsInteractively()
            tf = time.time()
            logger.info(f'Completed task {self.kind}')
            logger.info(f'Elapsed time: {fm.string_time_format(tf - t0)}')
        else:
            pass

    def submitBatchJobs(self):
        job_ids = []
        for sh_path in self.shell_scripts_paths:
            job_id = self.bsub.submitJob(cmd_to_execute=sh_path, logs_path=sh_path.parent, silent=True)
            job_ids.append(job_id)

        self.bsub.waitForJobsToComplete(job_ids)

    @staticmethod
    def executeCommand(cmd: "str"):
        t0 = time.time()
        proc = sp.run(cmd, shell=True, check=False, stdout=sp.PIPE, stderr=sp.PIPE)
        if proc.stderr:
            logger.error(proc.stderr.decode("utf-8"))
            raise JobFailError(f"Command {cmd} failed")
        if proc.stdout:
            logger.info(f'Stdout:\n{proc.stdout.decode("utf-8").strip()}')

        tf = time.time()
        logger.info(f'Elapsed time: {fm.string_time_format(tf - t0)}')

    def executeJobsInteractively(self):
        for sh_idx, sh_path in enumerate(self.shell_scripts_paths):
            logger.info(f'Task {self.kind} job {sh_idx + 1}/{len(self.shell_scripts_paths)}')
            logger.info(f'Job name: {sh_path.stem}')
            self.executeCommand(str(sh_path))


class PowerSpectrumTaskRunner(TaskRunner):
    config: mcfg.PowerSpectrumConfig

    def __init__(self, **kwargs):
        super(PowerSpectrumTaskRunner, self).__init__(**kwargs)
        self.checktime_resolution_secs = 60
        self.main_script_name = "compute_pmm"
        self.config = mcfg.PowerSpectrumConfig(json_input=self.json_input)
        self.outfile_pattern = 'p_mm.h5'
        self.task_name = "PowerSpectrum"

    def createJobDirStructure(self):
        super(PowerSpectrumTaskRunner, self).createJobDirStructure()
        self.createAndFillJobDir(dvar='central', step=0)
        for par_name, par in self.phys_pars.free_cosmological_parameters.items():
            if par.derivative_method == "SteM":
                logger.info(f"Creating power spectrum dir for SteM variations of {par_name}")
                for step in self.phys_pars.stem_steps:
                    logger.info(f"{par_name} step {step}")
                    self.createAndFillJobDir(dvar=par_name, step=step)

    def createAndFillJobDir(self, dvar=None, step=None):
        job_dir_name = self.getJobDirName(dvar=dvar, step=step)
        job_dir_path = self.outdir / job_dir_name
        job_dir_path.mkdir(exist_ok=True)
        self.fillJobDir(jobdir_path=job_dir_path)
        camb_ref_inifile = self.config.camb_ini_path
        shutil.copy(camb_ref_inifile, job_dir_path / camb_ref_inifile.name)

    def getJobDirName(self, dvar=None, step=None):
        return fsu.get_cosmology_jobdir_name(dvar, step)

    def setInputDirs(self, input_dirs: Dict[str, Path]):
        pass


class AngularTaskRunner(TaskRunner):
    config: mcfg.AngularConfig

    def __init__(self, **kwargs):
        super(AngularTaskRunner, self).__init__(**kwargs)
        self.main_script_name = "compute_cls"
        self.config = mcfg.AngularConfig(json_input=self.json_input)
        self.checktime_resolution_secs = 15
        self.outfile_pattern = 'cl*.h5'
        self.task_name = "Angular"

    def createJobDirStructure(self):
        super(AngularTaskRunner, self).createJobDirStructure()
        self.createAndFillJobDir(dvar='central', step=0)
        for par_name, par in self.phys_pars.free_physical_parameters.items():
            if par.derivative_method == "SteM":
                if self.verbose:
                    logger.info(f"Creating cl dir for SteM variations of {par_name}")
                for step in self.phys_pars.stem_steps:
                    if self.verbose:
                        logger.info(f"{par_name} step {step}")
                    self.createAndFillJobDir(dvar=par_name, step=step)
            else:
                logger.info(f"Differentiation method of {par_name} is {par.derivative_method}, "
                            f"not creating SteM step dir")

    def createAndFillJobDir(self, dvar=None, step=None):
        job_dir_name = self.getJobDirName(dvar=dvar, step=step)
        job_dir_path = self.outdir / job_dir_name
        job_dir_path.mkdir(exist_ok=True)
        self.fillJobDir(jobdir_path=job_dir_path)

    def getJobDirName(self, dvar=None, step=None):
        return fsu.get_cosmology_jobdir_name(dvar, step)


class DerivativeTaskRunner(TaskRunner):
    config: mcfg.DerivativeConfig

    def __init__(self, **kwargs):
        super(DerivativeTaskRunner, self).__init__(**kwargs)
        self.main_script_name = "compute_derivatives"
        self.config = mcfg.DerivativeConfig(json_input=self.json_input)
        self.checktime_resolution_secs = 15
        self.outfile_pattern = 'cl_derivative*.h5'
        self.task_name = "Derivative"

    def getJobDirName(self, dvar=None):
        return fsu.get_derivative_jobdir_name_from_dvar(dvar)

    def createJobDirStructure(self):
        super(DerivativeTaskRunner, self).createJobDirStructure()
        for par_name in self.phys_pars.free_physical_parameters:
            self.createAndFillJobDir(dvar=par_name)

    def createAndFillJobDir(self, dvar=None):
        jobdir_name = self.getJobDirName(dvar=dvar)
        jobdir_path = self.outdir / jobdir_name
        jobdir_path.mkdir(exist_ok=True)
        self.fillJobDir(jobdir_path=jobdir_path)


class FisherTaskRunner(TaskRunner):
    config: "mcfg.FisherConfig"

    def __init__(self, **kwargs):
        super(FisherTaskRunner, self).__init__(**kwargs)
        self.main_script_name = "compute_fisher_matrix"
        self.config = mcfg.FisherConfig(json_input=self.json_input)
        self.checktime_resolution_secs = 60
        self.outfile_pattern = '*.csv'
        self.RAM_memory_MB = 4000
        self.n_cores_per_job = 8
        self.task_name = "Fisher"

    def getJobDirName(self, dvar1=None, dvar2=None):
        return fsu.get_fisher_jobdir_name_from_dvar1_dvar2(dvar1, dvar2)

    def createJobDirStructure(self):
        pass

    def createAndFillJobDir(self, dvar_1=None, dvar_2=None):
        pass

    @staticmethod
    def loadFisherDatavectorBriefStrings():
        data_vectors = fisher_utils.load_selected_data_vectors()
        auto_dvs_brief = [v.toBriefString() for v in data_vectors[0:3]]
        cross_dvs_brief = [v.toBriefString() for v in data_vectors[3:]]

        return auto_dvs_brief, cross_dvs_brief

    def writeFisherCommand(self, datavectors_brief_strings: "List[str]"):
        return f"{self.main_script_name} {' '.join(datavectors_brief_strings)} --rundir {self.workspace.run_dir}"

    def clearOutput(self):
        if self.args.overwrite_results:
            logger.warning("Overwriting Fisher output")
            if self.workspace.base_fishers_dir.is_dir():
                shutil.rmtree(self.workspace.base_fishers_dir)
            if self.workspace.auto_f_ells_file.is_file():
                self.workspace.auto_f_ells_file.unlink()

    def submitBatchJobs(self):
        self.clearOutput()
        # set RAM usage according to the number of GCsp bins
        if self.forecast_config.n_sp_bins > 24:
            self.RAM_memory_MB = 10000

        self.bsub.setOptions(queue=self.args.queue, memory_MB=self.RAM_memory_MB, n_cores=self.args.n_cores_per_job)

        auto_dvs_brief, cross_dvs_brief = self.loadFisherDatavectorBriefStrings()
        logs_path = self.workspace.base_fishers_dir / f"logs_run_{fm.datetime_str_format(datetime.datetime.now())}"
        logs_path.mkdir(exist_ok=True, parents=True)

        if not self.workspace.auto_f_ells_file.exists():
            logger.info("Submitting auto-correlations computation...")
            auto_fish_cmd = self.writeFisherCommand(auto_dvs_brief)
            job_id = self.bsub.submitJob(cmd_to_execute=auto_fish_cmd, logs_path=logs_path, logs_start_str="auto_corrs",
                                         separate_err_out_dirs=True)

            self.bsub.waitForJobsToComplete([job_id])
            logger.info("auto-correlations job has finished.")
        else:
            logger.warning(f"auto fishers of ell file already exists at {self.workspace.auto_f_ells_file}, skipping "
                           f"auto-correlations computation")

        job_ids = []
        for brief_dv_str in cross_dvs_brief:
            fish_cmd = self.writeFisherCommand([brief_dv_str])
            logs_start_str = brief_dv_str
            job_id = self.bsub.submitJob(cmd_to_execute=fish_cmd, logs_path=logs_path, logs_start_str=logs_start_str,
                                         separate_err_out_dirs=True)
            job_ids.append(job_id)
            time.sleep(1.5)

        self.bsub.waitForJobsToComplete(job_ids)
        try:
            logger.info("Creating resource usage table")
            df = self.bsub.getResourceUsageTable(out_files_dir=logs_path / "out")
            df.to_csv(logs_path / "resources_usage.csv")
        except:
            logger.warning("Cannot create resource usage table for some reason, skipping")

    def executeJobsInteractively(self):
        self.clearOutput()
        auto_dvs_brief, cross_dvs_brief = self.loadFisherDatavectorBriefStrings()
        auto_fish_cmd = self.writeFisherCommand(auto_dvs_brief)
        logger.info("Starting auto-correlations computation...")
        self.executeCommand(auto_fish_cmd)

        logger.info("Starting cross-combinations computation...")
        cross_fish_cmd = self.writeFisherCommand(cross_dvs_brief)
        logger.info(f"Running command:\n{cross_fish_cmd}")
        self.executeCommand(cross_fish_cmd)


class ForecastRunner:
    rundir: "Path"
    submitters_dict: "Dict[str, TaskRunner]"
    tasks: "Dict[str, Task]"
    forecast_config: "fcfg.ForecastConfig"
    fisher_config: "mcfg.FisherConfig"
    workspace: "WorkSpace"
    args: "JobSubmitterCommandLineArgs"

    # noinspection PyTypeChecker
    def __init__(self, cmd_line_args: "Dict" = None, workdir: "Union[str, Path]" = Path("."),
                 rundir: "Union[str, Path]" = None):
        self.base_workdir = Path(workdir)
        self.args = None
        self.run_id = None
        self.rundir = Path(rundir) if rundir is not None else None
        self.workspace = None
        self.input_data_dir = None
        self.tasks = None
        self.forecast_config_file = None
        self.forecast_config = None
        self.submitters_dict = {}
        self.input_files_dict = None
        self.external_dirs = {}
        self.start_datetime = None
        self.end_datetime = None
        self.metadata = {}
        self.bsub = BsubInterface()
        if cmd_line_args is not None:
            self.readCmdLineArgs(cmd_line_args)
    
    def doForecast(self):
        self.loadForecastConfiguration()
        if self.rundir is None:
            self.setRunID()
        self.buildTasks()
        self.prepareWorkspace()
        self.buildTaskRunners()
        self.populateInitialMetadata()
        # Write down first metadata, just in case of possible fail of the "run" method
        self.writeRunMetadata()
        self.run()
        self.addTimingDataToMetadata()
        # Write down updated metadata after the run
        self.writeRunMetadata()

    @property
    def cl_task(self):
        return self.tasks['Angular']

    @property
    def deriv_task(self):
        return self.tasks['Derivative']

    @property
    def fisher_task(self):
        return self.tasks['Fisher']

    def readCmdLineArgs(self, cmd_line_args: "Dict"):
        self.args = JobSubmitterCommandLineArgs(cmd_line_args)
        self.input_data_dir = Path(self.args['input_data_dir'])

    def loadForecastConfiguration(self):
        if self.rundir is not None:
            self.workspace = WorkSpace(self.rundir)
            self.forecast_config_file = self.workspace.getForecastConfigFilePath()
        else:
            self.forecast_config_file = self.args['forecast_config']

        if not isinstance(self.forecast_config_file, Path):
            raise TypeError(f"{self.forecast_config_file} of type {type(self.forecast_config_file)} is not a valid "
                            f"forecast config file")
        else:
            if not self.forecast_config_file.is_file():
                raise FileNotFoundError(self.forecast_config_file)

        self.forecast_config = fcfg.ForecastConfig(input_file=self.forecast_config_file,
                                                   input_data_dir=self.input_data_dir)

        self.forecast_config.loadPhysicalParametersFromJSONConfig()

    def setRunID(self):
        if self.args.run_id is not None:
            self.run_id = f'{self.args.run_id}_{VERSION}'
        else:
            fcfg_id = self.forecast_config.getConfigID()
            if fcfg_id is not None:
                self.run_id = f"{fcfg_id}_{VERSION}"
            else:
                self.run_id = f"run_seyfert_{VERSION}"

    def buildTasks(self):
        self.tasks = {
            name: self.createSingleTask(name, self.args)
            for name in ["PowerSpectrum", "Angular", "Derivative", "Fisher"]
        }

    def prepareWorkspace(self):
        if self.rundir is not None and self.rundir.is_dir():
            logger.info(f"directory {self.rundir} already exists, using it as rundir")
        else:
            logger.info("Creating rundir")
            self.createRundir()
            logger.info("Creating input files dir")
            self.createInputFilesDir()

    def createRundir(self) -> "None":
        self.rundir = (self.base_workdir / self.run_id).resolve()
        if self.args.test:
            logger.info('Test mode, not imposing timestamp suffix to rundir')
            if self.rundir.exists():
                logger.info(f'Test mode, removing {self.rundir}')
                shutil.rmtree(self.rundir)
            logger.info(f'Creating rundir {self.rundir}')
            self.rundir.mkdir()
        else:
            timestamp = fm.datetime_str_format(datetime.datetime.now())
            self.rundir = self.rundir.with_name(f'{self.rundir.name}_{timestamp}')
            logger.info(f'Creating rundir {self.rundir}')
            self.rundir.mkdir()

        self.workspace = WorkSpace(self.rundir)

    @staticmethod
    def createSingleTask(name: "str", cmd_line_args: "Dict") -> "Task":
        key_base = name.lower()
        is_present = cmd_line_args[key_base]
        cfg_file = Path(cmd_line_args[f'{key_base}_config'])
        ext_dir = cmd_line_args[f'{key_base}_dir']

        return Task(name, is_present, cfg_file, ext_dir)

    def createInputFilesDir(self):
        src_input_files = {'forecast': self.forecast_config_file}
        for name, task in self.tasks.items():
            src_input_files[name] = Path(task.input_file)

        self.workspace.createInputFilesDir(src_input_files=src_input_files,
                                           phys_pars=self.forecast_config.phys_pars,
                                           input_data_dir=self.input_data_dir)

    def buildTaskRunners(self):
        ext_dirs = {}
        for name, task in self.tasks.items():
            if task.is_present:
                job_submitter_type = f'{name}TaskRunner'
                logger.info(f'Task {name} is present, building {job_submitter_type}')
                job_submitter = getattr(this_module, job_submitter_type)(workspace=self.workspace, args=self.args,
                                                                         json_input=task.input_file,
                                                                         forecast_config=self.forecast_config)
                self.submitters_dict[name] = job_submitter
                logger.info(f'Creating directory structure for {job_submitter_type}')
                self.submitters_dict[name].createJobDirStructure()
            else:
                logger.info(f'Task {name} is absent')
                if task.external_dir is not None:
                    ext_dirs[name] = task.external_dir

        self.workspace.symlinkToExternalDirs(ext_dirs=ext_dirs, src_relative_to_rundir=True)
    
    def populateInitialMetadata(self):
        self.metadata.update({
            "run_id": self.run_id,
            "executor": os.getenv("USER", ""),
            "host_and_platform": gu.get_execution_host(),
            "code_version": VERSION,
            "external_dirs": self.external_dirs,
        })

    def computeDensities(self):
        logger.info("Computing redshift densities")
        if self.tasks['PowerSpectrum'].is_present and self.tasks['Angular'].is_present:
            if self.args.execution != 'no':
                pmm_cfg = self.workspace.getTaskJSONConfiguration('PowerSpectrum')
                z_grid = pmm_cfg.z_grid
            else:
                logger.info('Skipping density computation')
                return
        else:
            cosmo = Cosmology.fromHDF5(self.workspace.getPowerSpectrumFile(dvar='central', step=0))
            z_grid = cosmo.z_grid

        densities = cl_core.compute_densities(probe_configs=self.forecast_config.probe_configs, z_grid=z_grid)
        for name in densities:
            densities[name].saveToHDF5(self.workspace.niz_file, root=name)

    def computeDeltaCls(self):
        logger.info("Computing delta cls")
        if self.args.execution != 'no':
            delta_cls = cl_core.compute_delta_cls(self.workspace)
            delta_cls.saveToHDF5(self.workspace.delta_cls_file)
        else:
            logger.info('Execution is "no", skipping DeltaCls computation')

    def run(self):
        self.bsub.setOptions(queue=self.args.queue, memory_MB=self.args.memory, n_cores=self.args.n_cores_per_job)
        self.start_datetime = datetime.datetime.now()
        if self.tasks['Angular'].is_present:
            self.computeDensities()
        for name, task in self.tasks.items():
            if task.is_present:
                logger.info(f"running {name}")
                self.submitters_dict[name].run()
                if name == "Angular":
                    self.computeDeltaCls()

        if self.args.execution != 'no' and self.tasks['Fisher'].is_present:
            self.evaluateFinalResults()

        self.end_datetime = datetime.datetime.now()

    def evaluateFinalResults(self):
        if self.args.overwrite_results:
            results_dir = self.workspace.final_results_dir
            if results_dir.exists():
                logger.warning("Overwriting results")
                shutil.rmtree(results_dir)

        logger.info("Evaluating final results")
        # Write GCsp(Pk) fishers into base fishers dir
        self.writeGCspPkFishersTo(self.workspace.base_fishers_dir)
        # Call evaluate_final_results script
        cli_args = {
            "-ex": self.args.execution, "-i": self.rundir, "-q": "long", "-ncj": 1, "-mem": 3000,
            "-ow": self.args.overwrite_results, "-wd": self.workspace.run_dir
        }
        res_cmd = f'evaluate_final_results {" ".join(f"{key} {value}" for key, value in cli_args.items())}'
        sp.run(res_cmd, shell=True)

    def writeGCspPkFishersTo(self, outdir: "TPathLike"):
        outdir = Path(outdir)
        logger.info(f"Writing GCsp Pk IST fishers into {outdir}")
        for cosmology in ["w0_wa_CDM"]:
            file = fsu.get_ist_gcsp_pk_fisher_file(scenario=self.forecast_config.scenario, cosmology=cosmology)
            pk_fisher = FisherMatrix.from_ISTfile(file=file)
            pk_fisher.writeToFile(outfile=outdir / f"fisher_IST_gcsp_pk_{cosmology}.hdf5")
    
    def addTimingDataToMetadata(self):
        if self.end_datetime is not None:
            duration = fm.string_time_format((self.end_datetime - self.start_datetime).seconds)
            self.metadata.update({
                "start_time": fm.datetime_str_format(self.start_datetime, hour_sep=':'),
                "end_time": fm.datetime_str_format(self.end_datetime, hour_sep=':'),
                "duration": duration
            })
    
    def writeRunMetadata(self):
        with open(self.workspace.metadata_file, mode="w") as jsf:
            json.dump(self.metadata, jsf, indent=4)


def set_num_threads(num_threads):
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(num_threads)
