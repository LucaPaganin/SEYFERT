import argparse
from typing import Dict, Union, List
from pathlib import Path
import re
import json
from argparse import Namespace
from seyfert.base_structs.generic_dict import DictLike
from seyfert.utils import filesystem_utils as fsu
import logging

logger = logging.getLogger(__name__)


class GenericCommandLineArgs(DictLike):
    def __init__(self):
        super(GenericCommandLineArgs, self).__init__()

    def readArgs(self, args: "Union[Dict, Namespace]"):
        if isinstance(args, dict):
            self.update(args)
        elif isinstance(args, Namespace):
            self.update(vars(args))
        else:
            raise TypeError(f"args must be dict or argparse.Namespace, not {type(args)}")

        self.castArgTypes()

    def castArgTypes(self):
        pass


class JobSubmitterCommandLineArgs(GenericCommandLineArgs):
    def __init__(self, args: "Union[Dict, Namespace]" = None):
        super(JobSubmitterCommandLineArgs, self).__init__()
        self.run_id = None
        self.forecast_config = None
        self.powerspectrum_config = None
        self.angular_config = None
        self.derivative_config = None
        self.fisher_config = None
        self.input_data_dir = None
        self.powerspectrum_dir = None
        self.angular_dir = None
        self.derivative_dir = None
        self.fisher_dir = None
        self.execution = None
        self.queue = None
        self.memory = None
        self.n_cores_per_job = None
        self.verbose = None
        self.test = None
        self.doctor = None
        self.ignore_errfiles = None
        self.powerspectrum = None
        self.angular = None
        self.derivative = None
        self.fisher = None
        self.overwrite_results = None
        self.keep_cl_variations = None

        if args is not None:
            self.readArgs(args)

    def castArgTypes(self):
        path_like_regex = re.compile("_(config|dir)")
        for name, value in self.items():
            if path_like_regex.search(name) and value is not None:
                self[name] = Path(value)


class MainProgramCommandLineArgs(GenericCommandLineArgs):
    def __init__(self, args: "Union[Dict, Namespace]" = None):
        super(MainProgramCommandLineArgs, self).__init__()
        self.rundir = None
        self.workdir = None
        self.logfile = None
        self.main_config = None
        self.standalone = None
        self.forecast_config = None
        self.input_data_dir = None

        if args is not None:
            self.readArgs(args)

    def castArgTypes(self):
        path_like_regex = re.compile(r"(config|dir)$")
        for name, value in self.items():
            if path_like_regex.search(name) and value is not None:
                self[name] = Path(value)


class ClCLIArgs(MainProgramCommandLineArgs):
    def __init__(self, args: "Union[Dict, Namespace]" = None):
        super(ClCLIArgs, self).__init__()
        self.power_spectrum_dir = None
        self.dvar = None
        self.step = None
        self.niz_file = None

        if args is not None:
            self.readArgs(args)

    def castArgTypes(self):
        super(ClCLIArgs, self).castArgTypes()
        if self.step is not None:
            self.step = int(self.step)


class ClDerivativeCLIArgs(MainProgramCommandLineArgs):
    def __init__(self, args: "Union[Dict, Namespace]" = None):
        super(ClDerivativeCLIArgs, self).__init__()
        self.angular_dir = None
        self.dvar = None

        if args is not None:
            self.readArgs(args)


class FisherCLIArgs(MainProgramCommandLineArgs):
    def __init__(self, args: "Union[Dict, Namespace]" = None):
        super().__init__()
        self.fisher_names = None
        self.shot_noise_file = None
        self.outdir = None

        if args is not None:
            self.readArgs(args)


def main_program_options():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rundir", "-rd", help="Path to the run directory")
    parser.add_argument("--workdir", "-wd", help="Path to the working directory")
    parser.add_argument("--logfile", "-lg", help="Path to the logfile")
    parser.add_argument("--forecast_config", "-fcfg", help="Path to forecast config XML (for standalone mode)")
    parser.add_argument("--input_data_dir", "-i", help="Path to input data directory (for standalone mode)")
    return parser


def batch_options():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--n_cores_per_job', '-ncj', type=int, default=1,
                        help='OPTIONAL: number of cores per job to ask for')
    parser.add_argument('--memory', '-mem', type=int, default=2048,
                        help="OPTIONAL: RAM memory (expressed in MB) to request for batch mode. Default is 2048 MB")
    parser.add_argument('--queue', '-q', default="long",
                        help='OPTIONAL: Name of the queue for bsub; this option is used only when execution is '
                             '"batch". Viable options are: long, medium')

    return parser


def main_job_batch_options():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_threads', '-nt', type=int, default=None,
                        help='OPTIONAL: Number of threads to set to NUMEXPR_NUM_THREADS')

    return parser


def main_submitter_batch_options() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(add_help=False, parents=[batch_options()])

    parser.add_argument("--execution", "-ex", required=True, default=None,
                        help='REQUIRED: String to decide how to run the jobs. Viable possibilities: "no" for not '
                             'executing, "interactive" to run one job after another, "batch" for batch submission')

    return parser


def forecast_task_flags() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--powerspectrum", "-pmm", nargs='?', default=False, const=True,
                        help='OPTIONAL: Flag for computing matter power spectrum')
    parser.add_argument("--angular", "-cl", nargs='?', default=False, const=True,
                        help='OPTIONAL: Flag for computing angular coefficients')
    parser.add_argument("--derivative", "-der", nargs='?', default=False, const=True,
                        help='OPTIONAL: Flag for computing angular coefficients derivatives')
    parser.add_argument("--fisher", "-fish", nargs='?', default=False, const=True,
                        help='OPTIONAL: Flag for computing fisher matrix elements')

    return parser


def forecast_task_configs_options() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--powerspectrum_config", "-pmm_cfg", type=Path,
                        required=False, default=fsu.config_files_dir() / 'power_spectrum_config.json',
                        help="OPTIONAL: Path to the json file containing the power spectrum configuration")
    parser.add_argument("--angular_config", "-cl_cfg", type=Path,
                        required=False, default=fsu.config_files_dir() / 'angular_config.json',
                        help="OPTIONAL: Path to the json file containing the angular coefficients configuration")
    parser.add_argument("--derivative_config", "-der_cfg", type=Path,
                        required=False, default=fsu.config_files_dir() / 'derivative_config.json',
                        help="OPTIONAL: Path to the json file containing the Cl derivatives configuration")
    parser.add_argument("--fisher_config", "-fish_cfg", type=Path,
                        required=False, default=fsu.config_files_dir() / 'fisher_config.json',
                        help="OPTIONAL: Path to the json file containing the Fisher matrix configuration")
    parser.add_argument("--results_config", "-res_cfg", type=Path,
                        required=False, default=fsu.config_files_dir() / 'results_config.json',
                        help="OPTIONAL: Path to the json file containing the Fisher matrix configuration")

    return parser


def forecast_dirs_paths_options() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--powerspectrum_dir", "-pmm_dir", type=Path,
                        required=False, default=None,
                        help="OPTIONAL: Path to the power spectra directory")
    parser.add_argument("--angular_dir", "-cl_dir", type=Path,
                        required=False, default=None,
                        help="OPTIONAL: Path to the angular coefficients directory")
    parser.add_argument("--derivative_dir", "-der_dir", type=Path,
                        required=False, default=None,
                        help="OPTIONAL: Path to the angular coefficients derivatives directory")
    parser.add_argument("--fisher_dir", "-fish_elems_dir", type=Path,
                        required=False, default=None,
                        help="OPTIONAL: Path to the fisher matrix elements directory")

    return parser


def get_paths_from_files_list(paths_list: "List[Union[str, Path]]") -> "List[Path]":
    resulting_paths = []

    if all(Path(path).suffix == '.txt' for path in paths_list):
        logger.info("paths list contains only txt files")
        for file in paths_list:
            resulting_paths.extend([Path(line.strip()) for line in Path(file).read_text().splitlines()])
    else:
        logger.info("neither all json nor all txt, returning the entered list of paths")
        resulting_paths = [Path(path) for path in paths_list]

    return resulting_paths


def read_json_txt_list_file(filepath: "Union[str, Path]") -> "List[str]":
    filepath = Path(filepath)
    extension = filepath.suffix
    if extension == '.json':
        with filepath.open() as jsf:
            result = json.load(jsf)
    elif extension in {'.txt', '.dat'}:
        result = [line.strip() for line in filepath.read_text().splitlines()]
    else:
        raise Exception(f"Unrecognized file extension {extension}")

    return result
