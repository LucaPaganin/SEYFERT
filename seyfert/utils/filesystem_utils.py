from typing import Tuple, Dict, Union
import re
import json
import os
from pathlib import Path
import logging

from seyfert import SEYFERT_PATH

logger = logging.getLogger(__name__)


def repo_path():
    return Path(SEYFERT_PATH).parent


def config_files_dir():
    return Path(SEYFERT_PATH) / 'data/config_files'


def default_seyfert_input_data_dir() -> "Path":
    env_var = "SEYFERT_INPUT_DATA"
    input_data_dir = os.getenv(env_var)
    if isinstance(input_data_dir, str):
        input_data_dir = Path(input_data_dir)
        if input_data_dir.is_dir():
            logger.info(f"Found existing input dir {input_data_dir}")
        else:
            logger.warning(f"env var {env_var} is set to {input_data_dir}, but this directory does not exist. "
                           f"You should create it")
    else:
        input_data_dir = Path.home() / "spectrophoto/input_data"

    if not input_data_dir.is_dir():
        raise NotADirectoryError(f"Retrieved input data dir {input_data_dir} is not a directory!")

    return input_data_dir


def default_data_dir():
    return Path(SEYFERT_PATH) / 'data/input_data'


def the_test_data_dir():
    return Path(SEYFERT_PATH) / 'data/test_data'


def tables_aux_files_dir():
    return Path(SEYFERT_PATH) / 'data/tables'


def fisher_aux_files_dir():
    return Path(SEYFERT_PATH) / 'data/fishers'


def plots_aux_files_dir():
    return Path(SEYFERT_PATH) / 'data/plots_defs'


def get_input_files_dir_for_rundir(rundir: "Path") -> "Path":
    return rundir / 'input_files_dir'


def get_input_files_JSON_for_rundir(rundir: "Path") -> "Path":
    return rundir / 'input_files_dict.json'


def basic_forecast_config() -> "Path":
    return config_files_dir() / "basic_forecast_config.json"


def get_input_files_dict_for_rundir(rundir: "Path") -> "Dict":
    with open(get_input_files_JSON_for_rundir(rundir), 'r') as jsf:
        data = json.load(jsf)
    return data


def get_forecast_config_file_from_rundir(rundir: "Path") -> "Path":
    forecast_config_file = get_input_files_dict_for_rundir(rundir)['forecast']
    return rundir / forecast_config_file


def get_params_file_from_rundir(rundir: "Path") -> "Path":
    params_file = get_input_files_dict_for_rundir(rundir)['params']
    return rundir / params_file


def get_forecast_input_data_dir_from_rundir(rundir: "Path") -> "Path":
    return rundir / get_input_files_dict_for_rundir(rundir)['input_data_dir']


def get_dvar_step_from_cosmology_jobdir_name(dirname: str):
    dirname_regex = re.compile('dvar_(?P<dvar>[a-zA-Z0-9]+)_step(?P<sign>_[pm])*_(?P<step>[0-9])')
    match = dirname_regex.search(dirname)
    try:
        dvar = match.group('dvar')
        step = int(match.group('step'))
        sign = match.group('sign')
        if sign is not None:
            sign = sign.replace('_', '')
            if sign == 'm':
                step = -step
    except AttributeError:
        raise Exception(f'Cannot extract dvar step from dirname {dirname}')
    return dvar, step


def get_cosmology_jobdir_name(dvar: str, step: int) -> str:
    if step == 0:
        step_str = "0"
    else:
        step_str = f"m_{int(abs(step))}" if step < 0 else f"p_{int(abs(step))}"
    job_dir_name = f'dvar_{dvar}_step_{step_str}'
    return job_dir_name


def get_dvar_from_derivative_jobdir_name(dirname: str) -> str:
    return dirname


def get_derivative_jobdir_name_from_dvar(dvar: str) -> "str":
    return dvar


def get_dvar1_dvar2_from_fisher_jobdir_name(dirname: str) -> "Tuple[str]":
    dirname_regex = re.compile('(?P<dvar1>[a-zA-Z0-9]+)__(?P<dvar2>[a-zA-Z0-9]+)')
    match = dirname_regex.match(dirname)
    try:
        dvar1 = match.group('dvar1')
        dvar2 = match.group('dvar2')
    except AttributeError:
        raise Exception(f'Cannot extract dvar1, dvar2 from dirname {dirname}')
    return dvar1, dvar2


def get_fisher_jobdir_name_from_dvar1_dvar2(dvar1: str, dvar2: str) -> "str":
    return f'{dvar1}__{dvar2}'


def get_file_from_dir_with_pattern(directory: Path = None, pattern: str = None):
    candidates = list(directory.glob(pattern))
    if len(candidates) != 1:
        raise Exception(f'Found {len(candidates)} files with pattern {pattern} into '
                        f'directory {directory}')

    return candidates[0]


def get_fisher_dirname_from_switches(prefix="fishers_with_pk",
                                     wlph: bool = True, wlsp: bool = True, phsp: bool = True) -> "str":
    switches = {"WL-GCph": wlph, "WL-GCsp": wlsp, "GCph-GCsp": phsp}
    switches_string = "__".join(f"der_{xc_name}_{switch}" for xc_name, switch in switches.items())

    return f"{prefix}__{switches_string}"


def get_ist_gcsp_pk_fisher_file(scenario: "str", cosmology: "str" = "w0_wa_CDM") -> "Path":
    fisher_gcsp_pk_file = default_data_dir() / "ist_fishers" / cosmology / f"EuclidISTF_GCsp_w0wa_flat_{scenario}.txt"
    if not fisher_gcsp_pk_file.is_file():
        raise FileNotFoundError(fisher_gcsp_pk_file)

    return fisher_gcsp_pk_file
