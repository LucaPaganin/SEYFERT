import configparser
import datetime
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Union, Set, Callable, Iterable
import getpass
import socket
import platform

import numpy as np

from seyfert import PROBE_LONG_TO_ALIAS
from seyfert.utils.formatters import str_to_bool

logger = logging.getLogger(__name__)


_probe_short_to_long = {
    'wl': 'Lensing',
    'v': 'Void',
    'gcph': 'PhotometricGalaxy',
    'gcsp': 'SpectroscopicGalaxy'
}

_probe_long_to_short = {v: k for k, v in _probe_short_to_long.items()}
_probe_fullnames_to_aliases = PROBE_LONG_TO_ALIAS.copy()
_probe_aliases_to_fullnames = {value: key for key, value in _probe_fullnames_to_aliases.items()}

scenarios_dict = {
    "optm": "optimistic", "optimistic": "optimistic",
    "pess": "pessimistic", "pessimistic": "pessimistic",
}


def probe_alias_to_fullname(prb_alias: "str") -> "str":
    return _probe_aliases_to_fullnames[prb_alias]


def probe_fullname_to_alias(prb_fullname: "str") -> "str":
    return _probe_fullnames_to_aliases[prb_fullname]


def convert_keyval_params_type(params: Dict):
    bool_regex = re.compile('^([Tt]rue|[Ff]alse)$')
    int_regex = re.compile(r'^[0-9]+$')
    for key, value in params.items():
        if isinstance(value, str):
            bool_match = bool_regex.match(value)
            int_match = int_regex.match(value)
            if bool_match:
                params[key] = str_to_bool(value)
            elif int_match:
                params[key] = int(value)
    return params


def get_defaults_from_inifile(inifile: "Union[str, Path]") -> Dict:
    config = configparser.ConfigParser()
    config.read(inifile)
    defaults = convert_keyval_params_type(dict(config['defaults']))
    return defaults


def configure_logger(logger: "logging.Logger", logfile: "Union[str, Path]" = None):
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s: %(message)s'
    log_formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    if logfile is not None:
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)


def _log_script_separator(script_name: "str", main_logger: "logging.Logger", what: "str"):
    sep = "#" * 80
    main_logger.info(f"{sep}")
    main_logger.info(f"#")
    main_logger.info(f"# {what} {script_name} main method")
    main_logger.info(f"# username: {getpass.getuser()}")
    main_logger.info(f"# hostname: {socket.gethostname()}")
    main_logger.info(f"# system info:")
    for key, value in platform.uname()._asdict().items():
        main_logger.info(f"# - {key}: {value}")
    main_logger.info(f"# python version: {platform.python_version()}")
    main_logger.info(f"#")
    main_logger.info(f"{sep}")


def print_script_login(script_name: "str", main_logger: "logging.Logger"):
    _log_script_separator(script_name, main_logger, what="ENTERING")


def print_script_logout(script_name: "str", main_logger: "logging.Logger"):
    _log_script_separator(script_name, main_logger, what="FINALIZING")


def getMainLogger(logfile: "Union[str, Path]" = None):
    lgr = logging.getLogger()
    configure_logger(lgr, logfile=logfile)

    return lgr


def get_probes_combination_key(obs1: "str", obs2: "str") -> "str":
    return f'{obs1}_{obs2}'


def get_probes_from_comb_key(obs_comb_key: "str") -> "List[str]":
    return obs_comb_key.split('_')


def reverse_probes_comb_key(obs_comb_key: "str") -> "str":
    p1, p2 = get_probes_from_comb_key(obs_comb_key)
    return get_probes_combination_key(p2, p1)


def get_probe_comb_from_shortcut(short_p1: "str", short_p2: "str") -> "str":
    try:
        full_p1 = _probe_short_to_long[short_p1]
        full_p2 = _probe_short_to_long[short_p2]
    except KeyError:
        raise KeyError(f'Invalid shortcuts: {short_p1} and {short_p2}, must be both '
                       f'one of: {", ".join(_probe_short_to_long.keys())}')
    return get_probes_combination_key(full_p1, full_p2)


def compare_objects(obj1: "object", obj2: "object", exclude_attrs: "Set") -> bool:
    conds = []
    for key in obj1.__dict__:
        if key not in exclude_attrs:
            try:
                value1 = getattr(obj1, key)
                value2 = getattr(obj2, key)
            except AttributeError:
                raise AttributeError(f'obj1 {obj1} and/or {obj2} have no attribute {key}')
            if isinstance(value1, np.ndarray) or isinstance(value2, np.ndarray):
                cond = np.all(value1 == value2)
            else:
                cond = value1 == value2
            conds.append(cond)
    result = all(conds)
    if not result:
        cond_dict = {key: cond for key, cond in zip(obj1.__dict__, conds)}
        print(f'Object type: {type(obj1)}')
        for key, cond in cond_dict.items():
            if not cond:
                print(f'{key}: {obj1.__dict__[key]} vs {obj2.__dict__[key]}')
                print()
    return result


def map_nested_dict(dct: "Dict", func: "Callable") -> "Dict":
    if isinstance(dct, dict):
        return {k: map_nested_dict(v, func) for k, v in dct.items()}
    else:
        return func(dct)


def get_execution_host() -> "List[str]":
    host = os.getenv("HOSTNAME")
    pltfrm = sys.platform
    if host is None:
        if pltfrm == 'linux':
            proc = subprocess.run('hostname -I', shell=True, stdout=subprocess.PIPE)
            ip_addresses = set(proc.stdout.decode('utf-8').strip().split()) - {'127.0.0.1'}
            host = " ".join(ip_addresses)
        elif pltfrm == 'darwin':
            proc = subprocess.run('hostname', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            host = proc.stdout.decode('utf-8').strip()
        else:
            logger.warning(f'unrecognized platform {pltfrm}')

    return [host, pltfrm]


def split_run_id(run_id: "str") -> "Tuple[str, str, datetime.datetime]":
    regex = re.compile(r"^(.*)_(\d\.\d\.\d(?:dev)?)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")
    match = regex.match(run_id)
    if not match:
        run_name = run_id
        version = None
        timestamp = None
    else:
        run_name, version, timestamp = match.groups()
        timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H-%M-%S')

    return run_name, version, timestamp


def replace_fullnames_with_aliases(s: "str") -> "str":
    result = s
    for name, alias in _probe_fullnames_to_aliases.items():
        result = result.replace(name, alias)

    return result


def subset_dict(d: "Dict", subset_keys: "Iterable[str]") -> "Dict":
    return {
        key: value for key, value in d.items() if key in subset_keys
    }


def multiple_regex_replace(s, sub_dict):
    regex = re.compile("(%s)" % "|".join(map(re.escape, sub_dict.keys())))

    return regex.sub(lambda match: sub_dict[match.string[match.start():match.end()]], s)


def get_scenario_string(s: "str") -> "str":
    return scenarios_dict[s]
