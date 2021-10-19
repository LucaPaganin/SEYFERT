import re
from typing import Union, Dict, List
from pathlib import Path
import shutil
import argparse
import logging
import json
import time
import subprocess

from seyfert.utils import general_utils as gu
from seyfert.utils import filesystem_utils as fsu
from seyfert.utils import formatters as fm
from seyfert.utils.workspace import WorkSpace

logger = logging.getLogger(__name__)


def create_config_dir_tree(workdir: "Union[str, Path]", args: "argparse.Namespace"):
    cfg_dir = Path(workdir) / 'config'
    logger.info("Creating config dir")
    cfg_dir.mkdir(exist_ok=True)
    config_files = {
        'fcfg': None,
        'pmm_cfg': args.powerspectrum_config,
        'cl_cfg': args.angular_config,
        'der_cfg': args.derivative_config,
        'fish_cfg': args.fisher_config,
        'res_cfg': args.results_config,
    }
    for name, cfg_file in config_files.items():
        subdir = cfg_dir / name
        logger.info(f'Creating subdir {subdir}')
        subdir.mkdir(exist_ok=True)
        if cfg_file is not None:
            shutil.copy(cfg_file, subdir / cfg_file.name)

    ext_dirs = {
        'pmm_dir': None,
        'cl_dir': None,
        'der_dir': None
    }

    with open(cfg_dir / 'external_dirs.json', mode='w') as jsf:
        json.dump(ext_dirs, jsf, indent=2)


def get_configs(cfg_dir: "Union[str, Path]") -> "Dict":
    configs = {}
    for subdir in filter(lambda x: x.is_dir(), Path(cfg_dir).iterdir()):
        name = subdir.name
        content = list(subdir.iterdir())
        if name == 'fcfg':
            configs['fcfg'] = list(subdir.glob("*.json"))
        elif name == 'fish_cfg' or name == 'cl_cfg':
            configs[name] = list(subdir.glob("*.json"))
        else:
            configs[name] = content[0]

    with open(cfg_dir / 'external_dirs.json', mode='r') as jsf:
        configs['ext_dirs'] = json.load(jsf)

    return configs


def get_pmm_dirs(args: "argparse.Namespace") -> "List":
    if args.angular:
        pmms = args.powerspectra
        if pmms is None:
            raise ValueError("powerspectra '-pmms' argument is required when computing cls")
        if not pmms.exists():
            raise FileNotFoundError(pmms)
        if pmms.is_dir():
            pmm_dirs = [pmms]
        elif pmms.is_file():
            if not pmms.name.endswith('txt'):
                raise Exception(f"Invalid powerspectra file format {pmms.name}, it must have txt extension")
            pmm_dirs = [Path(line) for line in pmms.read_text().splitlines()]
            for pmm_dir in pmm_dirs:
                if not pmm_dir.exists():
                    raise FileNotFoundError(pmm_dir)
        else:
            raise Exception("pmms argument could be txt file of directory")
    else:
        pmm_dirs = [None]

    return pmm_dirs


def sort_forecast_config_files(files: "List[Path]"):
    regex = re.compile("([0-9]+)_sp_bins")
    if all([regex.search(f.name) for f in files]):
        files.sort(key=lambda p: int(regex.search(p.name).groups()[0]))


def do_multiple_new_seyfert_runs(pmm_dirs: "List[Union[str, Path]]", configs: "Dict",
                                 args: "argparse.Namespace") -> "List[str]":
    t0 = time.time()
    fcfgs = configs['fcfg']
    fish_cfgs = configs['fish_cfg']
    cl_cfgs = configs['cl_cfg']
    if not isinstance(pmm_dirs, list):
        raise TypeError(f"power spectra dirs must be list, not {type(pmm_dirs)}")
    n_fcfgs = len(fcfgs)
    n_pmms = len(pmm_dirs)
    n_fish_cfgs = len(fish_cfgs)
    n_cl_cfgs = len(cl_cfgs)
    if n_fcfgs == 0:
        raise Exception("There must be at least a forecast config xml file into config/fcfg/")
    if n_pmms == 0 and args.angular:
        raise Exception("There must be at least a power spectrum directory to use when computing cls")
    if n_fish_cfgs == 0:
        raise Exception("There must be at least one fisher config to use")
    if n_cl_cfgs == 0:
        raise Exception("There must be at least one cl config to use")

    use_fisher_as_id = n_fish_cfgs > 1
    use_angular_as_id = n_cl_cfgs > 1
    sort_forecast_config_files(fcfgs)
    logger.info("Order of execution:")
    for i, fcfg_file in enumerate(fcfgs):
        logger.info(f"{i+1}/{n_fcfgs}: {Path(fcfg_file).name}")

    run_ids = []
    idx_main = 1
    n_runs = n_pmms * n_fcfgs * n_fish_cfgs * n_cl_cfgs
    for idx_fcfg, fcfg in enumerate(fcfgs):
        for idx_fish, fish_cfg in enumerate(fish_cfgs):
            for idx_cl, cl_cfg in enumerate(cl_cfgs):
                for idx_pmm, pmm_dir in enumerate(pmm_dirs):
                    logger.info(f"Running forecast no. {idx_main}/{n_runs}")
                    logger.info(f"Forecast configuration {idx_fcfg + 1}/{n_fcfgs}: {fcfg.name}")
                    logger.info(f"Angular config {idx_cl + 1}/{n_cl_cfgs}: {cl_cfg}")
                    logger.info(f"Fisher config {idx_fish + 1}/{n_fish_cfgs}: {fish_cfg}")
                    keyval_args = {
                        'fcfg': fcfg, 'cl_cfg': cl_cfg, 'fish_cfg': fish_cfg
                    }
                    update_key_val_args_from_cli_args(args, keyval_args)
                    if pmm_dir is not None:
                        logger.info(f"Power spectra {idx_pmm + 1}/{n_pmms}: {pmm_dir}")
                        keyval_args['pmm_dir'] = pmm_dir
                    keyval_args.update({
                        key: configs[key] for key in ['pmm_cfg', 'der_cfg', 'res_cfg']
                    })
                    keyval_args.update({
                        dir_key: dir_path for dir_key, dir_path in configs['ext_dirs'].items() if dir_path is not None
                    })

                    run_id = do_new_seyfert_run(args=args, keyval_args=keyval_args,
                                                use_fisher_as_id=use_fisher_as_id,
                                                use_angular_as_id=use_angular_as_id)
                    run_ids.append(run_id)
                    logger.info("Done")
                    logger.info(f'\n{"-" * 115}\n')
                    idx_main += 1

    tf = time.time()
    logger.info(f"Done. Total elapsed time {fm.string_time_format(tf - t0)}")

    return run_ids


def do_new_seyfert_run(args: "argparse.Namespace", keyval_args: "Dict",
                       use_fisher_as_id=True, use_angular_as_id=True) -> "str":
    input_data_dir = fsu.default_seyfert_input_data_dir()

    fcfg = Path(keyval_args['fcfg'])
    fcfg_id, _, _ = gu.split_run_id(fcfg.stem)

    run_id = f'{fcfg_id}'
    if use_fisher_as_id:
        fish_id = Path(keyval_args['fish_cfg']).stem
        logger.info(f"Appending fisher id {fish_id} to run_id")
        run_id += f"_{fish_id}"
    if use_angular_as_id:
        cl_id = Path(keyval_args['cl_cfg']).stem
        logger.info(f"Appending cl id {cl_id} to run_id")
        run_id += f"_{cl_id}"

    if 'pmm_dir' in keyval_args:
        pmm_dir = Path(keyval_args['pmm_dir'])
        pmm_dirname = pmm_dir.parent.name if pmm_dir.name == 'PowerSpectrum' else pmm_dir.name
        pmm_id, _, _ = gu.split_run_id(pmm_dirname)
        logger.info(f"Appending pmm id {pmm_id} to run_id")
        run_id += f'_{pmm_id}'

    keyval_args['id'] = run_id
    keyval_args['i'] = input_data_dir

    cmd = build_seyfert_cmd(args, keyval_args)
    logger.info(f"RUN ID: {run_id}")
    logger.info(f"Running command: \n{cmd}")
    execution = keyval_args['ex']
    if execution == 'no':
        logger.info("execution is 'no', skipping")
    else:
        child_proc = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        if child_proc.stderr:
            logger.warning(f"Run with fcfg {keyval_args['fcfg']} produced a non-empty standard error")
            logger.warning(f"{child_proc.stderr.decode('utf-8')}")
            logger.warning("Going on...")

    return run_id


def build_seyfert_cmd(args: "argparse.Namespace", keyval_args: "Dict") -> "str":
    kwargs_str = " ".join([f"-{key} {value}" for key, value in keyval_args.items()])
    add_flags_str = " ".join([f"--{flag}" for flag in args.additional_flags])
    task_flags_str = " ".join([
        f"--{task_flag_key}" for task_flag_key in ["angular", "derivative", "fisher"]
        if getattr(args, task_flag_key)
    ])

    return f'job_submitter {task_flags_str} {kwargs_str} {add_flags_str}'


def do_multiple_seyfert_runs_on_existing_rundirs(args: "argparse.Namespace", input_rundirs: "List[Union[str, Path]]"):
    for rundir in input_rundirs:
        logger.info(f"Running seyfert on {rundir}")
        run_seyfert_on_existing_rundir(args, rundir)
        logger.info(f'\n{"-" * 115}\n')


def run_seyfert_on_existing_rundir(args: "argparse.Namespace", rundir: "Union[str, Path]"):
    ws = WorkSpace(rundir)
    input_files = ws.theInputFilesDict()
    keyval_args = {
        'rd': ws.run_dir,
        'i': fsu.default_seyfert_input_data_dir(),
        'fcfg': ws.run_dir / input_files['forecast'],
        'pmm_cfg': ws.run_dir / input_files['PowerSpectrum'],
        'cl_cfg': ws.run_dir / input_files['Angular'],
        'der_cfg': ws.run_dir / input_files['Derivative'],
        'fish_cfg': ws.run_dir / input_files['Fisher']
    }
    update_key_val_args_from_cli_args(args, keyval_args)
    cmd = build_seyfert_cmd(args, keyval_args)
    logger.info(f"Running cmd {cmd}")
    subprocess.run(cmd, shell=True)


def update_key_val_args_from_cli_args(args: "argparse.Namespace", keyval_args: "Dict"):
    keyval_args.update({
        'ex': args.execution, 'q': args.queue, 'ncj': args.n_cores_per_job, 'mem': args.memory
    })

    if args.ini_file is not None:
        keyval_args['ini'] = args.ini_file


def collect_rundirs(workdir: "Path", run_ids: "List[str]") -> "List[Path]":
    rundirs = []
    for x in workdir.iterdir():
        if x.is_dir():
            if any(x.name.startswith(run_id) for run_id in run_ids):
                rundirs.append(x)

    return rundirs
