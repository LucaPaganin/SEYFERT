import datetime
import pickle
import shutil
import time
from pathlib import Path
import logging
from typing import Dict

from seyfert import VERSION
from seyfert.config.forecast_config import ForecastConfig
from seyfert.cosmology import redshift_density, c_ells, cosmology
from seyfert.fisher.final_results_core import create_final_results
from seyfert.fisher.fisher_matrix import FisherMatrix
from seyfert.fisher.fisher_utils import load_selected_data_vectors
from seyfert.main import cl_core, cl_derivative_core, fisher_core
from seyfert.main.bsub_utils import BsubInterface
from seyfert.utils import formatters, filesystem_utils as fsu
from seyfert.utils.workspace import WorkSpace

logger = logging.getLogger(__name__)


def prepare_workspace(args_dict):
    now = datetime.datetime.now()
    input_data_dir = args_dict['input_data_dir']

    src_input_files = {
        'forecast': args_dict['forecast_config'],
        'PowerSpectrum': args_dict['powerspectrum_config'],
        'Angular': args_dict["angular_config"],
        'Derivative': args_dict["derivative_config"],
        'Fisher': args_dict["fisher_config"],
    }

    logger.info("Configuration files:")
    for key, value in src_input_files.items():
        logger.info(f"{key}: {value}")

    fcfg = ForecastConfig(input_file=src_input_files['forecast'], input_data_dir=input_data_dir)
    fcfg.loadPhysicalParametersFromJSONConfig()

    run_dir_name = f"run_{fcfg.getConfigID()}_{formatters.datetime_str_format(now)}"
    run_dir = args_dict['workdir']

    ws = WorkSpace(run_dir)
    ws.run_dir.mkdir(exist_ok=True, parents=True)
    ws.createInputFilesDir(src_input_files=src_input_files, phys_pars=fcfg.phys_pars, input_data_dir=input_data_dir)

    pmm_dir = args_dict['powerspectrum_dir']
    if pmm_dir is None:
        pmm_dir = Path.home() / "spectrophoto/powerspectra/istf_pmms"
        logger.info(f"Default power spectra: {pmm_dir}")

    ext_dirs = {
        "PowerSpectrum": pmm_dir
    }
    if args_dict['angular_dir'] is not None:
        ext_dirs["Angular"] = Path(args_dict['angular_dir'])
    if args_dict['derivative_dir'] is not None:
        ext_dirs["Derivative"] = Path(args_dict['derivative_dir'])

    ws.symlinkToExternalDirs(ext_dirs, link_delta_cls=False)

    return fcfg, ws


def compute_densities(ws, fcfg, fid_cosmo):
    if not ws.niz_file.is_file():
        densities = {}
        for probe, pcfg in fcfg.probe_configs.items():
            densities[probe] = redshift_density.RedshiftDensity.fromHDF5(pcfg.density_init_file)
            densities[probe].setUp()
            densities[probe].evaluate(z_grid=fid_cosmo.z_grid)
            densities[probe].evaluateSurfaceDensity()

        redshift_density.save_densities_to_file(densities=densities, file=ws.niz_file)
    else:
        densities = redshift_density.load_densities_from_file(file=ws.niz_file)

    return densities


def compute_fiducial_cls(ws, fcfg, main_configs, phys_pars, fid_cosmo, densities):
    ws.cl_dir.mkdir(exist_ok=True)
    fid_cl_file = ws.cl_dir / "dvar_central_step_0" / "cls_fiducial.h5"

    if not fid_cl_file.is_file():
        fid_cls = cl_core.compute_cls(cosmology=fid_cosmo, phys_pars=phys_pars, densities=densities,
                                      forecast_config=fcfg, angular_config=main_configs['Angular'])
        fid_cl_file.parent.mkdir(exist_ok=True, parents=True)
        fid_cls.saveToHDF5(fid_cl_file)
    else:
        fid_cls = c_ells.AngularCoefficientsCollector.fromHDF5(fid_cl_file)

    return fid_cls


def compute_cl_derivatives(ws, fcfg, main_configs, phys_pars, fid_cosmo, densities, fid_cls):
    cl_ders_file = ws.der_dir / "cl_ders.pickle"
    if not ws.der_dir.is_dir():
        ws.der_dir.mkdir()
        cl_ders_file = ws.der_dir / "cl_ders.pickle"

        data = {
            "fid_cls": fid_cls,
            "ws": ws,
            "phys_pars": phys_pars,
            "densities": densities,
            "forecast_config": fcfg,
            "angular_config": main_configs['Angular'],
            "fiducial_cosmology": fid_cosmo
        }

        ti = time.time()
        ders_dict = {}
        n_params = len(phys_pars.free_physical_parameters)

        for i, dvar in enumerate(phys_pars.free_physical_parameters):
            t1 = time.time()
            logger.info(f"{'#' * 40} Computing cl derivatives w.r.t. {dvar}: {i + 1}/{n_params} {'#' * 40}")
            ders_dict[dvar] = cl_derivative_core.compute_cls_derivatives_wrt(dvar, **data)
            t2 = time.time()
            logger.info(f"Elapsed time: {formatters.string_time_format(t2 - t1)}")

        tf = time.time()
        logger.info("")
        logger.info(f"Cl derivatives total elapsed time: {formatters.string_time_format(tf - ti)}")

        with open(cl_ders_file, mode="wb") as f:
            pickle.dump(ders_dict, f)
    else:
        with open(cl_ders_file, mode="rb") as f:
            ders_dict = pickle.load(f)

    return ders_dict


def compute_fishers(ws, fcfg, phys_pars, delta_cls, ders_dict):
    ws.base_fishers_dir.mkdir(exist_ok=True)

    auto_fishers = fisher_core.compute_and_save_fishers(["phph", "spsp", "wlwl"], ws.base_fishers_dir, ws, phys_pars,
                                                        delta_cls, ders_dict)

    datavectors = load_selected_data_vectors()

    brief_str_datavectors = [dv.toBriefString() for dv in datavectors]

    fishers = fisher_core.compute_and_save_fishers(brief_str_data_vectors=brief_str_datavectors,
                                                   outdir=ws.base_fishers_dir, ws=ws, phys_pars=phys_pars,
                                                   delta_cls=delta_cls, dcoll_dvar_dict=ders_dict)

    f_gcsp_pk = FisherMatrix.from_ISTfile(fsu.get_ist_gcsp_pk_fisher_file(fcfg.scenario))

    f_gcsp_pk.writeToFile(outfile=ws.base_fishers_dir / "fisher_IST_gcsp_pk.hdf5")

    return auto_fishers + fishers


def do_forecast(args_dict, logfile):
    fcfg, ws = prepare_workspace(args_dict)
    main_configs = ws.getTasksJSONConfigs()
    phys_pars = fcfg.phys_pars

    # Compute fiducial cosmology
    fid_cosmo = cosmology.Cosmology.fromHDF5(ws.getPowerSpectrumFile(dvar='central', step=0))
    fid_cosmo.evaluateOverRedshiftGrid()

    densities = compute_densities(ws, fcfg, fid_cosmo)

    fid_cls = compute_fiducial_cls(ws, fcfg, main_configs, phys_pars, fid_cosmo, densities)

    ders_dict = compute_cl_derivatives(ws, fcfg, main_configs, phys_pars, fid_cosmo, densities, fid_cls)

    if ws.delta_cls_file.is_file():
        delta_cls = cl_core.DeltaClCollection.fromHDF5(ws.delta_cls_file)
    else:
        delta_cls = cl_core.compute_delta_cls(ws)
        delta_cls.saveToHDF5(ws.delta_cls_file)

    compute_fishers(ws=ws, fcfg=fcfg, phys_pars=phys_pars, delta_cls=delta_cls, ders_dict=ders_dict)

    create_final_results(rundir=ws.run_dir, outdir_name="final_results")

    shutil.copy(logfile, ws.run_dir / Path(logfile).name)


def get_interactive_run_command(args_dict: "Dict") -> "str":
    args_dict['execution'] = 'interactive'
    args_str = " ".join(f"--{key} {value}" for key, value in args_dict.items() if value is not None)
    cmd_str = f"run_seyfert {args_str}"

    return cmd_str


def execute_interactively(args_dict, forecast_configs, logfile):
    n_fcfgs = len(forecast_configs)
    for i, fcfg_file in enumerate(forecast_configs):
        now = formatters.datetime_str_format(datetime.datetime.now())
        workdir = Path(args_dict['workdir']).resolve()
        if args_dict['is_batch_job']:
            logger.info(f"This is a batch job, directly using {workdir} as workdir")
        else:
            logger.info(f"Creating new rundir {workdir}")
            workdir = workdir / f"run_{fcfg_file.stem}_{VERSION}_{now}"

        workdir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Running forecast {i + 1}/{n_fcfgs}: {fcfg_file.name}")
        arguments = {}
        arguments.update(args_dict)
        arguments['forecast_config'] = fcfg_file
        arguments['workdir'] = workdir
        do_forecast(arguments, logfile)


def submit_batch_jobs(args_dict, forecast_configs):
    bsi = BsubInterface()
    for i, fcfg_file in enumerate(forecast_configs):
        now = formatters.datetime_str_format(datetime.datetime.now())
        workdir = Path(args_dict['workdir']).resolve() / f"run_{fcfg_file.stem}_{VERSION}_{now}"
        workdir.mkdir(exist_ok=True, parents=True)
        logs_path = workdir / "batch_logs"
        logs_path.mkdir(exist_ok=True, parents=True)

        logger.info(f"Submitting run with forecast config {fcfg_file.name}")
        fcfg = ForecastConfig(input_file=fcfg_file, input_data_dir=fsu.default_seyfert_input_data_dir())
        bsub_opts = compute_bsub_options(args_dict, fcfg)
        bsi.setOptions(**bsub_opts)
        arguments = {}
        arguments.update(args_dict)
        arguments['forecast_config'] = fcfg_file
        arguments['workdir'] = workdir
        arguments['is_batch_job'] = True
        cmd_str = get_interactive_run_command(arguments)
        bsi.submitJob(cmd_to_execute=cmd_str, logs_path=logs_path, separate_err_out_dirs=True,
                      logs_start_str=f"{fcfg.getConfigID()}", test=args_dict['test_batch'])
        time.sleep(1.5)


def compute_bsub_options(args_dict, fcfg: "ForecastConfig"):
    n_sp_bins = fcfg.n_sp_bins
    bsub_opts = {
        "memory_MB": args_dict['memory'],
        "queue": args_dict['queue'],
        "n_cores": args_dict['n_cores_per_job']
    }
    if n_sp_bins > 24 and bsub_opts['memory_MB'] < 1e4:
        logger.warning(f"Provided memory {bsub_opts['memory_MB']} could not be enough for {n_sp_bins} GCsp bins, "
                       f"defaulting to 10GB")
        bsub_opts['memory_MB'] = 10000

    return bsub_opts