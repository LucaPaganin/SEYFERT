import datetime
import pickle
import shutil
import time
from pathlib import Path
import logging
from typing import Dict

from seyfert.config.forecast_config import ForecastConfig
from .config.main_config import AngularConfig, DerivativeConfig, FisherConfig, PowerSpectrumConfig
from seyfert.cosmology import redshift_density, c_ells, cosmology
from seyfert.fisher.final_results_core import create_final_results
from seyfert.fisher.fisher_matrix import FisherMatrix
from seyfert.fisher.fisher_utils import load_selected_data_vectors
from seyfert.main import cl_core, cl_derivative_core, fisher_core
from seyfert.utils import formatters, filesystem_utils as fsu
from seyfert.utils.workspace import WorkSpace
from seyfert.main import seyfert_core as sc

logger = logging.getLogger('seyfert')


class Seyfert:
    fcfg: ForecastConfig
    pmm_cfg: PowerSpectrumConfig
    cl_cfg: AngularConfig
    der_cfg: DerivativeConfig
    fish_cfg: FisherConfig
    
    def __init__(self, args_dict={}) -> None:
        self.args_dict = args_dict
        self.ws = None
        self.fcfg = None
        self.pmm_cfg = None
        self.cl_cfg = None
        self.der_cfg = None
        self.fish_cfg = None
        self.phys_pars = None
        self.fid_cosmo = None
        self.redshift_densities = None
    
    @property
    def cosmo_pars(self):
        return self.phys_pars.cosmological_parameters
    
    @property
    def z_grid(self):
        return self.pmm_cfg.z_grid

    def prepareWorkspace(self):
        cfgdir = fsu.config_files_dir()
        datadir = fsu.default_data_dir()
        input_data_dir = self.args_dict.get('input_data_dir', datadir)
        test = self.args_dict.get("test", False)
        src_input_files = {
            'forecast': self.args_dict.get("forecast_config", cfgdir/"basic_forecast_config.json"),
            'PowerSpectrum': self.args_dict.get('powerspectrum_config', cfgdir/"power_spectrum_config.json"),
            'Angular': self.args_dict.get("angular_config", cfgdir/'angular_config.json'),
            'Derivative': self.args_dict.get('derivative_config', cfgdir/'derivative_config.json'),
            'Fisher': self.args_dict.get('fisher_config', cfgdir/'fisher_config.json'),
        }
        logger.info("Configuration files:")
        for key, value in src_input_files.items():
            logger.info(f"{key}: {value}")
        self.fcfg = ForecastConfig(
            input_file=src_input_files['forecast'], input_data_dir=input_data_dir)
        self.fcfg.loadPhysicalParametersFromJSONConfig()
        if test:
            run_dir = Path(".").resolve()/"test_seyfert_run"
        else:
            run_dir = self.args_dict['workdir']
        self.ws = WorkSpace(run_dir)
        self.ws.run_dir.mkdir(exist_ok=True, parents=True)
        self.ws.createInputFilesDir(src_input_files=src_input_files,
                                    phys_pars=self.fcfg.phys_pars,
                                    input_data_dir=input_data_dir)
        pmm_dir = self.args_dict.get('powerspectrum_dir')
        if pmm_dir is None:
            pmm_dir = self.ws.run_dir/"PowerSpectrum"
            pmm_dir.mkdir(exist_ok=True)
        ext_dirs = {}
        cldir = self.args_dict.get('angular_dir')
        derdir = self.args_dict.get('derivative_dir')
        if cldir is not None:
            ext_dirs["Angular"] = Path(cldir)
        if derdir is not None:
            ext_dirs["Derivative"] = Path(derdir)
        self.ws.symlinkToExternalDirs(ext_dirs, link_delta_cls=False)
        main_configs = self.ws.getTasksJSONConfigs()
        self.pmm_cfg = main_configs['PowerSpectrum']
        self.cl_cfg = main_configs['Angular']
        self.der_cfg = main_configs['Derivative']
        self.fish_cfg = main_configs['Fisher']
        self.phys_pars = self.fcfg.phys_pars

    def evaluateFiducialCosmology(self, compute_pmm=True, flat=True):
        self.fid_cosmo = cosmology.Cosmology(params=self.cosmo_pars,
                                             flat=flat, 
                                             z_grid=self.z_grid)
        if compute_pmm:
            self.fid_cosmo.evaluatePowerSpectrum(power_spectrum_config=self.pmm_cfg,
                                                 workdir=self.ws.pmm_dir)

    def evaluateRedshiftDensities(self):
        z_grid = self.pmm_cfg.z_grid
        niz_file = self.ws.niz_file
        if not niz_file.is_file():
            densities = {}
            for probe, pcfg in self.fcfg.probe_configs.items():
                densities[probe] = redshift_density.RedshiftDensity.fromHDF5(
                    pcfg.density_init_file)
                densities[probe].setUp()
                densities[probe].evaluate(z_grid=z_grid)
                densities[probe].evaluateSurfaceDensity()
            redshift_density.save_densities_to_file(
                densities=densities, file=niz_file)
        else:
            densities = redshift_density.load_densities_from_file(
                file=niz_file)
        self.redshift_densities = densities

    def computeCls(self):
        clcoll = cl_core.compute_cls(cosmology=self.fid_cosmo,
                                     phys_pars=self.phys_pars,
                                     densities=self.redshift_densities,
                                     forecast_config=self.fcfg,
                                     angular_config=self.cl_cfg)
        return clcoll
