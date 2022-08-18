from typing import TYPE_CHECKING, Dict
import logging
import numpy as np

from seyfert.utils.workspace import WorkSpace
from seyfert.cosmology.cosmology import Cosmology
from seyfert.cosmology.redshift_density import RedshiftDensity
from seyfert.cosmology.c_ells import AngularCoefficientsCollector
from seyfert.fisher.delta_cl import DeltaClCollection

if TYPE_CHECKING:
    from seyfert.config.forecast_config import ForecastConfig
    from seyfert.config.probe_config import ProbeConfig
    from seyfert.config.main_config import AngularConfig
    from seyfert.cosmology.parameter import PhysicalParametersCollection

logger = logging.getLogger(__name__)


def load_cosmology(ws: "WorkSpace", dvar: "str", step: "int", phys_pars: "PhysicalParametersCollection") -> "Cosmology":
    if dvar != 'central' and phys_pars[dvar].is_cosmological:
        file = ws.getPowerSpectrumFile(dvar=dvar, step=step)
    else:
        file = ws.getPowerSpectrumFile(dvar='central', step=0)

    cosmo = Cosmology.fromHDF5(file)
    if cosmo.params != phys_pars.cosmological_parameters:
        logger.error(f"Inconsistency between cosmology loaded from {file} and physical parameters")
        for par_name, param in cosmo.params.items():
            if par_name == "gamma":
                continue
            if param != phys_pars[par_name]:
                logger.error(f"parameter: {par_name}")
                logger.error(f"from cosmology: {param}")
                logger.error(f"from phys_pars {phys_pars[par_name]}")

    return cosmo


def compute_densities(probe_configs: "Dict[str, ProbeConfig]", z_grid: "np.ndarray") -> "Dict[str, RedshiftDensity]":
    densities = {}
    for probe, cfg in probe_configs.items():
        logger.info(f"Computing redshift density for probe {probe}")
        d = RedshiftDensity.fromHDF5(cfg.density_init_file)
        d.setUp()
        d.evaluate(z_grid)
        d.evaluateSurfaceDensity()
        densities[probe] = d

    return densities


def compute_cls(cosmology: "Cosmology", phys_pars: "PhysicalParametersCollection",
                densities: "Dict[str, RedshiftDensity]",
                forecast_config: "ForecastConfig", angular_config: "AngularConfig",
                fiducial_cosmology: "Cosmology" = None) -> "AngularCoefficientsCollector":

    cosmology.evaluateOverRedshiftGrid()
    if fiducial_cosmology is not None:
        fiducial_cosmology.evaluateOverRedshiftGrid()
    logger.info('Instantiating Cl collector')
    cl_collector = AngularCoefficientsCollector(phys_params=phys_pars, cosmology=cosmology,
                                                fiducial_cosmology=fiducial_cosmology,
                                                forecast_config=forecast_config, angular_config=angular_config)
    logger.info('Building angular coefficients')
    cl_collector.setUp(densities=densities)
    logger.info('Computing angular coefficients')
    cl_collector.evaluateAngularCoefficients()

    return cl_collector


def compute_cls_variations(dvar: "str", fid_cls, ws, phys_pars, densities, forecast_config, angular_config,
                           fiducial_cosmology=None) -> "Dict[int, AngularCoefficientsCollector]":
    cl_coll_variations = {0: fid_cls}
    for step in np.r_[-7:0, 1:8]:
        logger.info(f"dvar {dvar}, step {step}")
        phys_pars.updatePhysicalParametersForDvarStep(dvar=dvar, step=step)
        cosmology = load_cosmology(ws, dvar, step, phys_pars)
        cl_coll_varied = compute_cls(cosmology, phys_pars, densities, forecast_config, angular_config,
                                     fiducial_cosmology=fiducial_cosmology)
        cl_coll_variations[step] = cl_coll_varied
        phys_pars.resetPhysicalParametersToFiducial()

    return cl_coll_variations


def compute_delta_cls(ws: "WorkSpace", forecast_config: "ForecastConfig" = None) -> "DeltaClCollection":
    fisher_cfg = ws.getTaskJSONConfiguration('Fisher')
    fcfg = forecast_config if forecast_config is not None else ws.getForecastConfiguration()
    delta_cls = DeltaClCollection(fcfg=fcfg, fisher_cfg=fisher_cfg)
    delta_cls.loadInpuDataFromWorkspace(ws)
    if fcfg.shot_noise_file is not None:
        delta_cls.loadShotNoiseFromFile(fcfg.shot_noise_file)
    delta_cls.evaluateSingleBlocks()
    delta_cls.buildXCBlocks()

    return delta_cls
