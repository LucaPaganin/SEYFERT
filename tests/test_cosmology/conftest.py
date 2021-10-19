import numpy as np
from pytest import fixture
from seyfert.cosmology.cosmology import Cosmology
from seyfert.cosmology import power_spectrum as ps
from seyfert.utils import general_utils as gu
from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.config import main_config as mcfg
import logging

logger = logging.getLogger()
gu.configure_logger(logger)


@fixture(scope="session")
def z_fixt():
    return np.linspace(0.001, 2.500, 300, dtype=np.float64)


@fixture(scope="session")
def k_fixt():
    return np.logspace(-5, -1, 100, dtype=np.float64)


@fixture(scope="session")
def pmm_fixt(cfg_pmm_fixt: "mcfg.PowerSpectrumConfig",
             phys_par_coll: "PhysicalParametersCollection",
             z_fixt: "np.ndarray", k_fixt: "np.ndarray") -> "ps.PowerSpectrum":
    logger.info('Instantiating power spectrum fixture')
    pmm = ps.PowerSpectrum(power_spectrum_config=cfg_pmm_fixt, cosmo_pars=phys_par_coll.cosmological_parameters)
    pmm.z_grid = z_fixt
    pmm.k_grid = k_fixt
    pmm.lin_p_mm_z_k = np.ones((len(pmm.z_grid), len(pmm.k_grid)))
    pmm.nonlin_p_mm_z_k = pmm.lin_p_mm_z_k
    pmm.transfer_function = np.zeros(pmm.k_grid.shape)
    dummy_growth_z = 1 + np.sqrt(z_fixt)
    return pmm


@fixture(scope="session", params=["Flat", "NonFlat"])
def cosmo_fixt(z_fixt, pmm_fixt, phys_par_coll, request) -> Cosmology:
    logger.info('Instantiating cosmology fixture')
    flat = request.param == "Flat"
    if flat:
        phys_par_coll['OmDE'].is_free_parameter = False
    cosmology = Cosmology(params=phys_par_coll.cosmological_parameters, flat=flat)
    cosmology.z_grid = z_fixt
    cosmology.power_spectrum = pmm_fixt
    cosmology.evaluateOverRedshiftGrid()
    return cosmology
