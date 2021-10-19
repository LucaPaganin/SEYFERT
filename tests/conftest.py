from xml.etree import ElementTree
from pytest import fixture
import copy
import itertools
import numpy as np
from typing import Dict

import seyfert.utils.array_utils
from seyfert.config.forecast_config import ForecastConfig
from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.config import main_config as mcfg
from seyfert.utils import filesystem_utils as fsu
from seyfert.utils import general_utils as gu

probe_combs = list(itertools.combinations_with_replacement(["Lensing",
                                                            "PhotometricGalaxy",
                                                            "SpectroscopicGalaxy",
                                                            "Void"], 2))


@fixture(scope="session")
def fc_fixt() -> "ForecastConfig":
    fc = ForecastConfig(input_data_dir=fsu.default_data_dir(), input_file=fsu.the_test_data_dir() / "test_fcfg.json")
    fc.loadProbeConfigurationsFromJSONConfig()
    fc.loadPhysicalParametersFromJSONConfig()

    return fc


@fixture(scope='session')
def phys_par_coll(fc_fixt) -> "PhysicalParametersCollection":
    return fc_fixt.phys_pars


@fixture(scope='session')
def cosmo_par_coll(phys_par_coll) -> "PhysicalParametersCollection":
    cosmo_pars = copy.deepcopy(phys_par_coll)
    for name in list(cosmo_pars.keys()):
        if not cosmo_pars[name].is_cosmological:
            del cosmo_pars[name]
    return cosmo_pars


@fixture(scope="session")
def cfg_cl_fixt() -> "mcfg.AngularConfig":
    test_clcfg_file = fsu.config_files_dir() / 'angular_config.json'
    cl_cfg = mcfg.AngularConfig(json_input=test_clcfg_file)

    return cl_cfg


@fixture(scope="session")
def cfg_pmm_fixt() -> "mcfg.PowerSpectrumConfig":
    test_pmmcfg_file = fsu.config_files_dir() / 'power_spectrum_config.json'
    pmmcfg = mcfg.PowerSpectrumConfig(json_input=test_pmmcfg_file)

    return pmmcfg


@fixture(scope="session")
def cfg_fish_fixt() -> "mcfg.FisherConfig":
    test_fishcfg_file = fsu.config_files_dir() / 'fisher_config.json'
    cfg = mcfg.FisherConfig(json_input=test_fishcfg_file)

    return cfg


@fixture(scope='module')
def cl_dict() -> "Dict[str, np.ndarray]":

    return {
        'Lensing_Lensing': seyfert.utils.array_utils.symmetrize_clij_arr(np.random.random((100, 10, 10))),
        'PhotometricGalaxy_PhotometricGalaxy': seyfert.utils.array_utils.symmetrize_clij_arr(np.random.random((100, 10, 10))),
        'SpectroscopicGalaxy_SpectroscopicGalaxy': seyfert.utils.array_utils.symmetrize_clij_arr(np.random.random((100, 10, 10))),
        'Void_Void': seyfert.utils.array_utils.symmetrize_clij_arr(np.random.random((100, 10, 10))),
        'Lensing_PhotometricGalaxy': np.random.random((100, 10, 10)),
        'Lensing_SpectroscopicGalaxy': np.random.random((100, 10, 4)),
        'Lensing_Void': np.random.random((100, 10, 10)),
        'PhotometricGalaxy_SpectroscopicGalaxy': np.random.random((100, 10, 4)),
        'PhotometricGalaxy_Void': np.random.random((100, 10, 10)),
        'SpectroscopicGalaxy_Void': np.random.random((100, 4, 10)),
    }
