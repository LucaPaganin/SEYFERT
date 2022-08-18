import os

VERSION = "1.3.8"
SEYFERT_PATH = os.path.dirname(__file__)
PROBES_LONG_NAMES = ['PhotometricGalaxy', 'SpectroscopicGalaxy', 'Lensing', 'Void']
PROBES_SHORT_NAMES = ['ph', 'sp', 'wl', 'vd']
PROBES_ALIAS_NAMES = ['GCph', 'GCsp', 'WL', 'V']


PROBE_SHORT_TO_LONG = dict(zip(PROBES_SHORT_NAMES, PROBES_LONG_NAMES))
PROBE_LONG_TO_SHORT = {value: key for key, value in PROBE_SHORT_TO_LONG.items()}

PROBE_LONG_TO_ALIAS = dict(zip(PROBES_LONG_NAMES, PROBES_ALIAS_NAMES))
PROBE_ALIAS_TO_LONG = {value: key for key, value in PROBE_LONG_TO_ALIAS.items()}

PROBE_SHORT_TO_ALIAS = dict(zip(PROBES_SHORT_NAMES, PROBES_ALIAS_NAMES))
PROBE_ALIAS_TO_SHORT = {value: key for key, value in PROBE_SHORT_TO_ALIAS.items()}

DENSITY_FILES_RELPATHS = [
    "gcph_dndz_redbook.h5",
    "gcph_dndz_flagship_tutusaus.h5",
    "gcsp_dndz_4_bins.h5",
    "voids_dndz_flagship_old.h5"
]
BIAS_FILES_RELPATHS = [
    "gcph_bias_piecewise.h5",
    "gcsp_bias_piecewise_4_bins.h5",
    "gcph_bias_constant.h5",
    "gcsp_bias_constant.h5",
    "voids_bias_fiducial_growth.h5"
]

__all__ = [
    'cosmology',
    'config',
    'utils',
    'numeric',
    'derivatives',
    'fisher'
]

from seyfert.seyfert import Seyfert
from seyfert.cosmology.cosmology import Cosmology
from seyfert.cosmology.redshift_density import RedshiftDensity
