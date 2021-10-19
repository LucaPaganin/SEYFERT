import pytest
import numpy as np
import itertools

import seyfert.utils.array_utils
from seyfert.fisher import fisher_utils as fu
from seyfert.utils import general_utils as gu


PROBES = ["WL", "GCph", "GCsp"]
XC_COMBOS = list(itertools.combinations(PROBES, 2)) + [tuple(PROBES)]
