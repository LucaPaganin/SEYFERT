import pytest
from pytest import approx
from seyfert.numeric import general as nu
import numpy as np

TOL_PARAMS = {'abs': 1e-14, 'rel': 1e-14}


@pytest.mark.parametrize("prop_factor", np.arange(0.1, 2.0 + 0.2, 0.2))
def test_pad(prop_factor):
    x = np.random.random((100, 100))
    y = prop_factor * x
    expected = 200 * np.abs((1-prop_factor)/(1+prop_factor)) * np.ones(x.shape)
    result = nu.pad(x, y)

    assert approx(result, **TOL_PARAMS) == expected
