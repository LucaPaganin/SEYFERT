import pytest
import numpy as np

from seyfert.utils import array_utils


@pytest.mark.parametrize("spacing", ["lin", "linear", "Linear"])
def test_ell_arrays_linear_spacing(spacing):
    l_min, l_max = (10, 750)
    n_ell = int(l_max - l_min)
    l_c, l_w = array_utils.compute_multipoles_arrays(l_min, l_max, spacing)

    assert l_c.shape[0] == l_w.shape[0]
    assert l_c.shape[0] == n_ell

    uniq_w = np.unique(l_w)
    assert uniq_w.shape[0] == 1
    assert uniq_w[0] == 1

    with pytest.raises(ValueError):
        _, _ = array_utils.compute_multipoles_arrays(l_min=l_max, l_max=l_min, spacing=spacing)

    with pytest.raises(KeyError):
        _, _ = array_utils.compute_multipoles_arrays(l_min=l_min, l_max=l_max, spacing="unknown_spacing")


@pytest.mark.parametrize("spacing", ["log", "logarithmic", "Logarithmic"])
def test_ell_arrays_log_spacing(spacing):
    l_min, l_max = (10, 750)
    n_ell = 100
    l_c, l_w = array_utils.compute_multipoles_arrays(l_min, l_max, spacing, n_ell=n_ell)

    assert l_c.shape[0] == l_w.shape[0]
    assert l_c.shape[0] == n_ell

    with pytest.raises(ValueError):
        _, _ = array_utils.compute_multipoles_arrays(l_min=l_max, l_max=l_min, spacing=spacing)

    with pytest.raises(KeyError):
        _, _ = array_utils.compute_multipoles_arrays(l_min=l_min, l_max=l_max, spacing="unknown_spacing")

    # missing n_ell
    with pytest.raises(ValueError):
        _, _ = array_utils.compute_multipoles_arrays(l_min=l_max, l_max=l_min, spacing=spacing)

    # wrong type n_ell
    with pytest.raises(ValueError):
        _, _ = array_utils.compute_multipoles_arrays(l_min=l_max, l_max=l_min, spacing=spacing, n_ell="100")


def test_array_intersection():
    a = np.array([1, 2, 2, 3, 4, 5, 5])
    b = np.array([1, 2, 5, 7])

    print(array_utils.compute_arrays_intersection1d([a, b]))

