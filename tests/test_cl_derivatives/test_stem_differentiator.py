import pytest
import numpy as np

from seyfert.derivatives.differentiator import SteMClDifferentiator


@pytest.fixture(params=['w0', 'wa', 'Omm', 'Omb'])
def mock_cl_der_data(request, phys_par_coll):
    def _mock_cl_dvar_data(noise_eps):
        name = request.param
        dvar_vals = phys_par_coll.computePhysParSTEMValues(name)
        shape = (100, 10, 4)
        slopes = 1e-7 * np.random.random(shape)
        intercepts = 1e-6 * np.random.random(shape)
        c_dvar_lij = slopes * np.expand_dims(dvar_vals, (1, 2, 3)) + intercepts
        noise = np.random.normal(0, 1, c_dvar_lij.shape)*noise_eps*c_dvar_lij
        c_dvar_lij += noise

        return dvar_vals, c_dvar_lij, slopes

    return _mock_cl_dvar_data


@pytest.mark.parametrize('noise_eps', np.logspace(-7, -3, 10))
def test_vectorizedSteM(noise_eps, mock_cl_der_data):
    diff = SteMClDifferentiator()
    x, y, slopes = mock_cl_der_data(noise_eps)

    dydx = diff.vectorizedSteM(x, y)

    assert np.abs(dydx - slopes).max() < noise_eps


