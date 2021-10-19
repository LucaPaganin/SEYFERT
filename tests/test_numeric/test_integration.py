import numpy as np
from scipy import integrate
import pytest
from seyfert.utils import general_utils as gu
from seyfert.numeric.integration import CompositeNewtonCotesIntegrator

logger = gu.getMainLogger()

TOL_PARAMS = {'abs': 1e-15, 'rel': 1e-15}


@pytest.fixture(scope="module")
def sine_test_xy():
    def _sine_test_xy(N=500):
        x = np.linspace(0, np.pi, N)
        y = np.sin(x)
        return x, y

    return _sine_test_xy


def test_against_scipy_simps(sine_test_xy):
    x, y = sine_test_xy()
    itg = CompositeNewtonCotesIntegrator(x=x, y=y, order=2, axis=0)
    scipy_result = integrate.simps(y, x, even='first')
    my_result = itg.computeIntegral()
    logger.info(f"Newton-Cotes order 2 vs Scipy Simps {my_result - scipy_result}")
    assert pytest.approx(my_result, **TOL_PARAMS) == scipy_result


@pytest.mark.parametrize("order", list(range(2, 6)))
def test_analytic_sine(sine_test_xy, order):
    x, y = sine_test_xy(N=1000)
    itg = CompositeNewtonCotesIntegrator(x=x, y=y, order=order, axis=0)
    my_result = itg.computeIntegral()
    analytic_result = 2
    logger.info(f"{my_result - analytic_result}")

    assert pytest.approx(my_result, abs=1e-10, rel=1e-11) == analytic_result
