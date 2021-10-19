from seyfert.utils import formatters as fm
import pytest


@pytest.mark.parametrize("input, expected", [("1.0", 1.0),
                                             ("1.0e-03", 1e-3),
                                             ("1.0E+03", 1000.0),
                                             ("1/3", 1./3)])
def test_str_to_float(input, expected):
    assert fm.str_to_float(input) == expected
