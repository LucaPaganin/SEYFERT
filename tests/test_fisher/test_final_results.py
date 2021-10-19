import pytest

import seyfert.fisher.final_results_core


@pytest.fixture(scope='module')
def mock_fishers_dict():
    return {"A": 1, "B": 2, "A+B+XC(A,B)": 4, "A+XC(A,B)": 3, "B+XC(A,B)": 3}


NAMES_EXPECTED_MAP = {
    "A + B": 3,
    "A - B": -1,
    "A+XC(A,B) - B": 1,
    "A+XC(A,B) + A - B": 2,
    "B+XC(A,B) + B - A": 4
}


@pytest.mark.parametrize("name, expected", list(NAMES_EXPECTED_MAP.items()), ids=NAMES_EXPECTED_MAP.keys())
def test_fisher_a_posteriori_computation(name, expected, mock_fishers_dict):
    result = seyfert.fisher.final_results_core._compute_a_posteriori_fisher_comb_from_name(name, mock_fishers_dict)
    assert result == expected

