import pytest
from typing import Dict, Callable, List
import itertools
import pandas as pd
import numpy as np

from seyfert.fisher import fisher_matrix as fm
from seyfert.fisher import fisher_analysis as fa
from seyfert.fisher import fisher_utils as fu


AUTO_FISHER_NAMES = ["A", "B", "C"]
BASE_FISHER_NAMES = AUTO_FISHER_NAMES + ["A+B+XC", "A+C+XC", "B+C+XC", "A+B+C+XC"]


@pytest.fixture()
def base_fishers(build_fisher_mock, phys_par_coll) -> "Dict[str, fm.FisherMatrix]":
    cosmo_pars = set(phys_par_coll.free_cosmo_pars_fiducials)
    nuis_pars = {
        'A': {'n1A', 'n2A'},
        'B': {'n1B', 'n2B', 'n3B', 'n4B'},
        'C': {'n1C', 'n2C', 'n3C'}
    }

    xc_names = set(BASE_FISHER_NAMES) - set(AUTO_FISHER_NAMES)
    for xc_name in xc_names:
        probes = fu.get_probes_from_fisher_name(xc_name)
        nuis_pars[xc_name] = set().union(*[nuis_pars[auto_key] for auto_key in probes])

    base_fish_dict = {
        name: build_fisher_mock(name, cosmo_pars, nuis_pars[name])
        for name in BASE_FISHER_NAMES
    }

    return base_fish_dict


@pytest.fixture(scope='module')
def build_mock_analysis(phys_par_coll, base_fishers) -> "Callable":
    def _build_analysis() -> "fa.FisherAnalysis":
        an = fa.FisherAnalysis(analysis_name='test_analysis',
                               cosmo_pars_fiducial=phys_par_coll.free_cosmo_pars_fiducials,
                               cosmology_name='w0_wa_CDM')
        an.base_fishers = base_fishers

        return an

    return _build_analysis


@pytest.fixture()
def mock_analysis(phys_par_coll, base_fishers) -> "fa.FisherAnalysis":
    an = fa.FisherAnalysis(analysis_name='test_analysis',
                           marginalize_nuisance=base_fishers['A'].marginalize_nuisance,
                           cosmo_pars_fiducial=phys_par_coll.free_cosmo_pars_fiducials,
                           cosmology_name='w0_wa_CDM')
    an.base_fishers = base_fishers

    return an


POSSIBLE_SUMS = ["A+B", "A+C", "B+C", "A+B+C"]
SUM_COMBOS = []
for k in range(2, 5):
    SUM_COMBOS += list(itertools.combinations(POSSIBLE_SUMS, k))


@pytest.fixture(params=SUM_COMBOS, ids=[", ".join(combo) for combo in SUM_COMBOS])
def sum_combs(request) -> "List[List[str]]":
    combos = request.param

    return [comb.split('+') for comb in combos]


def test_addend_list(mock_analysis, sum_combs):
    mock_analysis.sliceBaseFishers()
    orig_base_fishers = {}
    for key, fish in mock_analysis.base_fishers.items():
        orig_base_fishers[key] = fish.copy()
        orig_base_fishers[key].selectRelevantFisherSubMatrix()
    add_lst = [[name] for name in BASE_FISHER_NAMES] + sum_combs
    mock_analysis.evaluateFishersFromAddendsList(add_lst)
    mock_analysis.prepareFisherMatrices()

    for base_fish_name in BASE_FISHER_NAMES:
        assert mock_analysis.fisher_matrices[base_fish_name] is mock_analysis.base_fishers[base_fish_name]
        assert orig_base_fishers == mock_analysis.base_fishers

    for combo in sum_combs:
        sum_name = "+".join(combo)
        f_sum = mock_analysis.base_fishers[combo[0]].copy()
        for addend in combo[1:]:
            f_sum += mock_analysis.base_fishers[addend]

        assert mock_analysis.fisher_matrices[sum_name] == f_sum


def test_base_fishers(mock_analysis):
    assert mock_analysis.fisher_matrices == {}
    mock_analysis.useBaseFishers()
    assert mock_analysis.fisher_matrices == mock_analysis.base_fishers


def test_eval_errs(mock_analysis):
    mock_analysis.useBaseFishers()
    mock_analysis.prepareFisherMatrices()
    errs = pd.DataFrame({name: fmat.getMarginalizedErrors(fom=True, only_cosmo=True)
                         for name, fmat in mock_analysis.fisher_matrices.items()}).T
    errs = errs.sort_index(axis=0).sort_index(axis=1)

    mock_analysis.evaluateMarginalizedErrors()
    mock_analysis.marginalized_errors.sort_index(axis=0, inplace=True)
    mock_analysis.marginalized_errors.sort_index(axis=1, inplace=True)
    assert np.all(errs == mock_analysis.marginalized_errors)


def test_eval_rel_errs(mock_analysis):
    mock_analysis.useBaseFishers()
    mock_analysis.prepareFisherMatrices()
    mock_analysis.evaluateMarginalizedErrors()
    mock_analysis.evaluateRelativeMarginalizedErrors()

    errs = mock_analysis.marginalized_errors.copy()

    denoms = pd.Series({
        name: abs(fiducial) if fiducial != 0 else 1
        for name, fiducial in mock_analysis.cosmo_pars_fiducials.items()
    })

    if 'FoM' in errs.columns:
        denoms.loc['FoM'] = 1

    rel_errs = errs / denoms

    assert np.all(rel_errs == mock_analysis.relative_marginalized_errors)
