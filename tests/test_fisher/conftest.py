import pytest
import numpy as np
import pandas as pd
import itertools
from typing import Tuple, Dict, Callable
from functools import partial

import seyfert.utils.array_utils
from seyfert.utils import general_utils as gu
from seyfert.fisher import fisher_matrix as fm


@pytest.fixture(scope='module')
def mock_block_data() -> "Tuple[Dict[str, int], Dict[str, np.ndarray], Dict[str, np.ndarray]]":
    probes = ["A", "B", "C"]
    ells_single_probes = {"A": np.arange(100), "B": np.arange(50, 150), "C": np.arange(25, 100)}
    n_bins_dict = {"A": 10, "B": 4, "C": 12}
    ells_dict = {}
    blocks_dict = {}
    for p1, p2 in itertools.combinations_with_replacement(probes, 2):
        ells = seyfert.utils.array_utils.compute_arrays_intersection1d([ells_single_probes[p1], ells_single_probes[p2]])
        n_ells = len(ells)
        nA = n_bins_dict[p1]
        nB = n_bins_dict[p2]
        key = gu.get_probes_combination_key(p1, p2)
        data = np.random.random((n_ells, nA, nB))
        if p1 == p2:
            data = (data + np.transpose(data, axes=(0, 2, 1))) / 2
        blocks_dict[key] = data
        ells_dict[key] = ells

    return n_bins_dict, ells_dict, blocks_dict


@pytest.fixture(scope='module')
def single_probes_data():
    probes = ["A", "B", "C", "D"]
    ells_single_probes = {
        "A": np.arange(100),
        "B": np.arange(50, 150),
        "C": np.arange(25, 100),
        "D": np.arange(10, 90)
    }
    n_bins_dict = {"A": 10, "B": 4, "C": 12, "D": 20}

    return probes, ells_single_probes, n_bins_dict


@pytest.fixture(scope='module')
def blocks_factory(single_probes_data):
    probes, ells_single_probes, n_bins_dict = single_probes_data

    def _block_factory():
        ells_dict = {}
        blocks_dict = {}
        for p1, p2 in itertools.combinations_with_replacement(probes, 2):
            ells = seyfert.utils.array_utils.compute_arrays_intersection1d([ells_single_probes[p1], ells_single_probes[p2]])
            n_ells = len(ells)
            nA = n_bins_dict[p1]
            nB = n_bins_dict[p2]
            key = gu.get_probes_combination_key(p1, p2)
            data = np.random.normal(0, 1, (n_ells, nA, nB))
            if p1 == p2:
                # data = (data + np.transpose(data, axes=(0, 2, 1))) / 2
                data = np.einsum('lji,ljk->lik', data, data)
            blocks_dict[key] = data
            ells_dict[key] = ells

        return ells_dict, blocks_dict

    return _block_factory


@pytest.fixture(scope='module')
def basic_fisher_build() -> "Callable":
    def _build_fisher(name, cosmo_pars, nuis_pars, marg_nuisance=True) -> "fm.FisherMatrix":
        index = sorted(list(cosmo_pars | nuis_pars))
        n_pars = len(index)
        data = np.random.random((n_pars, n_pars))
        data = data.T @ data
        matrix_df = pd.DataFrame(data, index=index, columns=index)
        fishmat = fm.FisherMatrix(name=name, matrix_df=matrix_df, cosmological_parameters=cosmo_pars,
                                  nuisance_parameters=nuis_pars, marginalize_nuisance=marg_nuisance)
        return fishmat

    return _build_fisher


@pytest.fixture(scope='module', params=[True, False], ids=['nuis_marg', 'not_nuis_marg'])
def build_fisher_mock(request, basic_fisher_build) -> "Callable":
    marg_nuisance = request.param

    return partial(basic_fisher_build, marg_nuisance=marg_nuisance)
