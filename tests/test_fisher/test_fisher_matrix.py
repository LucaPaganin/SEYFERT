from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import pytest
import copy
from seyfert.fisher import fisher_matrix as fm


@pytest.fixture(params=["[WL]", "[WL+GCph+XC(WL,GCph)]"])
def fisher_mock(build_fisher_mock, request) -> "fm.FisherMatrix":
    cosmo_pars = {'w0', 'wa', 'Omm', 'Omb', 'sigma8', 'ns', 'h'}
    nuis_pars = {'b1', 'b2', 'b3', 'b4'}
    name = request.param

    fmat = build_fisher_mock(name, cosmo_pars, nuis_pars)
    fmat.selectRelevantFisherSubMatrix()

    return fmat


def test_fisher_io_regular(fisher_mock, tmp_path):
    file = tmp_path / 'fisher_test.hdf5'
    fisher_mock.writeToFile(outfile=file)
    new_fisher = fm.FisherMatrix.fromHDF5(file)
    assert new_fisher == fisher_mock

    new_fisher = fm.FisherMatrix()
    new_fisher.loadFromFile(file, file_ext='hdf5')
    assert new_fisher == fisher_mock


@pytest.mark.parametrize('overwrite', [True, False])
def test_fisher_io_overwrite(fisher_mock, tmp_path, overwrite):
    outfile = tmp_path / 'fisher_test.hdf5'
    fisher_mock.writeToFile(outfile=outfile)
    t0 = datetime.datetime.fromtimestamp(Path(outfile).stat().st_mtime)
    if overwrite:
        fisher_mock.writeToFile(outfile=outfile, overwrite=overwrite)
        t1 = datetime.datetime.fromtimestamp(Path(outfile).stat().st_mtime)
        assert t1 > t0
    else:
        with pytest.raises(FileExistsError):
            fisher_mock.writeToFile(outfile=outfile, overwrite=overwrite)


@pytest.mark.parametrize('skip_if_exists', [True, False])
def test_fisher_io_skip_if_exists(fisher_mock, tmp_path, skip_if_exists):
    outfile = tmp_path / 'fisher_test.hdf5'
    fisher_mock.writeToFile(outfile=outfile)
    t0 = datetime.datetime.fromtimestamp(Path(outfile).stat().st_mtime)
    if skip_if_exists:
        fisher_mock.writeToFile(outfile=outfile, skip_if_exists=skip_if_exists)
        t1 = datetime.datetime.fromtimestamp(Path(outfile).stat().st_mtime)
        assert t0 == t1
    else:
        with pytest.raises(FileExistsError):
            fisher_mock.writeToFile(outfile=outfile, skip_if_exists=skip_if_exists)


def test_invert(fisher_mock):
    inv = np.linalg.inv(fisher_mock.matrix)
    fisher_mock.evaluateInverse()
    assert np.all(inv == fisher_mock.inverse)


@pytest.mark.parametrize('binary_op', [lambda x, y: x + y, lambda x, y: x - y], ids=['sum', 'diff'])
def test_sum(build_fisher_mock, binary_op):
    cosmo_pars = {'w0', 'wa'}
    nuis_pars1 = {'b1', 'b2'}
    nuis_pars2 = {'b3', 'b4'}
    f1: "fm.FisherMatrix" = build_fisher_mock('A', cosmo_pars, nuis_pars1)
    f2: "fm.FisherMatrix" = build_fisher_mock('B', cosmo_pars, nuis_pars2)

    f_res: "fm.FisherMatrix" = binary_op(f1, f2)

    assert set(f1.index) == (cosmo_pars | nuis_pars1)
    assert set(f1.columns) == (cosmo_pars | nuis_pars1)
    assert f1.cosmological_parameters == cosmo_pars
    assert f1.nuisance_parameters == nuis_pars1

    assert set(f2.index) == (cosmo_pars | nuis_pars2)
    assert set(f2.columns) == (cosmo_pars | nuis_pars2)
    assert f2.cosmological_parameters == cosmo_pars
    assert f2.nuisance_parameters == nuis_pars2

    f1, f2 = fm.FisherMatrix.zeroPadMatrices(f1, f2)

    res_df = binary_op(f1.matrix, f2.matrix)

    # Trick to recognize addition from subtraction
    if binary_op(0, 1) == 1 and binary_op(1, 0) == 1:
        expected_name = 'A + B'
    elif binary_op(0, 1) == -1 and binary_op(1, 0) == 1:
        expected_name = 'A - B'
    else:
        raise Exception("Invalid binary operation")

    assert f_res.name == expected_name
    assert np.all(f_res.matrix == res_df)
    assert f_res.cosmological_parameters == (f1.cosmological_parameters | f2.cosmological_parameters)
    assert f_res.nuisance_parameters == (f1.nuisance_parameters | f2.nuisance_parameters)
    assert f_res.physical_parameters == (f1.physical_parameters | f2.physical_parameters)


def test_slice_submatrix(fisher_mock):
    f_old = copy.deepcopy(fisher_mock)
    fisher_mock.selectRelevantFisherSubMatrix()
    if fisher_mock.marginalize_nuisance:
        assert f_old == fisher_mock
    else:
        relevant = f_old.physical_parameters - f_old.nuisance_parameters
        assert set(fisher_mock.index) == relevant
        assert set(fisher_mock.columns) == relevant


def test_get_errs(fisher_mock):
    fisher_mock.evaluateInverse()
    for par in fisher_mock.physical_parameters:
        assert fisher_mock.getParameterSigma(par) == np.sqrt(fisher_mock.inverse.loc[par, par])

    marg_errs = pd.Series(np.sqrt(np.diag(fisher_mock.inverse)), index=fisher_mock.index)
    assert np.all(marg_errs == fisher_mock.getMarginalizedErrors(only_cosmo=False, fom=False))

    marg_errs.loc['FoM'] = fisher_mock.getFigureOfMerit()
    assert np.all(marg_errs.sort_index() ==
                  fisher_mock.getMarginalizedErrors(only_cosmo=False, fom=True).sort_index())

    cosmo_marg_errs = marg_errs.loc[fisher_mock.cosmological_parameters]
    assert np.all(cosmo_marg_errs.sort_index() ==
                  fisher_mock.getMarginalizedErrors(only_cosmo=True, fom=False).sort_index())

    cosmo_marg_errs.loc['FoM'] = fisher_mock.getFigureOfMerit()
    assert np.all(cosmo_marg_errs.sort_index() ==
                  fisher_mock.getMarginalizedErrors(only_cosmo=True, fom=True).sort_index())


@pytest.mark.parametrize(argnames=['cosmo_to_fix', 'nuis_to_fix'],
                         argvalues=[[{'w0', 'wa'}, set()], [set(), {'b1', 'b2'}], [{'w0', 'wa'}, {'b1', 'b2'}]],
                         ids=['fix_only_cosmo', 'fix_only_nuis', 'fix_both'])
def test_fix_params(build_fisher_mock, cosmo_to_fix, nuis_to_fix):
    orig_cosmo = {'w0', 'wa', 'Omm', 'Omb'}
    orig_nuis = {'b1', 'b2', 'b3', 'b4'}

    fmat: "fm.FisherMatrix" = build_fisher_mock('test', orig_cosmo, orig_nuis)

    phys_to_fix = cosmo_to_fix | nuis_to_fix
    fmat.fixParameters(phys_to_fix)

    expected_cosmo = orig_cosmo - cosmo_to_fix
    expected_nuis = orig_nuis - nuis_to_fix
    expected_phys = expected_cosmo | expected_nuis

    assert fmat.cosmological_parameters == expected_cosmo
    assert fmat.nuisance_parameters == expected_nuis
    assert fmat.physical_parameters == expected_phys

    assert set(fmat.matrix.columns) == expected_phys
    assert set(fmat.matrix.index) == expected_phys


@pytest.mark.parametrize('ret_copy', [True, False])
def test_marginalize(fisher_mock, ret_copy):
    fisher_mock.evaluateInverse()
    old_mat = fisher_mock.copy()
    old_inv = fisher_mock.inverse

    if ret_copy:
        resulting = fisher_mock.marginalizeNuisance(ret_copy=ret_copy)
    else:
        fisher_mock.marginalizeNuisance(ret_copy=ret_copy)
        resulting = fisher_mock

    if resulting.marginalize_nuisance:
        cosmo_pars = resulting.cosmological_parameters
        sliced_inv = old_inv.loc[cosmo_pars, cosmo_pars].sort_index(axis=0).sort_index(axis=1)
        assert resulting.nuisance_parameters == set()
        assert np.all(sliced_inv == resulting.inverse.sort_index(axis=0).sort_index(axis=1))
    else:
        assert resulting == old_mat
