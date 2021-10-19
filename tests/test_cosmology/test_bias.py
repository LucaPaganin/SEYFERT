import pytest
from pytest import fixture, approx
import numpy as np
import copy
from scipy import interpolate
import logging
from seyfert.file_io.hdf5_io import H5FileModeError
from seyfert.cosmology import bias as bs
from seyfert.utils import filesystem_utils as fsu
from seyfert import BIAS_FILES_RELPATHS

logger = logging.getLogger()

TOL_PARAMS = {"abs": 1e-15, "rel": 1e-15}


@fixture(scope='class', params=BIAS_FILES_RELPATHS)
def bias(request) -> "bs.Bias":
    file = fsu.default_data_dir() / request.param
    the_bias = bs.Bias.fromHDF5(file)
    logger.info('instantiating bias')
    return the_bias


@fixture(scope='class')
def eval_bias(bias, z_fixt, cosmo_fixt) -> bs.Bias:
    if bias.model_name == 'piecewise':
        assert isinstance(bias.z_bin_edges, np.ndarray)
        bias.initBiasModel()
    elif bias.model_name == 'constant':
        assert isinstance(bias.z_bin_edges, np.ndarray)
        bias.initBiasModel()
    elif bias.model_name == 'fiducial_growth_void':
        bias.initBiasModel(fiducial_cosmology=cosmo_fixt)
    bias.evaluateBias(z_fixt)
    return bias


class TestBiasBase:
    @fixture(autouse=True)
    def setup(self, bias, z_fixt):
        self.b = bias
        self.z = z_fixt

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_factory(self, tmp_path, root):
        file = tmp_path / 'test_bias.h5'
        self.b.saveToHDF5(file, root=root)
        new = bs.Bias.fromHDF5(file, root=root)
        assert self.b == new
        if root == '/':
            with pytest.raises(H5FileModeError):
                _ = bs.Bias.fromHDF5(file, root='/absent_grp')

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_io(self, tmp_path, root):
        file = tmp_path / 'test_bias.h5'
        self.b.saveToHDF5(file, root=root)
        new = bs.Bias()
        new.loadFromHDF5(file, root=root)
        assert self.b == new
        if root != '/':
            with pytest.raises(H5FileModeError):
                new.loadFromHDF5(file, root='/absent_grp')

    def test_equality(self, bias):
        new = copy.deepcopy(bias)
        assert new == bias


class TestBiasCompute(TestBiasBase):
    @fixture(autouse=True)
    def setup(self, eval_bias, z_fixt):
        self.b = eval_bias
        self.z = self.b.z_grid

    def test_compute_bias(self):
        b_i_z = self.b.model.computeBias(self.b.z_grid)
        assert np.all(self.b.b_i_z == b_i_z)

        if isinstance(self.b.model, bs.PiecewiseBias):
            b_i = np.array(list(self.b.nuisance_parameters.values()))
            zbe = self.b.z_bin_edges
            cond_list, vals_list = self.b.model.getConditionAndValuesLists(self.z)
            expected_b_z = np.piecewise(self.z, cond_list, vals_list)
            expected_b_i_z = np.repeat(expected_b_z[np.newaxis, :], len(b_i), axis=0)

        elif isinstance(self.b.model, bs.ConstantBias):
            b_i = np.array(list(self.b.nuisance_parameters.values()))
            expected_b_i_z = np.repeat(b_i[:, np.newaxis], len(self.z), axis=1)

        elif isinstance(self.b.model, bs.FiducialGrowthVoidBias):
            b_z = self.b.model.nuisance_parameters['voidbias0'] / self.b.model.cosmology.growth_factor_z
            spline = interpolate.InterpolatedUnivariateSpline(x=self.z, y=b_z, k=3)
            b_i = spline(self.b.model.z_bin_centers)
            expected_b_i_z = np.repeat(b_i[:, np.newaxis], len(self.z), axis=1)

        else:
            expected_b_i_z = self.b.b_i_z

        assert approx(b_i_z, **TOL_PARAMS) == expected_b_i_z
