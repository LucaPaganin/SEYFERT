import pytest
from pytest import approx, fixture
import numpy as np
from seyfert.cosmology import kernel_functions as kfs
from seyfert.cosmology import weight_functions as wfs
from seyfert.file_io.hdf5_io import H5FileModeError
from seyfert.utils import general_utils as gu
import logging
import copy
from seyfert import PROBES_LONG_NAMES
import itertools

logger = logging.getLogger(__name__)
gu.configure_logger(logger)


TOL_PARAMS = {'abs': 1e-15, 'rel': 1e-15}
PROBE_COMBOS = list(itertools.combinations_with_replacement(PROBES_LONG_NAMES, 2))


@fixture(scope="module", params=PROBE_COMBOS, ids=[f'{p1}_{p2}' for p1, p2 in PROBE_COMBOS])
def mock_kernel(request, z_fixt, cosmo_fixt):
    p1, p2 = request.param
    w1 = wfs.weight_function_for_probe(p1)
    w2 = wfs.weight_function_for_probe(p2)

    ni = np.random.randint(1, 20)
    nj = np.random.randint(1, 20) if p1 != p2 else ni
    nz = len(z_fixt)

    w1.z_grid = z_fixt
    w2.z_grid = z_fixt
    w1.w_bin_z = np.ones((ni, nz))
    w2.w_bin_z = np.ones((nj, nz))

    k = kfs.KernelFunction(weight1=w1, weight2=w2, cosmology=cosmo_fixt)
    k.z_grid = z_fixt
    k.k_ijz = np.ones((ni, nj, nz))
    return k


class TestKernelBase:
    @fixture(autouse=True)
    def setup(self, mock_kernel):
        self.k = mock_kernel
        self.p1 = self.k.probe1
        self.p2 = self.k.probe2
        self.w1 = self.k.weight1
        self.w2 = self.k.weight2
        self.z = self.k.cosmology.z_grid
        self.cosmo = self.k.cosmology

    def test_kernel_properties(self):
        assert self.k.obs_key == gu.get_probes_combination_key(self.k.probe1, self.k.probe2)
        assert self.w1.probe == self.k.probe1
        assert self.w2.probe == self.k.probe2

    def test_equality(self):
        new = copy.deepcopy(self.k)
        assert self.k == new

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_kernel_io(self, tmp_path, root):
        file = tmp_path / 'test_kernels.h5'
        self.k.saveToHDF5(file=file, root=root)
        new = kfs.KernelFunction()
        new.loadFromHDF5(file=file, root=root)
        assert np.all(self.k.z_grid == new.z_grid)
        assert np.all(self.k.k_ijz == new.k_ijz)
        if root == '/':
            with pytest.raises(H5FileModeError):
                new.loadFromHDF5(file=file, root='/absent_grp')

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_kernel_factory(self, tmp_path, root):
        file = tmp_path / 'test_kernels.h5'
        self.k.saveToHDF5(file=file, root=root)
        new_kernel = kfs.KernelFunction.fromHDF5(file=file, root=root)
        assert np.all(self.k.z_grid == new_kernel.z_grid)
        assert np.all(self.k.k_ijz == new_kernel.k_ijz)
        if root == '/':
            with pytest.raises(H5FileModeError):
                _ = kfs.KernelFunction.fromHDF5(file=file, root='/absent_grp')

    def test_kernel_computation(self):
        expected = np.zeros((self.w1.n_bins, self.w2.n_bins, self.z.shape[0]))
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                expected[i, j, :] = self.w1.w_bin_z[i, :] * self.w2.w_bin_z[j, :] * \
                                    self.cosmo.c_km_s / (self.cosmo.H_z * self.cosmo.r_z ** 2)
        self.k.evaluateOverRedshiftGrid(self.z, overwrite=True)
        result = self.k.k_ijz
        assert approx(result, **TOL_PARAMS) == expected
