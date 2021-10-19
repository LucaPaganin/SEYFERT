import pytest
from pytest import fixture
from seyfert.utils import filesystem_utils as fsu
from seyfert.utils import general_utils as gu
import numpy as np
import os
import copy
from seyfert.cosmology import power_spectrum as ps
from seyfert.file_io.hdf5_io import H5FileModeError
import logging

logger = logging.getLogger(__name__)
gu.configure_logger(logger)

TOL_PARAMS = {'abs': 1e-15, 'rel': 1e-15}


class TestPowerSpectrum:
    @fixture(autouse=True)
    def setup(self, pmm_fixt):
        logger.info(f'Set up class {self.__class__.__name__}')
        pmm = copy.deepcopy(pmm_fixt)
        self.pmm = pmm

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_factory(self, tmp_path, root):
        file = tmp_path / 'test_pmm.h5'
        self.pmm.saveToHDF5(file, root=root)
        new_pmm = ps.PowerSpectrum.fromHDF5(file, root=root)
        assert new_pmm == self.pmm
        if root != '/':
            with pytest.raises(KeyError):
                _ = ps.PowerSpectrum.fromHDF5(file, root='/')
        if root == '/':
            with pytest.raises(H5FileModeError):
                _ = ps.PowerSpectrum.fromHDF5(file, root='/absent')

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_io(self, tmp_path, root):
        file = tmp_path / 'test_pmm.h5'
        self.pmm.saveToHDF5(file, root=root)
        new_pmm = ps.PowerSpectrum()
        new_pmm.loadFromHDF5(file, root=root)
        assert new_pmm == self.pmm
        if root != '/':
            with pytest.raises(KeyError):
                new_pmm.loadFromHDF5(file, root='/')
        else:
            with pytest.raises(H5FileModeError):
                new_pmm.loadFromHDF5(file, root='/absent')

    @pytest.mark.skipif(os.getenv("USER", "") == "lucapaganin", reason="not calling CAMB in local")
    def test_eval_pmm(self, cfg_pmm_fixt):
        workdir = fsu.the_test_data_dir()
        self.pmm.evaluateLinearAndNonLinearPowerSpectra(workdir)
        attrs = ["z_grid",
                 "k_grid",
                 "lin_p_mm_z_k",
                 "nonlin_p_mm_z_k",
                 "transfer_function"]
        gen = self.pmm.boltzmann_solver
        for name in attrs:
            assert np.all(getattr(self.pmm, name) == getattr(gen, name))
