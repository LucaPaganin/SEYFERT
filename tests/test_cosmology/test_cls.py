import pytest
import numpy as np
from os.path import join
from pytest import approx
from pytest import fixture
from seyfert.cosmology import c_ells
from seyfert.cosmology import kernel_functions as kfs
from seyfert.cosmology import weight_functions as wfs
from seyfert.file_io.hdf5_io import H5FileModeError
from scipy import interpolate, integrate
from seyfert.numeric import general as nu
from seyfert.utils import general_utils as gu
import copy
import logging
import itertools
from seyfert import PROBES_LONG_NAMES

logger = logging.getLogger(__name__)

TOL_PARAMS = {'abs': 1e-15, 'rel': 1e-15}
PROBE_COMBOS = list(itertools.combinations_with_replacement(PROBES_LONG_NAMES, 2))
PROBE_KEYS = [gu.get_probes_combination_key(p1, p2) for p1, p2 in PROBE_COMBOS]
PROBE_NBINS = dict(zip(PROBES_LONG_NAMES, [10, 10, 10, 4]))


@fixture(scope='module')
def weight_mock(z_fixt, fc_fixt, cosmo_fixt):
    def _weight(probe):
        w = wfs.weight_function_for_probe(probe, probe_config=fc_fixt.probe_configs[probe],
                                          fiducial_cosmology=cosmo_fixt)
        n_bins = PROBE_NBINS[probe]
        w.z_grid = z_fixt
        w.w_bin_z = np.ones((n_bins, len(w.z_grid))) * 1e-4
        return w

    return _weight


@fixture(scope='module')
def kernel_mock(z_fixt, weight_mock, cosmo_fixt):
    def _ker(probe1, probe2):
        w1 = weight_mock(probe1)
        w2 = weight_mock(probe2)
        k = kfs.KernelFunction(weight1=w1, weight2=w2, cosmology=cosmo_fixt)
        k.z_grid = z_fixt
        k.k_ijz = np.ones((k.weight1.n_bins, k.weight2.n_bins, len(k.z_grid))) * 1e-5
        return k

    return _ker


@fixture(scope='module')
def cl_mock(kernel_mock, cfg_cl_fixt, fc_fixt):
    def _cl(probe1, probe2) -> "c_ells.AngularCoefficient":
        kernel = kernel_mock(probe1, probe2)
        cl = c_ells.AngularCoefficient(probe1=probe1, probe2=probe2, kernel=kernel,
                                       angular_config=cfg_cl_fixt, forecast_config=copy.deepcopy(fc_fixt))
        n_ell = 20
        cl.c_lij = np.ones((n_ell, cl.weight1.n_bins, cl.weight2.n_bins))
        cl.l_bin_centers = np.arange(n_ell)
        cl.l_bin_widths = np.arange(n_ell)

        return cl

    return _cl


@fixture(scope='module')
def mock_cl_coll(cl_mock, cosmo_fixt) -> "c_ells.AngularCoefficientsCollector":
    coll = c_ells.AngularCoefficientsCollector(cosmology=cosmo_fixt)
    for p1, p2 in itertools.combinations_with_replacement(PROBES_LONG_NAMES, 2):
        probe_key = gu.get_probes_combination_key(p1, p2)
        cl = cl_mock(p1, p2)
        if p1 not in coll.weight_dict:
            coll.weight_dict[p1] = cl.weight1
        if p2 not in coll.weight_dict:
            coll.weight_dict[p2] = cl.weight2
        coll.cl_dict[probe_key] = cl
    return coll


class TestClCompute:
    @fixture(autouse=True, params=PROBE_COMBOS, ids=PROBE_KEYS)
    def setup(self, request, cl_mock):
        p1, p2 = request.param
        logger.info(f'setup cl {p1} {p2}')
        self.cl = cl_mock(p1, p2)
        yield
        self.cl = cl_mock(p1, p2)

    @property
    def k(self):
        return self.cl.kernel

    @property
    def w1(self):
        return self.cl.kernel.weight1

    @property
    def w2(self):
        return self.cl.kernel.weight2

    def test_cl_properties(self):
        assert self.cl.power_spectrum is self.cl.cosmology.power_spectrum
        assert self.cl.z_array is self.cl.cosmology.z_grid
        assert self.cl.weight1 is self.cl.kernel.weight1
        assert self.cl.weight2 is self.cl.kernel.weight2
        assert self.cl.n_i is self.cl.weight1.n_bins
        assert self.cl.n_j is self.cl.weight2.n_bins
        assert self.cl.n_ell == len(self.cl.l_bin_centers)

    def test_equality(self):
        new = copy.deepcopy(self.cl)
        assert self.cl == new

    def test_evaluate_limber_pmm(self):
        k_lz = (np.expand_dims(self.cl.l_bin_centers, 1) + 0.5) / self.cl.cosmology.r_z
        pmm_log_spline = interpolate.RectBivariateSpline(self.cl.z_array,
                                                         np.log10(self.cl.power_spectrum.k_grid),
                                                         np.log10(self.cl.power_spectrum.nonlin_p_mm_z_k))
        P_l_z = np.zeros(k_lz.shape)
        for z_idx, myz in enumerate(self.cl.z_array):
            P_l_z[:, z_idx] = pmm_log_spline(myz, np.log10(k_lz[:, z_idx]))
        P_l_z = 10 ** P_l_z
        if 'Void' in self.cl.obs_key:
            k_cut = self.cl.forecast_config.probe_configs["Void"].specific_settings["void_kcut_invMpc"]
            k_cut_w = self.cl.forecast_config.probe_configs["Void"].specific_settings["void_kcut_width_invMpc"]
            k_min = k_cut - k_cut_w
            k_max = k_cut + k_cut_w
            smooth_arr = np.zeros(P_l_z.shape)
            for z_idx, myz in enumerate(self.cl.z_array):
                smooth_arr[:, z_idx] = 1 - nu.smoothstep(x=k_lz[:, z_idx], x_min=k_min, x_max=k_max, N=3)
            P_l_z *= smooth_arr
        self.cl.evaluateLimberApproximatedPowerSpectrum()

        assert approx(self.cl.limber_power_spectrum_l_z, **TOL_PARAMS) == P_l_z

    def test_evaluate_angular_correlation(self):
        self.cl.limber_power_spectrum_l_z = np.ones((self.cl.n_ell, len(self.cl.z_array))) * 1e2
        integrand_lijz = np.expand_dims(self.cl.limber_power_spectrum_l_z, axis=(1, 2)) * \
                         np.expand_dims(self.cl.kernel.k_ijz, 0)
        clij = self.cl.computeClIntegral(integrand_lijz, axis=-1)
        if self.cl.is_auto_correlation and self.cl.probe1 == 'SpectroscopicGalaxy':
            probe_cfg = self.cl.forecast_config.probe_configs[self.cl.probe1]
            if not probe_cfg.specific_settings['compute_gcsp_cl_offdiag']:
                idxs = np.indices(self.cl.c_lij.shape)
                off_diag_mask = idxs[1] != idxs[2]
                clij[off_diag_mask] = 0

        self.cl.evaluateAngularCorrelation()
        assert np.all(clij == self.cl.c_lij)

    def test_spectro_off_diagonal_cls(self):
        if self.cl.is_auto_correlation and self.cl.probe1 == 'SpectroscopicGalaxy':
            self.cl.limber_power_spectrum_l_z = np.ones((self.cl.n_ell, len(self.cl.z_array))) * 1e2
            spec_settings = self.cl.forecast_config.probe_configs['SpectroscopicGalaxy'].specific_settings
            orig_flag = spec_settings['compute_gcsp_cl_offdiag']
            spec_settings['compute_gcsp_cl_offdiag'] = True
            self.cl.evaluateAngularCorrelation()
            idxs = np.indices(self.cl.c_lij.shape)
            off_diag_mask = idxs[1] != idxs[2]
            assert not np.all(self.cl.c_lij[off_diag_mask] == 0)
            spec_settings['compute_gcsp_cl_offdiag'] = False
            self.cl.evaluateAngularCorrelation()
            assert np.all(self.cl.c_lij[off_diag_mask] == 0)
            spec_settings['compute_gcsp_cl_offdiag'] = orig_flag
        else:
            pass


class TestClIO:
    @fixture(autouse=True, params=PROBE_COMBOS, ids=PROBE_KEYS)
    def setup(self, request, cl_mock):
        p1, p2 = request.param
        self.cl = cl_mock(p1, p2)
        self.h5cl = c_ells.H5Cl(probe1=self.cl.probe1, probe2=self.cl.probe2)

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_cl_io(self, tmp_path, root):
        file = tmp_path / 'test_cl.h5'
        self.cl.saveToHDF5(file, root)
        new_cl = c_ells.AngularCoefficient(probe1=self.cl.probe1, probe2=self.cl.probe2)
        new_cl.loadFromHDF5(file, root=root)
        assert self.cl == new_cl
        wrong_root = join(root, 'absent_grp')
        with pytest.raises(H5FileModeError):
            new_cl.loadFromHDF5(file, root=wrong_root)

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_cl_factory(self, tmp_path, root):
        file = tmp_path / 'test_cl.h5'
        self.cl.saveToHDF5(file, root)
        new_cl = c_ells.AngularCoefficient.fromHDF5(file, probe1=self.cl.probe1, probe2=self.cl.probe2, root=root)
        assert self.cl == new_cl
        wrong_root = join(root, 'dummy')
        with pytest.raises(H5FileModeError):
            _ = c_ells.AngularCoefficient.fromHDF5(file, probe1=self.cl.probe1, probe2=self.cl.probe2, root=wrong_root)

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_h5cl_io(self, tmp_path, root):
        file = tmp_path / 'file.h5'
        self.h5cl.save(self.cl, file, root)
        new_cl = self.h5cl.load(file, root)
        assert self.cl == new_cl
        wrong_root = join(root, 'dummy')
        with pytest.raises(H5FileModeError):
            _ = self.h5cl.load(file, wrong_root)


class TestClCollector:
    @fixture(autouse=True)
    def setup(self, mock_cl_coll):
        self.coll = mock_cl_coll

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_io(self, tmp_path, root):
        file = tmp_path / 'test.h5'
        self.coll.saveToHDF5(file, root)
        new = c_ells.AngularCoefficientsCollector()
        new.loadFromHDF5(file, root)
        assert self.coll == new

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_factory(self, tmp_path, root):
        file = tmp_path / 'test.h5'
        self.coll.saveToHDF5(file, root)
        new = c_ells.AngularCoefficientsCollector.fromHDF5(file, root)
        assert self.coll == new

