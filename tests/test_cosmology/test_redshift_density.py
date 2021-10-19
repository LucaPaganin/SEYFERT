import pytest
import numpy as np
import copy
from scipy import integrate
from pytest import fixture, approx
from seyfert.utils import filesystem_utils as fsu
from seyfert.cosmology.redshift_density import RedshiftDensity, DensityError, get_bins_overlap
from seyfert.file_io.hdf5_io import H5FileModeError

from seyfert import DENSITY_FILES_RELPATHS

TOL_PARAMS = {'abs': 1e-15, 'rel': 1e-15}


@fixture(scope='module', params=DENSITY_FILES_RELPATHS)
def z_dens(request) -> RedshiftDensity:
    print('Instantiating unevaluated redshift density')
    file = fsu.default_data_dir() / request.param
    d = RedshiftDensity.fromHDF5(file)
    d.setUp()
    return d


@fixture(scope='module')
def eval_z_dens(z_dens, z_fixt) -> RedshiftDensity:
    print('Instantiating evaluated redshift density')
    z_dens.evaluate(z_fixt)
    return z_dens


class TestDensityBase:
    dens: RedshiftDensity
    z_grid: np.ndarray
    tol_params = TOL_PARAMS
    __test__ = True

    @fixture(autouse=True)
    def setup(self, eval_z_dens, z_fixt):
        self.dens = eval_z_dens
        self.z_grid = z_fixt

    @property
    def has_niz_from_input(self) -> bool:
        return self.dens.has_niz_from_input

    @property
    def n_bins(self):
        return self.dens.n_bins

    @property
    def bins_idxs(self):
        return np.arange(self.n_bins)

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_factory(self, tmp_path, root):
        file = tmp_path / 'test_dens.h5'
        self.dens.saveToHDF5(file, root=root)
        new = RedshiftDensity.fromHDF5(file, root=root)
        assert self.dens == new
        if root == "/":
            with pytest.raises(H5FileModeError):
                _ = RedshiftDensity.fromHDF5(file, root='/absent')
        else:
            with pytest.raises(KeyError):
                _ = RedshiftDensity.fromHDF5(file, root='/')

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_io(self, tmp_path, root):
        file = tmp_path / 'test.h5'
        self.dens.saveToHDF5(file, root=root)
        new = RedshiftDensity()
        new.loadFromHDF5(file, root=root)
        assert new == self.dens
        if root != '/':
            with pytest.raises(KeyError):
                new.loadFromHDF5(file, root='/')

    def test_equality(self):
        new = copy.deepcopy(self.dens)
        assert self.dens == new

    def test_properties(self):
        if not self.dens.has_niz_from_input:
            good = {
                'amplitude': 1 - self.dens.instrument_response['f_out'],
                'z_mean': self.dens.instrument_response['z_b'],
                'sigma': self.dens.instrument_response['sigma_b'],
                'c': self.dens.instrument_response['c_b'],
            }
            bad = {
                'amplitude': self.dens.instrument_response['f_out'],
                'z_mean': self.dens.instrument_response['z_o'],
                'sigma': self.dens.instrument_response['sigma_o'],
                'c': self.dens.instrument_response['c_o'],
            }
            assert self.dens.good_fraction_params == good
            assert self.dens.catastrophic_fraction_params == bad
        assert self.dens.z_min == self.dens.input_z_domain[0]
        assert self.dens.z_max == self.dens.input_z_domain[-1]
        assert self.dens.n_bins == len(self.dens.z_bin_edges) - 1
        assert np.all(self.dens.z_bin_centers == (self.dens.z_bin_edges[:-1] + self.dens.z_bin_edges[1:]) / 2)


class TestDensityCompute(TestDensityBase):
    __test__ = True

    @fixture(autouse=True)
    def setup(self, z_dens, z_fixt):
        self.dens = z_dens
        self.z_grid = z_fixt

    def test_modified_gaussian_resp(self):
        if not self.has_niz_from_input:
            z = np.expand_dims(self.z_grid, 0)
            z_p = np.expand_dims(self.z_grid, 1)
            for params in [self.dens.good_fraction_params, self.dens.catastrophic_fraction_params]:
                amplitude = params['amplitude']
                c = params['c']
                sigma = params['sigma']
                z_mean = params['z_mean']
                expected = (amplitude / (np.sqrt(2 * np.pi) * sigma * (1 + z))) * \
                           np.exp(-((z - c * z_p - z_mean) ** 2 / (2 * (sigma * (1 + z)) ** 2)))
                result = self.dens.modifiedGaussianResponse(z_p, z, **params)
                assert approx(result, **self.tol_params) == expected
        else:
            pass

    def test_instr_resp(self):
        z = np.expand_dims(self.z_grid, 0)
        z_p = np.expand_dims(self.z_grid, 1)
        if not self.has_niz_from_input:
            expected = self.dens.modifiedGaussianResponse(z_p, z, **self.dens.good_fraction_params) + \
                       self.dens.modifiedGaussianResponse(z_p, z, **self.dens.catastrophic_fraction_params)
            result = self.dens.computeInstrumentResponse(z_p, z)
            assert approx(result, **self.tol_params) == expected
        else:
            with pytest.raises(DensityError):
                _ = self.dens.computeInstrumentResponse(z_p, z)

    def test_convolve_instr_resp(self):
        z_bins = self.dens.z_bin_edges
        z = self.z_grid
        if not self.has_niz_from_input:
            for i in self.bins_idxs:
                expected = np.zeros(z.shape)
                for z_idx, myz in enumerate(z):
                    expected[z_idx] = integrate.quad(self.dens.computeInstrumentResponse,
                                                     z_bins[i], z_bins[i + 1], args=myz)[0]
                expected *= self.dens.dN_dz_dOmega_spline(z)
                result = self.dens.convolvedNdzdOmegaWithInstrumentResponse(z, i)
                assert approx(result, **self.tol_params) == expected
        else:
            for i in self.bins_idxs:
                with pytest.raises(DensityError):
                    _ = self.dens.convolvedNdzdOmegaWithInstrumentResponse(z, i)

    def test_compute_norm_factor(self):
        if not self.has_niz_from_input:
            for i in self.bins_idxs:
                result = self.dens.computeBinNormFactor(i)
                expected, _ = integrate.quad(self.dens.convolvedNdzdOmegaWithInstrumentResponse,
                                             self.dens.z_min, self.dens.z_max, args=i)
                assert approx(result, **self.tol_params) == expected
        else:
            for i in self.bins_idxs:
                with pytest.raises(DensityError):
                    _ = self.dens.computeBinNormFactor(i)

    def test_compute_norm_density(self):
        expected_norm_density = np.zeros((self.n_bins, len(self.z_grid)))
        result = np.zeros(expected_norm_density.shape)
        if not self.has_niz_from_input:
            self.dens.evaluateBinNormFactors()
            norm_factors = [self.dens.computeBinNormFactor(i) for i in self.bins_idxs]
            for i in self.bins_idxs:
                expected_norm_density[i] = self.dens.convolvedNdzdOmegaWithInstrumentResponse(self.z_grid, i)
                expected_norm_density[i] /= norm_factors[i]
        else:
            for i in self.bins_idxs:
                expected_norm_density[i] = self.dens.n_iz_splines[i](self.z_grid)
        for i in self.bins_idxs:
            result[i] = self.dens.computeNormalizedDensityAtBinAndRedshift(i, self.z_grid)
        assert approx(result, **self.tol_params) == expected_norm_density

    def test_surface_density(self):
        result = np.array([self.dens.computeSurfaceDensityAtBin(i) for i in self.bins_idxs])
        if self.has_niz_from_input:
            expected = self.dens.dN_dOmega_bins
        else:
            expected = [integrate.quad(self.dens.dN_dz_dOmega_spline, self.dens.z_bin_edges[i], self.dens.z_bin_edges[i + 1])[0]
                        for i in self.bins_idxs]
            expected = np.array(expected)
        assert approx(result, **self.tol_params) == expected


class TestEvalDensity(TestDensityBase):
    __test__ = True

    pass


BIN_COMBS = [
    (0.0, 1.0, 1.0, 2.0, None),
    (0.0, 1.0, 2.0, 3.0, None),
    (1.0, 1.5, 1.0, 2.0, (1.0, 1.5)),
    (1.5, 2.0, 1.0, 2.0, (1.5, 2.0)),
    (1.0, 1.5, 1.25, 2.0, (1.25, 1.5)),
    (1.0, 1.5, 1.25, 1.75, (1.25, 1.5)),
    (1.0, 1.5, 0.5, 2.0, (1.0, 1.5))
]


@pytest.mark.parametrize("z1_l, z1_r, z2_l, z2_r, expected", BIN_COMBS)
def test_z_bins_overlap(z1_l, z1_r, z2_l, z2_r, expected):
    overlap = get_bins_overlap(z1_l, z1_r, z2_l, z2_r)
    assert overlap == get_bins_overlap(z2_l, z2_r, z1_l, z1_r)
    assert overlap == expected


@pytest.mark.parametrize("z1_l, z1_r, z2_l, z2_r", [(0.1, 0.0, 1.0, 2.0), (0.0, 0.0, 1.0, 2.0), (0.0, 0.0, 1.0, 1.0)])
def test_raise_z_bin_overlap(z1_l, z1_r, z2_l, z2_r):
    with pytest.raises(Exception):
        _ = get_bins_overlap(z1_l, z1_r, z2_l, z2_r)
    with pytest.raises(Exception):
        _ = get_bins_overlap(z2_l, z2_r, z1_l, z1_r)

