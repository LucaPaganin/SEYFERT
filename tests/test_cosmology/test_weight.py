import pytest
import numpy as np
import copy
import os
from pytest import fixture, approx
from scipy import integrate
from seyfert.cosmology import weight_functions as wfs
from seyfert.cosmology.redshift_density import RedshiftDensity
from seyfert.cosmology.bias import Bias
from seyfert.file_io.hdf5_io import H5FileModeError
from seyfert import PROBES_LONG_NAMES

TOL_PARAMS = {'abs': 1e-10, 'rel': 1e-10}


@fixture(scope='class', params=PROBES_LONG_NAMES)
def weight(request, fc_fixt, cosmo_fixt, cfg_cl_fixt) -> "wfs.WeightFunction":
    probe = request.param
    probe_cfg = fc_fixt.probe_configs[probe]
    nuis_pars = fc_fixt.phys_pars.getNuisanceParametersForProbe(probe)
    w = wfs.weight_function_for_probe(probe=probe, probe_config=probe_cfg,
                                      nuisance_params=nuis_pars,
                                      cosmology=cosmo_fixt,
                                      fiducial_cosmology=copy.deepcopy(cosmo_fixt),
                                      angular_config=cfg_cl_fixt)
    return w


class TestWeightFunction:
    @fixture(autouse=True)
    def setup(self, weight):
        self.w = weight

    def test_setup(self):
        self.w.setUp()
        assert isinstance(self.w.density, RedshiftDensity)
        if isinstance(self.w, wfs.WeightFunctionWithBias):
            assert isinstance(self.w.bias, Bias)

    def test_equality(self):
        assert self.w == copy.deepcopy(self.w)

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_factory(self, tmp_path, z_fixt, root):
        file = tmp_path / 'weight_file.h5'
        self.w.z_grid = z_fixt
        self.w.w_bin_z = np.repeat(z_fixt[:, np.newaxis], self.w.n_bins, axis=1)
        self.w.saveToHDF5(file, root)
        if isinstance(self.w, wfs.WeightFunctionWithBias):
            new_w = wfs.WeightFunctionWithBias.fromHDF5(file, root)
        else:
            new_w = wfs.WeightFunction.fromHDF5(file, root)
        assert self.w == new_w
        if root == '/':
            with pytest.raises(H5FileModeError):
                _ = wfs.WeightFunction.fromHDF5(file, root='/absent')
        else:
            with pytest.raises(KeyError):
                _ = wfs.WeightFunction.fromHDF5(file, root='/')

    def test_weight_function_for_probe(self):
        probe = self.w.probe
        if probe == "Lensing":
            assert isinstance(self.w, wfs.LensingWeightFunction)
            assert self.w.density.probe == "PhotometricGalaxy"
        elif probe == "PhotometricGalaxy":
            assert isinstance(self.w, wfs.PhotometricGalaxyWeightFunction)
            assert self.w.density.probe == "PhotometricGalaxy"
        elif probe == "SpectroscopicGalaxy":
            assert isinstance(self.w, wfs.SpectroscopicGalaxyWeightFunction)
            assert self.w.density.probe == "SpectroscopicGalaxy"
        elif probe == "Void":
            assert isinstance(self.w, wfs.VoidWeightFunction)
            assert self.w.density.probe == "Void"

    def test_eval(self):
        self.w.setUp()
        if isinstance(self.w, wfs.WeightFunctionWithBias):
            pass


@fixture(scope="function")
def lens_weight(cosmo_fixt, fc_fixt, cfg_cl_fixt) -> "wfs.LensingWeightFunction":
    w = wfs.LensingWeightFunction(probe_config=fc_fixt.probe_configs['Lensing'],
                                  cosmology=cosmo_fixt,
                                  fiducial_cosmology=cosmo_fixt,
                                  angular_config=cfg_cl_fixt)
    w.setUp()
    return w


def test_compute_lensing_efficiency(lens_weight, z_fixt):
    lens_weight.z_grid = z_fixt
    i_values = list(range(10))
    z_grid = lens_weight.z_grid
    cosmo = lens_weight.cosmology
    lens_weight.density.evaluate(z_grid)

    right_wl_eff = np.zeros((len(i_values), len(z_grid)))
    if lens_weight.lensing_efficiency_integration_method == "simpson":
        for i_idx, myi in enumerate(i_values):
            for z_idx, myz in enumerate(z_grid):
                integrand_array = (1 - cosmo.r_z[z_idx] / cosmo.r_z) * lens_weight.density.norm_density_iz[myi, :]
                right_wl_eff[i_idx, z_idx] = integrate.simps(y=integrand_array[z_idx:], x=z_grid[z_idx:])
    else:
        def integrand(x, z=None, i=None):
            return (1 - cosmo.computeComovingDistance(z) / cosmo.computeComovingDistance(x)) * \
                   lens_weight.density.computeNormalizedDensityAtBinAndRedshift(i, x)

        for i_idx, myi in enumerate(i_values):
            for z_idx, myz in enumerate(z_grid):
                right_wl_eff[i_idx, z_idx] = integrate.quad(integrand, myz, lens_weight.z_max, args=(myz, myi))[0]

    res_wl_eff = lens_weight.computeLensingEfficiency(z_grid)

    assert approx(res_wl_eff, **TOL_PARAMS) == right_wl_eff
