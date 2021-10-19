import pytest
from pytest import approx
from pytest import fixture
import copy
from seyfert.cosmology.cosmology import Cosmology, CosmologyError
from seyfert.file_io.hdf5_io import H5FileModeError

TOL_PARAMS = {'abs': 1e-15, 'rel': 1e-15}


class TestCosmology:
    @fixture(autouse=True,
             params=[(True, 'Omm'), (True, 'OmDE'), (False, 'Omm'), (False, 'OmDE'), (False, 'OmDE', 'OmDE')],
             ids=['Flat_OmmFree', 'Flat_OmDEFree', 'NonFlat_OmmFree', 'NonFlat_OmDEFree', 'NonFlat_OmmOmDEFree'])
    def setup(self, phys_par_coll, pmm_fixt, z_fixt, request):
        flat = request.param[0]
        free_params = request.param[1:]
        params = copy.deepcopy(phys_par_coll)
        not_free_params = {'Omm', 'OmDE'} - set(free_params)
        for par in free_params:
            params[par].is_free_parameter = True
        for par in not_free_params:
            params[par].is_free_parameter = False
        self.cosmo = Cosmology(params=params.cosmological_parameters, flat=flat, model_name='CPL')
        self.cosmo.z_grid = z_fixt
        self.cosmo.power_spectrum = pmm_fixt
        self.z = self.cosmo.z_grid

    def test_equality(self):
        assert self.cosmo == copy.deepcopy(self.cosmo)
        self.cosmo.evaluateOverRedshiftGrid()
        assert self.cosmo == copy.deepcopy(self.cosmo)

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_factory(self, tmp_path, root):
        file = tmp_path / 'test_cosmo.h5'
        self.cosmo.saveToHDF5(file, root)
        new_cosmo = Cosmology.fromHDF5(file, root)
        assert self.cosmo == new_cosmo
        if root != '/':
            with pytest.raises(KeyError):
                _ = Cosmology.fromHDF5(file, root='/')
        else:
            with pytest.raises(H5FileModeError):
                _ = Cosmology.fromHDF5(file, root='/absent')

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_io(self, tmp_path, root):
        file = tmp_path / 'test_cosmo.h5'
        self.cosmo.saveToHDF5(file, root)
        new_cosmo = Cosmology()
        new_cosmo.loadFromHDF5(file, root)
        assert self.cosmo == new_cosmo
        if root != '/':
            with pytest.raises(KeyError):
                new_cosmo.loadFromHDF5(file, root='/')
        else:
            with pytest.raises(H5FileModeError):
                new_cosmo.loadFromHDF5(file, root='/absent')

    @pytest.mark.parametrize("root", ["/grp", "/grp/subgrp", "/"])
    def test_io_after_evaluation(self, tmp_path, root):
        file = tmp_path / 'test_cosmo.h5'
        self.cosmo.saveToHDF5(file, root)
        new_cosmo = Cosmology()
        new_cosmo.loadFromHDF5(file, root)
        assert self.cosmo == new_cosmo
        if root != '/':
            with pytest.raises(KeyError):
                new_cosmo.loadFromHDF5(file, root='/')
        else:
            with pytest.raises(H5FileModeError):
                new_cosmo.loadFromHDF5(file, root='/absent')

    def test_evaluate_grid(self):
        self.cosmo.evaluateOverRedshiftGrid()
        Ez = self.cosmo.computeDimensionlessHubbleParameter(self.cosmo.z_grid)
        Hz = self.cosmo.computeHubbleParameter(self.cosmo.z_grid)
        rz = self.cosmo.computeComovingDistance(self.cosmo.z_grid)
        rz_tilde = self.cosmo.computeDimensionlessComovingDistance(self.cosmo.z_grid)

        assert approx(self.cosmo.E_z, **TOL_PARAMS) == Ez
        assert approx(self.cosmo.H_z, **TOL_PARAMS) == Hz
        assert approx(self.cosmo.r_z, **TOL_PARAMS) == rz
        assert approx(self.cosmo.r_tilde_z, **TOL_PARAMS) == rz_tilde

    def test_properties(self):
        assert self.cosmo.E_z is self.cosmo.dimensionless_hubble_array
        assert self.cosmo.r_tilde_z is self.cosmo.dimensionless_comoving_distance_array
        for name, par in self.cosmo.params.items():
            assert getattr(self.cosmo, name) == par.current_value
        assert self.cosmo.H0 == self.cosmo.h * 100

    @pytest.mark.parametrize("param_name, physical_value, unphysical_value", [("w0", -1, 2), ("wa", 0, 3)])
    def test_w0wa_values(self, param_name, physical_value, unphysical_value):
        if param_name in self.cosmo.params:
            par = self.cosmo.params[param_name]
            par.current_value = unphysical_value
            with pytest.raises(CosmologyError):
                self.cosmo.checkParameters()
            par.current_value = physical_value

            par.fiducial = unphysical_value
            with pytest.raises(CosmologyError):
                self.cosmo.checkParameters()
            par.fiducial = physical_value

        print('done')

    def test_check_parameters_call(self, mocker):
        m = mocker.patch.object(Cosmology, "checkParameters")
        _ = Cosmology(params=self.cosmo.params, flat=self.cosmo.is_flat)
        m.assert_called_once()

    def test_check_parameters(self):
        def reset_cosmo_params(csm: "Cosmology"):
            csm.cosmo_pars_current["Omm"] = 0.32
            csm.cosmo_pars_current["OmDE"] = 0.68

        cosmo = copy.deepcopy(self.cosmo)
        if self.cosmo.is_flat:
            cosmo.params["Omm"].current_value = 2 - cosmo.OmDE
            with pytest.raises(CosmologyError):
                cosmo.checkParameters()
            reset_cosmo_params(cosmo)

            cosmo.params["OmDE"].current_value = 2 - cosmo.Omm
            with pytest.raises(CosmologyError):
                cosmo.checkParameters()
            reset_cosmo_params(cosmo)
