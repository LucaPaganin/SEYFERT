from seyfert.cosmology.parameter import PhysicalParameter, PhysicalParametersCollection, ParameterError
import pytest
from seyfert import PROBES_LONG_NAMES
from pytest import fixture
import numpy as np
import logging
import copy
import json

logger = logging.getLogger(__name__)

BASE_STEM_DISPS = np.array([0.00625, 0.0125, 0.01875, 0.025, 0.0375, 0.05, 0.1])
FULL_STEM_DISPS = np.concatenate([-np.flip(BASE_STEM_DISPS), BASE_STEM_DISPS])
STEM_DICT = dict(zip(np.r_[-len(BASE_STEM_DISPS):0, 1:len(BASE_STEM_DISPS)+1], FULL_STEM_DISPS))


@fixture(scope='class', params=[(PhysicalParameter.COSMO_PAR_STRING, 1.0, 1.0, 0.01),
                                (PhysicalParameter.NUISANCE_PAR_STRING, 0.0, 1.0, 0.02),
                                (PhysicalParameter.COSMO_PAR_STRING, -1.0, 1.0, 0.00),
                                (PhysicalParameter.COSMO_PAR_STRING, 0.006, 7.0, -0.02),
                                (PhysicalParameter.COSMO_PAR_STRING, -1.0, 7.0, -0.01)])
def phys_par(request):
    kind, fiducial, stem_factor, eps = request.param

    delta = eps * stem_factor
    if fiducial != 0:
        delta *= fiducial
    current = fiducial + delta
    probe = "probe" if kind == PhysicalParameter.NUISANCE_PAR_STRING else None

    par = PhysicalParameter(name='name', fiducial=fiducial, current_value=current, kind=kind, probe=probe,
                            is_free_parameter=True, stem_factor=stem_factor)
    return par


class TestParameter:
    def test_identity(self, phys_par):
        if phys_par.kind == phys_par.COSMO_PAR_STRING:
            assert phys_par.is_cosmological
        elif phys_par.kind == phys_par.NUISANCE_PAR_STRING:
            assert phys_par.is_nuisance

    def test_to_from_dict(self, phys_par):
        data = phys_par.to_dict()
        new_par = PhysicalParameter.from_dict(data)
        assert phys_par == new_par

    def test_reset_fiducial(self, phys_par):
        phys_par.current_value = phys_par.fiducial + 0.1
        phys_par.resetCurrentValueToFiducial()
        assert phys_par.fiducial == phys_par.current_value

    def test_compute_displaced_value(self, phys_par):
        eps_vals = np.arange(-1.0, 1.0 + 0.1, 0.1)
        for eps in eps_vals:
            delta = eps * phys_par.stem_factor
            if phys_par.fiducial != 0:
                delta *= phys_par.fiducial
            expected = phys_par.fiducial + delta
            result = phys_par.computeValueForDisplacement(eps)
            assert expected == result

            phys_par.updateValueForDisplacement(eps)
            assert phys_par.current_value == expected
            phys_par.resetCurrentValueToFiducial()

    def test_to_JSON(self, phys_par):
        js_str = phys_par.to_JSON()
        data = json.loads(js_str)
        new_par = PhysicalParameter.from_dict(data)
        assert phys_par == new_par


class TestPhysicalParameterCollection:
    coll: "PhysicalParametersCollection"

    @fixture(autouse=True)
    def setup(self, phys_par_coll):
        self.coll = copy.deepcopy(phys_par_coll)

    def test_to_from_JSON(self, tmp_path):
        file = tmp_path / 'test.json'
        self.coll.writeJSON(file)
        new = PhysicalParametersCollection.fromJSON(file)
        assert self.coll == new

    @pytest.mark.parametrize("probe", PROBES_LONG_NAMES)
    def test_nuis_params_for_probe(self, probe):
        nuis_pars = self.coll.getNuisanceParametersForProbe(probe)
        assert all([par.probe == probe and par.is_nuisance
                    for par in nuis_pars.values()])

    def test_params_kind(self):
        cosmo_pars = self.coll.cosmological_parameters
        assert all([par.is_cosmological for par in cosmo_pars.values()])
        nuis_pars = self.coll.nuisance_parameters
        assert all([par.is_nuisance for par in nuis_pars.values()])

    def test_update_params(self):
        self.coll.is_universe_flat = False
        assert self.coll.stem_dict == STEM_DICT
        dvars = self.coll.free_physical_parameters.keys()
        for dvar in dvars:
            logger.info(f'Testing update {dvar}')
            for step, eps in self.coll.stem_dict.items():
                par = self.coll.free_physical_parameters[dvar]
                cur_val = par.computeValueForDisplacement(eps)
                self.coll.updatePhysicalParametersForDvarStep(dvar=dvar, step=step)
                assert self.coll.free_physical_parameters[dvar].current_value == cur_val
                logger.info(f'step {step} eps {eps} cur_val {cur_val}')

    @pytest.mark.parametrize("Omm_free, OmDE_free", [(True, True), (True, False), (False, False), (False, True)])
    def test_flatness_consistency(self, Omm_free, OmDE_free):
        self.coll.is_universe_flat = True
        Omm = self.coll['Omm']
        OmDE = self.coll['OmDE']
        Omm.is_free_parameter = Omm_free
        OmDE.is_free_parameter = OmDE_free
        if Omm_free and OmDE_free:
            for step in self.coll.stem_dict:
                with pytest.raises(ParameterError):
                    self.coll.updatePhysicalParametersForDvarStep(dvar="Omm", step=step)
                with pytest.raises(ParameterError):
                    self.coll.updatePhysicalParametersForDvarStep(dvar="OmDE", step=step)
        elif Omm_free and not OmDE_free:
            for step in self.coll.stem_dict:
                self.coll.updatePhysicalParametersForDvarStep(dvar="Omm", step=step)
                assert OmDE.current_value == 1 - Omm.current_value
        elif not Omm_free and OmDE_free:
            for step in self.coll.stem_dict:
                self.coll.updatePhysicalParametersForDvarStep(dvar="OmDE", step=step)
                assert Omm.current_value == 1 - OmDE.current_value

    def test_flatness_fiducials_consistency(self):
        self.coll.is_universe_flat = True
        Omm = self.coll['Omm']
        OmDE = self.coll['OmDE']
        Omm.is_free_parameter = True
        OmDE.is_free_parameter = True
        Omm.fiducial = 2 - OmDE.fiducial
        with pytest.raises(ParameterError):
            for step in self.coll.stem_dict:
                self.coll.updatePhysicalParametersForDvarStep(dvar="Omm", step=step)
            for step in self.coll.stem_dict:
                self.coll.updatePhysicalParametersForDvarStep(dvar="OmDE", step=step)
