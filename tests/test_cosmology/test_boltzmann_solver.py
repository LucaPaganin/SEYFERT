import pytest_mock
from pytest import fixture, approx
from seyfert.cosmology.boltzmann_solver import CAMBBoltzmannSolver

TOL_PARAMS = {'abs': 1e-15, 'rel': 1e-15}


@fixture(scope='class')
def solver(tmp_path_factory, cfg_pmm_fixt, phys_par_coll) -> "CAMBBoltzmannSolver":
    cosmo_pars = phys_par_coll.cosmological_parameters
    workdir = tmp_path_factory.mktemp('camb_test_workdir')
    _solver = CAMBBoltzmannSolver(workdir, cfg_pmm_fixt, cosmo_pars)
    yield _solver


class TestCAMBBoltzmannSolver:
    @fixture(autouse=True)
    def setup(self, solver):
        self.solver = solver

    @property
    def cosmo_pars(self):
        return self.solver.cosmological_parameters

    def test_evaluate_pmm(self, mocker: "pytest_mock.MockerFixture"):
        mock_write_ini = mocker.patch.object(CAMBBoltzmannSolver, "writeUpdatedCAMBIniFile")
        mock_finetune = mocker.patch.object(CAMBBoltzmannSolver, "finetuneScalarAmpFromSigma8")
        mock_run = mocker.patch.object(CAMBBoltzmannSolver, "run")
        self.solver.evaluateLinearAndNonLinearPowerSpectra()
        mock_write_ini.assert_called_once()
        mock_finetune.assert_called_once()
        mock_run.assert_called_once()

    def test_computeCAMBBasis(self):
        cosmo_current = {key: self.cosmo_pars[key].current_value for key in self.cosmo_pars}
        h = cosmo_current['h']
        omnu = cosmo_current['mnu'] / (93.14 * h * h)
        omm = cosmo_current['Omm']
        omb = cosmo_current['Omb']
        omde = cosmo_current['OmDE']

        expected = {
            'hubble': h * 100,
            'w': cosmo_current['w0'],
            'wa': cosmo_current['wa'],
            'omk': 1 - omm - omde,
            'omnuh2': omnu * h * h,
            "scalar_spectral_index": cosmo_current['ns'],
            "omch2": (omm - omb - omnu) * h * h,
            "ombh2": omb * h * h
        }
        camb_basis = self.solver.computeCAMBCosmologicalBasis()
        assert approx(camb_basis, **TOL_PARAMS) == expected

    def test_CAMB_inifile(self):
        inif_dict = self.solver.readCAMBIniFileToDict(self.solver.ref_ini_file)
        camb_basis = self.solver.computeCAMBCosmologicalBasis()
        inif_dict.update({name: str(value) for name, value in camb_basis.items()})
        self.solver.writeUpdatedCAMBIniFile()
        new_inif_dict = self.solver.readCAMBIniFileToDict(self.solver.ini_file)
        assert new_inif_dict == inif_dict






