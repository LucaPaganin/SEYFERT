import time
from pathlib import Path
import logging
import numpy as np
import re
from seyfert.utils import formatters as fm
from abc import ABC, abstractmethod
from typing import Dict, Union
from seyfert.config.main_config import PowerSpectrumConfig
from seyfert.cosmology.parameter import PhysicalParameter

logger = logging.getLogger(__name__)

try:
    import camb
except Exception as e:
    logger.error('camb not found')
    logger.error(e)

try:
    from classy import Class
except Exception as e:
    logger.error('class not found')
    logger.error(e)


class ExternalBoltzmannSolver(ABC):
    cosmological_parameters: "Dict[str, PhysicalParameter]"
    config: "PowerSpectrumConfig"
    z_grid: "np.ndarray"
    k_grid: "np.ndarray"
    transfer_function: "np.ndarray"
    growth_factor: "np.ndarray"
    nonlin_p_mm_z_k: "np.ndarray"
    lin_p_mm_z_k: "np.ndarray"

    def __init__(self, workdir: "Union[str, Path]",
                 config: "PowerSpectrumConfig",
                 cosmological_parameters: "Dict[str, PhysicalParameter]"):
        self.workdir = workdir
        self.cosmological_parameters = cosmological_parameters
        self.config = config
        self.z_grid = self.config.z_grid
        self.k_grid = None
        self.transfer_function = None
        self.lin_p_mm_z_k = None
        self.nonlin_p_mm_z_k = None

    @abstractmethod
    def evaluateLinearAndNonLinearPowerSpectra(self):
        pass

    @abstractmethod
    def run(self):
        pass


class CAMBBoltzmannSolver(ExternalBoltzmannSolver):
    def __init__(self, *args):
        super(CAMBBoltzmannSolver, self).__init__(*args)
        self.k_max = float(self.config.camb_settings['k_max'])
        self.k_per_logint = int(self.config.camb_settings['k_per_logint'])
        self.n_z_division = self.computeCAMBRedshiftGroupsNumber(len(self.z_grid))
        self.ref_ini_file = self.config.camb_ini_path
        self.ini_file = self.workdir / self.ref_ini_file.name

    def evaluateLinearAndNonLinearPowerSpectra(self):
        logger.info('Writing update CAMB inifile')
        self.writeUpdatedCAMBIniFile()
        logger.info("Finetuning As from sigma8")
        self.finetuneScalarAmpFromSigma8()
        logger.info("Done")
        t0 = time.time()
        self.run()
        tf = time.time()
        logger.info(f'CAMB run total elapsed time: {fm.string_time_format(tf-t0)}')

    def writeUpdatedCAMBIniFile(self):
        camb_basis_pars = self.computeCAMBCosmologicalBasis()
        ref_ini_dict = self.readCAMBIniFileToDict(self.ref_ini_file)
        ref_ini_dict.update({name: str(value) for name, value in camb_basis_pars.items()})
        self.writeDictToIniFile(ref_ini_dict, self.ini_file)

    def run(self, get_growth: 'bool' = True, get_transfer_func: 'bool' = True):
        logger.info('Computing matter power spectrum')
        camb_z_array_list = np.array_split(self.z_grid, self.n_z_division)
        camb_pars = camb.read_ini(str(self.ini_file))
        P_mm_lin_list = []
        P_mm_nonlin_list = []
        t0 = time.time()
        for camb_z_array in camb_z_array_list:
            camb_pars.set_matter_power(redshifts=camb_z_array, kmax=self.k_max, k_per_logint=self.k_per_logint,
                                       nonlinear=True, accurate_massive_neutrino_transfers=True)
            bkg_camb = camb.get_background(camb_pars)
            camb_results = camb.get_results(camb_pars)
            logger.info(f'Computing linear matter power for z in {camb_z_array[0]:.3f} - {camb_z_array[-1]:.3f}')
            t0_lin = time.time()
            camb_k_array, z_camb_array, P_mm_lin = bkg_camb.get_linear_matter_power_spectrum(
                hubble_units=False, k_hunit=False, have_power_spectra=True, params=camb_pars
            )
            tf_lin = time.time()
            logger.info(f'Elapsed time: {fm.string_time_format(tf_lin - t0_lin)}')
            logger.info(f'Computing NON-linear matter power for z in {camb_z_array[0]:.3f} - {camb_z_array[-1]:.3f}')
            t0_nonlin = time.time()
            camb_k_array, z_camb_array, P_mm_nonlin = bkg_camb.get_nonlinear_matter_power_spectrum(
                hubble_units=False, k_hunit=False, have_power_spectra=True, params=camb_pars
            )
            tf_nonlin = time.time()
            logger.info(f'Elapsed time: {fm.string_time_format(tf_nonlin - t0_nonlin)}')
            if self.k_grid is None:
                self.k_grid = camb_k_array

            P_mm_lin_list.append(P_mm_lin)
            P_mm_nonlin_list.append(P_mm_nonlin)

            if get_transfer_func:
                if self.transfer_function is None:
                    self.transfer_function = self.getTransferFunctionFromResults(camb_results)
        self.lin_p_mm_z_k = np.concatenate(P_mm_lin_list, axis=0)
        self.nonlin_p_mm_z_k = np.concatenate(P_mm_nonlin_list, axis=0)

        tf = time.time()
        logger.info(f'Power spectrum total elapsed time: {fm.string_time_format(tf - t0)}')

    def finetuneScalarAmpFromSigma8(self) -> "None":
        sigma8 = self.cosmological_parameters['sigma8'].current_value
        sigma8_eval = 10
        inifile_dict = self.readCAMBIniFileToDict(self.ini_file)
        As = float(inifile_dict["scalar_amp"])
        while np.abs((sigma8 - sigma8_eval) / sigma8) > 1e-12:
            camb_pars = camb.read_ini(str(self.ini_file))
            camb_pars.set_matter_power(redshifts=[0], kmax=0.5)  # Melita dice che tutti usano k = 0.5 e Pk lineare
            camb_pars.NonLinear = camb.model.NonLinear_none
            results = camb.get_results(camb_pars)
            sigma8_eval = results.get_sigma8_0()
            corr = (sigma8 / sigma8_eval) ** 2
            As *= corr
            inifile_dict["scalar_amp"] = str(As)
            with open(self.ini_file, "w") as new_inifile:
                for key, val in inifile_dict.items():
                    new_inifile.write(f"{key} = {val}\n")

    def computeCAMBCosmologicalBasis(self) -> "Dict":
        cosmo_pars_current = {
            name: par.current_value
            for name, par in self.cosmological_parameters.items()
        }
        h = cosmo_pars_current['h']
        omm = cosmo_pars_current['Omm']
        omb = cosmo_pars_current['Omb']
        mnu = cosmo_pars_current['mnu']
        omde = cosmo_pars_current['OmDE']

        ombh2 = omb * h * h
        omnuh2 = mnu / 93.14
        omch2 = omm * h * h - ombh2 - omnuh2
        omk = 1 - omm - omde
        camb_basis_pars = {
            "hubble": 100 * h,
            "w": cosmo_pars_current["w0"],
            "wa": cosmo_pars_current["wa"],
            "scalar_spectral_index": cosmo_pars_current['ns'],
            "omch2": omch2,
            "ombh2": ombh2,
            "omnuh2": omnuh2,
            "omk": omk
        }
        return camb_basis_pars

    @staticmethod
    def readCAMBIniFileToDict(ini_file: "Union[str, Path]") -> "Dict[str, str]":
        param_line_regex = re.compile('^([^#].*)=(.*)$')
        lines = [line.strip() for line in Path(ini_file).read_text().splitlines()]
        entries = {}
        for line in lines:
            match = param_line_regex.match(line)
            if match:
                name, value = match.groups()
                entries[name.strip()] = value.strip()
            else:
                pass
        return entries

    @staticmethod
    def writeDictToIniFile(ini_dict: "Dict", file: "Union[str, Path]") -> "None":
        with open(file, mode='w') as fhandle:
            for key, value in ini_dict.items():
                fhandle.write(f'{key} = {value}\n')

    @staticmethod
    def getTransferFunctionFromResults(camb_results) -> "np.ndarray":
        trans = camb_results.get_matter_transfer_data()
        transfer = trans.transfer_data[camb.model.Transfer_tot - 1, :, 0]
        return transfer

    @staticmethod
    def computeCAMBRedshiftGroupsNumber(n_redshifts: "int") -> "int":
        max_CAMB_n_redshifts = 150
        return int(np.ceil(n_redshifts / max_CAMB_n_redshifts))


class CLASSBoltzmannSolver(ExternalBoltzmannSolver):
    def __init__(self, *args):
        super(CLASSBoltzmannSolver, self).__init__(*args)
        self.k_max = float(self.config.class_settings['k_max'])
        self.k_min = float(self.config.class_settings['k_min'])
        self.n_k = int(self.config.class_settings['n_k'])

    def computeCLASSCosmologicalBasis(self) -> "Dict":
        cosmo_pars_current = {
            name: par.current_value
            for name, par in self.cosmological_parameters.items()
        }
        h = cosmo_pars_current['h']
        omm = cosmo_pars_current['Omm']
        omb = cosmo_pars_current['Omb']
        mnu = cosmo_pars_current['mnu']
        omde = cosmo_pars_current['OmDE']
        sigma8 = cosmo_pars_current['sigma8']

        omnu = mnu / (93.14 * h * h)
        omc = omm - omb - omnu
        omk = round(1. - cosmo_pars_current['Omm'] - cosmo_pars_current['OmDE'], 6)

        params_CLASS = {
            'output': 'mPk',
            'non linear': 'halofit',
            'Omega_b': omb,
            'Omega_cdm': omc,
            'N_ur': 2.0328,
            'h': h,
            'sigma8': sigma8,
            'n_s': cosmo_pars_current['ns'],
            'm_ncdm': mnu,
            'P_k_max_1/Mpc': self.k_max,
            'z_max_pk': self.z_grid[-1],
            'use_ppf': 'yes',
            'w0_fld': cosmo_pars_current["w0"],
            'Omega_k': omk,
            'Omega_fld': omde,
            'wa_fld': cosmo_pars_current["wa"],
            'cs2_fld': 1.,
            'N_ncdm': 1,
            'tau_reio': 0.058,
        }

        return params_CLASS

    def run(self):
        logger.info('Computing matter power spectrum')
        class_pars = self.computeCLASSCosmologicalBasis()
        cosmo = Class()
        cosmo.set(class_pars)
        cosmo.compute()

        t0 = time.time()
        self.k_grid = np.logspace(np.log10(self.k_min), np.log10(self.k_max), num=self.n_k)
        logger.info(f'Computing linear matter power')
        t0_lin = time.time()
        self.lin_p_mm_z_k = np.array([[cosmo.pk_lin(ki, zi) for ki in self.k_grid] for zi in self.z_grid])
        tf_lin = time.time()
        logger.info(f'Elapsed time: {fm.string_time_format(tf_lin - t0_lin)}')
        logger.info(f'Computing NON-linear matter power')
        t0_nonlin = time.time()
        self.nonlin_p_mm_z_k = np.array([[cosmo.pk(ki, zi) for ki in self.k_grid] for zi in self.z_grid])
        tf_nonlin = time.time()
        logger.info(f'Elapsed time: {fm.string_time_format(tf_nonlin - t0_nonlin)}')
        tf = time.time()
        logger.info(f'Power spectrum total elapsed time: {fm.string_time_format(tf - t0)}')

    def evaluateLinearAndNonLinearPowerSpectra(self):
        t0 = time.time()
        self.run()
        tf = time.time()
        logger.info(f'CLASS run total elapsed time: {fm.string_time_format(tf-t0)}')
