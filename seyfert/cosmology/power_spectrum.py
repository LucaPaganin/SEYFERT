"""
Module hosting PowerSpectrum class and H5PowerSpectrum, its HDF5 file-interface class.
"""

from typing import TYPE_CHECKING
import logging
import numpy as np
from os.path import join
from pathlib import Path
from seyfert.file_io.hdf5_io import AbstractH5FileIO
from seyfert.cosmology.parameter import PhysicalParameter
from seyfert.config.main_config import PowerSpectrumConfig
from typing import Dict, Union

if TYPE_CHECKING:
    from seyfert.cosmology.boltzmann_solver import ExternalBoltzmannSolver

logger = logging.getLogger(__name__)


class PowerSpectrum:
    """
    Class to manage the matter power spectrum. It can be used to calculate the linear and non linear power spectra,
     using a given boltzmann code (e.g. CAMB), or as a container storing all the relevant information about the matter
     power spectrum. In order to compute the power spectra an instance the class :class:`ExternalBoltzmannSolver` is
     used. Methods for doing file I/O are present, and the file format employed is HDF5 (.h5), since it allows to store
     binary data with high compression options.

    :param k_grid: the wavenumber grid over which the power spectra are evaluated
    :param z_grid: the redshift grid over which the power spectra are evaluated
    :param lin_p_mm_z_k: the linear matter power spectrum
    :param nonlin_p_mm_z_k: the nonlinear matter power spectrum

    """
    boltzmann_solver: "ExternalBoltzmannSolver"
    power_spectrum_config: "PowerSpectrumConfig"
    lin_p_mm_z_k: "np.ndarray"
    k_grid: "np.ndarray"
    z_grid: "np.ndarray"
    cosmological_parameters: "Dict[str, PhysicalParameter]"

    def __init__(self, power_spectrum_config: "PowerSpectrumConfig" = None,
                 cosmo_pars: "Dict[str, PhysicalParameter]" = None):
        """

        :param power_spectrum_config: power spectrum configuration object
        :type power_spectrum_config: :class:`cfg.PowerSpectrumConfig`
        :param cosmo_pars: dictionary of cosmological parameters
        :type cosmo_pars: :class:`dict`
        """
        self.cosmological_parameters = cosmo_pars
        self.power_spectrum_config = power_spectrum_config
        self.boltzmann_code = "CAMB"
        self.boltzmann_solver = None
        self.k_grid = None
        self.z_grid = None
        self.lin_p_mm_z_k = None
        self.nonlin_p_mm_z_k = None
        self.transfer_function = None

        if self.power_spectrum_config is not None:
            self.boltzmann_code = self.power_spectrum_config.boltzmann_code
            self.z_grid = self.power_spectrum_config.z_grid

    def getGrowthFromLinearPowerSpectrum(self) -> "np.ndarray":
        pmm_ratio_z_k = self.lin_p_mm_z_k / self.lin_p_mm_z_k[0]
        D_z = np.sqrt(pmm_ratio_z_k[:, 0])

        return D_z

    def evaluateLinearAndNonLinearPowerSpectra(self, workdir: "Union[str, Path]") -> "None":
        """
        Method for calling the selected Boltzmann solver.
        """
        if self.boltzmann_code == 'CAMB':
            logger.info('Boltzmann code: CAMB')
            from seyfert.cosmology.boltzmann_solver import CAMBBoltzmannSolver
            self.boltzmann_solver = CAMBBoltzmannSolver(workdir, self.power_spectrum_config,
                                                        self.cosmological_parameters)
        elif self.boltzmann_code == 'CLASS':
            logger.info('Boltzmann code: CLASS')
            from seyfert.cosmology.boltzmann_solver import CLASSBoltzmannSolver
            self.boltzmann_solver = CLASSBoltzmannSolver(workdir, self.power_spectrum_config,
                                                         self.cosmological_parameters)
        self.boltzmann_solver.evaluateLinearAndNonLinearPowerSpectra()
        self.getResultsFromMatterPowerGenerator()

    def getResultsFromMatterPowerGenerator(self) -> "None":
        """
        Method for read and store into attributes the Boltzmann solver results.
        """
        self.z_grid = self.boltzmann_solver.z_grid
        self.k_grid = self.boltzmann_solver.k_grid
        self.lin_p_mm_z_k = self.boltzmann_solver.lin_p_mm_z_k
        self.nonlin_p_mm_z_k = self.boltzmann_solver.nonlin_p_mm_z_k
        self.transfer_function = self.boltzmann_solver.transfer_function

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", root: "str" = "/") -> "PowerSpectrum":
        h5pmm = H5PowerSpectrum()
        return h5pmm.load(file, root)

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = "/") -> "None":
        h5pmm = H5PowerSpectrum()
        h5pmm.save(self, file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", root: "str" = "/") -> "None":
        h5pmm = H5PowerSpectrum()
        h5pmm.openFile(file=file, mode='r', root=root)
        h5pmm.writeToObject(self)
        h5pmm.closeFile()

    def __eq__(self, other: "PowerSpectrum") -> "bool":
        return all([
            np.all(self.z_grid == other.z_grid),
            np.all(self.k_grid == other.k_grid),
            np.all(self.lin_p_mm_z_k == other.lin_p_mm_z_k),
            np.all(self.nonlin_p_mm_z_k == other.nonlin_p_mm_z_k),
            np.all(self.transfer_function == other.transfer_function)
        ])


class H5PowerSpectrum(AbstractH5FileIO):
    def __init__(self, **kwargs):
        super(H5PowerSpectrum, self).__init__(**kwargs)
        self.builder_func = PowerSpectrum

    def writeToObject(self, obj: "PowerSpectrum") -> "None":
        super(H5PowerSpectrum, self).writeToObject(obj)
        self.populateObject(obj, base_grp_path=join(self.root_path, 'power_spectrum'))
        obj.boltzmann_code = self.root['power_spectrum'].attrs['boltzmann_code']
        if 'transfer_function' in self.root:
            obj.transfer_function = self.root['transfer_function'][()]

    def writeObjectToFile(self, obj: "PowerSpectrum") -> "None":
        pmm_group = self.createGroup(name='power_spectrum')
        pmm_group.attrs['boltzmann_code'] = obj.boltzmann_code
        names = ['z_grid', 'k_grid', 'lin_p_mm_z_k', 'nonlin_p_mm_z_k']
        self.createDatasets(data_dict=dict(zip(names, [getattr(obj, name) for name in names])),
                            base_grp=pmm_group)
        if obj.transfer_function is not None:
            self.createDataset(name='transfer_function', data=obj.transfer_function)
