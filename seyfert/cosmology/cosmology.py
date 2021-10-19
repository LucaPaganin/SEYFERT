"""
Module hosting the cosmology class and its HDF5 file-interface.
The Cosmology possesses cosmological parameters values and it can compute all relevant cosmological functions, like the
Hubble parameter and the co-moving distance. For now accepted cosmological parameters are:
        - :math:`w_0`, CPL dark energy equation of state intercept
        - :math:`w_a`, CPL dark energy equation of state slope
        - :math:`h`, the dimensionless Hubble constant
        - :math:`\\sum m_{\\nu}`, the sum of the neutrino mass eigenstates
        - :math:`\\sigma_8`
        - :math:`n_s`, the scalar spectral index
        - :math:`\\Omega_m`, the cold matter density parameter
        - :math:`\\Omega_b`, the baryon density parameter
        - :math:`\\Omega_{\\rm DE}`, the dark energy density parameter
"""

import logging
from os.path import join
import numpy as np
from numba import jit
from scipy import integrate
from scipy import constants
from seyfert.cosmology.power_spectrum import PowerSpectrum, H5PowerSpectrum
from seyfert.config.main_config import PowerSpectrumConfig
from seyfert.numeric import general as nu
from seyfert.file_io.hdf5_io import AbstractH5FileIO
from seyfert.cosmology.parameter import PhysicalParameter
from typing import Union, Dict
from pathlib import Path


logger = logging.getLogger(__name__)


class CosmologyError(Exception):
    pass


class Cosmology:
    """
    Class for managing the cosmology.

    :param params: a dictionary with the cosmological parameters, stored as :class:`.PhysicalParameter` instances.
    :param z_grid: a :class:`np.ndarray` storing the redshift grid on which to compute functions.
    :param dimensionless_hubble_array: a :class:`np.ndarray` storing the values of the dimensionless Hubble parameter
     :math:`E(z)` sampled on the grid.
    :param dimensionless_comoving_distance_array: a :class:`np.ndarray` storing the values of the comoving distance
     :math:`r(z)` sampled on the grid.
    :param power_spectrum: the matter power spectrum, instance of :class:`PowerSpectrum`.
    """

    dimensionless_hubble_array: "np.ndarray"
    dimensionless_comoving_distance_array: "np.ndarray"
    power_spectrum: "PowerSpectrum"
    c_km_s: "float"
    z_grid: "np.ndarray"
    params: "Dict[str, PhysicalParameter]"

    def __init__(self, params: "Dict[str, PhysicalParameter]" = None, flat: "bool" = None, model_name: "str" = None,
                 z_grid: "np.ndarray" = None, cosmology_name: "str" = None):
        self.dimensionless_hubble_array = None
        self.dimensionless_comoving_distance_array = None
        self.c_km_s = constants.c / 1000
        self.z_grid = z_grid
        self.model_name = model_name
        self.is_flat = flat
        self._transfer_func_k = None
        self.growth_factor_z = None
        self.power_spectrum = None
        self.params = params
        self.name = cosmology_name
        if (self.params is not None) and (self.is_flat is not None):
            self.checkParameters()
        elif (self.params is not None) ^ (self.is_flat is not None):
            raise CosmologyError('parameters and flatness must be specified together or not specified at all')

    @property
    def cosmo_pars_current(self) -> "Dict[str, float]":
        """Current values of the cosmological parameters
        """
        return {name: par.current_value for name, par in self.params.items()}

    @property
    def H0(self) -> "float":
        """Hubble constant in :math:`\\mathrm{km \, s^{-1}\, {Mpc}^{-1}}`
        """
        return self.h * 100

    @property
    def h(self) -> "float":
        """Dimensionless Hubble constant
        """
        return self.cosmo_pars_current["h"]

    @property
    def w0(self) -> "float":
        """Dark energy :math:`w_0` CPL parameter
        """
        return self.cosmo_pars_current["w0"]

    @property
    def wa(self) -> "float":
        """Dark energy :math:`w_a` CPL (slope) parameter
        """
        return self.cosmo_pars_current["wa"]

    @property
    def Omm(self) -> "float":
        """
        Cold matter density parameter :math:`\\Omega_m`
        """
        return self.cosmo_pars_current["Omm"]

    @property
    def OmDE(self) -> "float":
        """
        Dark energy density parameter :math:`\\Omega_{\\rm DE}`
        """
        return self.cosmo_pars_current["OmDE"]

    @property
    def OmK(self) -> "float":
        """
        Curvature density parameter :math:`\\Omega_k`
        """
        return round(1 - (self.Omm + self.OmDE), 10)

    @property
    def Omb(self) -> "float":
        """
        Baryon density parameter :math:`\\Omega_b`
        """
        return self.cosmo_pars_current["Omb"]

    @property
    def sigma8(self) -> "float":
        """
        :math:`\\sigma_8` parameter
        """
        return self.cosmo_pars_current["sigma8"]

    @property
    def ns(self) -> "float":
        """
        Scalar spectral index :math:`n_s`
        """
        return self.cosmo_pars_current["ns"]

    @property
    def mnu(self) -> "float":
        """
        Sum of the neutrino mass eigenstates :math:`\\sum m_{\\nu}` in :math:`\\mathrm{eV}`
        """
        return self.cosmo_pars_current["mnu"]

    @property
    def gamma(self) -> "float":
        return self.cosmo_pars_current["gamma"]

    @property
    def transfer_function_k(self) -> "np.ndarray":
        try:
            return self.power_spectrum.transfer_function
        except AttributeError:
            raise Exception(f'power spectrum {self.power_spectrum} does not have transfer function')

    @property
    def k_grid(self) -> "np.ndarray":
        return self.power_spectrum.k_grid

    @property
    def r_z(self) -> "np.ndarray":
        return (self.c_km_s / self.H0) * self.r_tilde_z

    @property
    def r_tilde_z(self) -> "np.ndarray":
        return self.dimensionless_comoving_distance_array

    @property
    def E_z(self) -> "np.ndarray":
        return self.dimensionless_hubble_array

    @property
    def H_z(self) -> "np.ndarray":
        return self.H0 * self.dimensionless_hubble_array

    def checkParameters(self) -> "None":
        """Method checking values of cosmological parameters, returning error if values are invalid.
        """
        if 'w0' in self.params:
            w0 = self.params['w0']
            if w0.fiducial >= 1 or w0.current_value >= 1:
                raise CosmologyError("w0 cannot be >= 1")
        if 'wa' in self.params:
            wa = self.params['wa']
            if np.abs(wa.fiducial) >= 2 or np.abs(wa.current_value) >= 2:
                raise CosmologyError("|wa| cannot be >= 2")
        if self.is_flat:
            if self.OmK != 0:
                raise CosmologyError(f'Omk must be zero for flat Universe, not {self.OmK}')

    def computeGrowthFactor(self, z):
        integrand = self.computeLogGrowthIntegrand
        if isinstance(z, float):
            result, _ = integrate.quad(integrand, 0, z)
        elif isinstance(z, np.ndarray):
            result = np.vectorize(lambda x: integrate.quad(integrand, 0, x)[0])(z)
        else:
            raise TypeError(f'Expected "Union[float, np.ndarray]", got {type(z)}')

        return np.exp(-result)

    def computeLogGrowthIntegrand(self, z):
        return (self.Omega_m_of_z(z))**self.gamma / (1 + z)

    def rescalePowerSpectrumByComputedGrowth(self):
        D_z_pmm = self.power_spectrum.getGrowthFromLinearPowerSpectrum()
        scaling_ratio_z_k = np.expand_dims((self.growth_factor_z / D_z_pmm)**2, axis=1)

        self.power_spectrum.lin_p_mm_z_k *= scaling_ratio_z_k
        self.power_spectrum.nonlin_p_mm_z_k *= scaling_ratio_z_k

    def evaluatePowerSpectrum(self, workdir: "Union[str, Path]",
                              power_spectrum_config: "PowerSpectrumConfig") -> "None":
        """
        Method computing the power spectrum for the given cosmology. It will make a call to the Boltzmann code
        selected for the computation.

        :param workdir: working directory
        :param power_spectrum_config: power spectrum configuration, instance of :class:`PowerSpectrumConfig`.
        """
        self.power_spectrum = PowerSpectrum(power_spectrum_config=power_spectrum_config, cosmo_pars=self.params)
        self.power_spectrum.evaluateLinearAndNonLinearPowerSpectra(Path(workdir))

    @staticmethod
    @jit(nopython=True)
    def _dimensionlessHubbleParameter(z: "Union[float, np.ndarray]",
                                      w0: "float", wa: "float",
                                      Omm: "float", OmDE: "float") -> "Union[float, np.ndarray]":
        """
        Static method which computes the dimensionless Hubble parameter, i.e. :math:`E(z) \\equiv \\frac{H(z)}{H_0}`

        :param z: the redshift value(s) at which to compute the function. Provide a numpy array for multiple redshift
            values
        :param w0: the value of the dark energy equation of state parameter :math:`w_0` to use
        :param wa: the value of the dark energy equation of state parameter :math:`w_a` to use
        :param Omm: the value of the matter density parameter :math:`\\Omega_m` to use
        :param OmDE: the dark energy density parameter at present time
        :return: the value(s) of :math:`E(z)` at the given redshift(s) z
        """
        return np.sqrt(Omm * (1 + z)**3 + OmDE * (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
                       + (1 - Omm - OmDE) * (1 + z)**2)

    def Omega_m_of_z(self, z):
        return self.Omm * (1 + z)**3 / self.computeDimensionlessHubbleParameter(z)**2

    def computeDimensionlessHubbleParameter(self, z: "Union[float, np.ndarray]") -> "Union[float, np.ndarray]":
        """
        Method to compute the dimensionless Hubble parameter as a function of the redshift.
        """
        return self._dimensionlessHubbleParameter(z, w0=self.w0, wa=self.wa, Omm=self.Omm, OmDE=self.OmDE)

    def computeReciprocalDimensionlessHubbleParameter(self,
                                                      z: "Union[float, np.ndarray]") -> "Union[float, np.ndarray]":
        """
        Shortcut method to compute integrals of the reciprocal of :math:`E(z)`
        """
        return 1.0 / self.computeDimensionlessHubbleParameter(z)

    def computeHubbleParameter(self, z: "Union[float, np.ndarray]") -> "Union[float, np.ndarray]":
        """
        Method to compute the Hubble parameter as a function of the redshift.
        """
        return self.H0 * self.computeDimensionlessHubbleParameter(z)

    def computeComovingDistance(self, z: "Union[float, np.ndarray]") -> "Union[float, np.ndarray]":
        """
        Method to compute the comoving distance as a function of the redshift.
        """
        return (self.c_km_s / self.H0) * self.computeDimensionlessComovingDistance(z)

    def computeDimensionlessComovingDistance(self, z: "Union[float, np.ndarray]") -> "Union[float, np.ndarray]":
        """
        Method to compute the dimensionless comoving distance at a given redshift, computed according the formula:

            .. math:: \\tilde{r}(z) = \\int_{0}^{z} \\frac{dz}{E(z)}

        :param z: the value of the redshift at which to compute the comoving distance
        :return: the value of the comoving distance at the given redshift(s) z
        """
        integrand = self.computeReciprocalDimensionlessHubbleParameter
        if isinstance(z, float):
            result, _ = integrate.quad(integrand, 0, z)
        elif isinstance(z, np.ndarray):
            result = np.vectorize(lambda x: integrate.quad(integrand, 0, x)[0])(z)
        else:
            raise TypeError(f'Expected "Union[float, np.ndarray]", got {type(z)}')
        return result

    def evaluateOverRedshiftGrid(self) -> "None":
        """Method which evaluates the Hubble parameter and the comoving distance on the redshift grid.
        """
        if self.z_grid is None:
            raise ValueError('missing redshift grid')
        if self.dimensionless_hubble_array is None:
            logger.info('Evaluating Hubble over redshift grid')
            self.dimensionless_hubble_array = self.computeDimensionlessHubbleParameter(self.z_grid)
        else:
            logger.info('Hubble array already stored in memory')
        if self.dimensionless_comoving_distance_array is None:
            logger.info('Evaluating comoving distance over redshift grid')
            self.dimensionless_comoving_distance_array = self.computeDimensionlessComovingDistance(self.z_grid)
        else:
            logger.info('Comoving distance array already stored in memory')
        if self.growth_factor_z is None:
            # self.growth_factor_z = self.computeGrowthFactor(self.z_grid)
            self.growth_factor_z = self.power_spectrum.getGrowthFromLinearPowerSpectrum()
        else:
            logger.info('Growth factor array already stored in memory')

    @staticmethod
    def sigmaR(R: Union[int, float], P_lin: np.ndarray, k: np.ndarray) -> "float":
        """
        Function to compute the value :math:`\\sigma_R` of the matter fluctuation variance filtered at a given
        radius value R

        :param R: the radius of the filter
        :param P_lin: the linear matter power spectrum as a function of the wavenumber :math:`k`
        :param k: the wavenumber grid :math:`k`

        :return: the value of :math:`\\sigma_R`

        """
        integrand = k ** 2 * P_lin * nu.top_hat_filter(k, 1 / R) ** 2 / (2 * np.pi ** 2)
        return np.sqrt(integrate.simps(integrand, k))

    def computeSigmaR(self, R: Union[int, float, np.ndarray]) -> "Union[float, np.ndarray]":
        """Method to compute the variance of the filtered matter density fluctuations.
        It computes the variance using a top-hat filter in Fourier space with at a given radius R

        :param R: radius or radii grid, units are Mpc

        :return: the value(s) of :math:`\\sigma_R` at given radius value(s)
        """
        p_lin_z_k_today = self.power_spectrum.lin_p_mm_z_k[0, :]
        k_grid = self.power_spectrum.k_grid
        if isinstance(R, (int, float)):
            sigma_r = self.sigmaR(R, P_lin=p_lin_z_k_today, k=k_grid)
        elif isinstance(R, np.ndarray):
            sigma_r = np.zeros(R.shape)
            for R_idx, myR in enumerate(R):
                sigma_r[R_idx] = self.sigmaR(myR, P_lin=p_lin_z_k_today, k=k_grid)
        else:
            raise TypeError(f'Expected"float"or numpy.ndarray for R parameter, got {type(R)}')
        return sigma_r

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", root: "str" = "/", load_power_spectrum: "bool" = True) -> "Cosmology":
        h5cosmo = H5Cosmology()
        obj = h5cosmo.load(file, root)
        if load_power_spectrum:
            obj.power_spectrum = PowerSpectrum.fromHDF5(file, root)

        return obj

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = "/", save_power_spectrum: "bool" = True) -> "None":
        h5cosmo = H5Cosmology()
        h5cosmo.save(self, file, root)
        if self.power_spectrum is not None and save_power_spectrum:
            h5pmm = H5PowerSpectrum()
            h5pmm.save(self.power_spectrum, file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", root: "str" = "/", load_power_spectrum: "bool" = True) -> "None":
        h5cosmo = H5Cosmology()
        h5cosmo.openFile(file=file, mode='r', root=root)
        h5cosmo.writeToObject(self)
        h5cosmo.closeFile()
        self.checkParameters()
        if load_power_spectrum:
            if self.power_spectrum is not None:
                self.power_spectrum.loadFromHDF5(file, root)
            else:
                self.power_spectrum = PowerSpectrum.fromHDF5(file, root)

    def __eq__(self, other: "Cosmology") -> "bool":
        return all([
            self.params == other.params,
            self.model_name == other.model_name,
            self.is_flat == other.is_flat,
            self.power_spectrum == other.power_spectrum
        ])


class H5Cosmology(AbstractH5FileIO):
    def __init__(self, **kwargs):
        super(H5Cosmology, self).__init__(**kwargs)
        self.builder_func = Cosmology

    def readBuildingData(self) -> "None":
        self.build_data['flat'] = self.root['cosmology'].attrs['is_flat']
        self.build_data['model_name'] = self.root['cosmology'].attrs['cosmological_model']
        self.build_data['params'] = {}
        for par_name, par_grp in self.root['cosmology/cosmological_parameters'].items():
            self.build_data['params'][par_name] = PhysicalParameter.from_dict(par_grp.attrs)

    def writeToObject(self, obj: "Cosmology") -> "None":
        super(H5Cosmology, self).writeToObject(obj)
        self.populateObject(obj, base_grp_path=join(self.root_path, 'cosmology'))
        if not self.build_data:
            self.readBuildingData()
        if obj.is_flat is None:
            obj.is_flat = self.build_data['flat']
        if obj.model_name is None:
            obj.model_name = self.build_data['model_name']
        if obj.params is None:
            obj.params = self.build_data['params']

    def writeObjectToFile(self, obj: "Cosmology") -> "None":
        cosmo_grp = self.createGroup(name='cosmology')
        cosmo_grp.attrs['is_flat'] = obj.is_flat
        cosmo_grp.attrs['cosmological_model'] = obj.model_name
        self.createDataset(name='z_grid', data=obj.z_grid, base_grp=cosmo_grp)
        if obj.E_z is not None:
            self.createDataset(name='dimensionless_hubble_array', data=obj.E_z, base_grp=cosmo_grp)
        else:
            logger.warning('Hubble parameter absent from cosmology, not writing to file')
        if obj.r_tilde_z is not None:
            self.createDataset(name='dimensionless_comoving_distance_array', data=obj.r_tilde_z, base_grp=cosmo_grp)
        else:
            logger.warning('co-moving distance absent from cosmology, not writing to file')
        if obj.growth_factor_z is not None:
            self.createDataset(name='growth_factor_z', data=obj.growth_factor_z, base_grp=cosmo_grp)
        else:
            logger.warning('growth factor absent from cosmology, not writing to file')
        if obj.params is not None:
            cosmo_pars_grp = self.createGroup(name='cosmological_parameters', base_grp=cosmo_grp)
            for name, par in obj.params.items():
                grp = self.createGroup(name=name, base_grp=cosmo_pars_grp)
                grp.attrs.update(par.to_dict())
        else:
            logger.warning('cosmological parameters absent from cosmology, not writing them to file')
        self.closeFile()


def fromHDF5(file: "Union[str, Path]", root: "str" = "/", load_power_spectrum: "bool" = True) -> "Cosmology":
    return Cosmology.fromHDF5(file=file, root=root, load_power_spectrum=load_power_spectrum)
