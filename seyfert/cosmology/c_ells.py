"""
Module hosting the classes for computing and storing the angular power spectra.
"""
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Union
import h5py
import numpy as np
from os.path import join
from scipy import integrate
from scipy import interpolate
import logging
import time
import matplotlib.pyplot as plt

from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.config.forecast_config import ForecastConfig
from seyfert.config.main_config import AngularConfig
from seyfert.cosmology.cosmology import Cosmology
from seyfert.cosmology import kernel_functions as kfs
from seyfert.cosmology import weight_functions as wfs
from seyfert.cosmology.redshift_density import RedshiftDensity
from seyfert.cosmology import power_spectrum as ps
from seyfert.file_io.hdf5_io import AbstractH5FileIO
from seyfert.numeric import general as nu
from seyfert.utils import general_utils as gu
from seyfert.utils import formatters as fm
from seyfert.numeric.integration import CompositeNewtonCotesIntegrator
from seyfert.plot_utils.cl_plot import grid_plot
from seyfert.utils.shortcuts import ClKey

logger = logging.getLogger(__name__)


class ClError(Exception):
    pass


class AngularCoefficient:
    """Class representing a particular kind of angular power spectrum (i.e. auto-correlation or cross-correlation.
    These are computed in the Limber approximation, meaning that a single redshift integral is performed.

    :param c_lij: np 3D array (l, i, j) storing the values of the Cl tomographic matrix. Here l is the multipole
    :param l_bin_centers: the multipole bin centers where to compute the Cl, np 1D array.
    :param l_bin_widths: the multipole bin widths, np 1D array.
    :param limber_power_spectrum_l_z: np 2D array (l, z) storing the matter power spectrum evaluated in the Limber
     approximation, i.e. at wave-numbers :math:`k = \\frac{l+1/2}{r(z)}`. The redshift dimension is define by the
     redshift grid
    :param cosmology: instance of :class:`Cosmology` class storing the current cosmology.
    :param angular_config: configuration for the Cl computation, instance of :class:`AngularConfig`.
    :param forecast_config: forecast configuration, instance of :class:`ForecastConfig`.
    :param kernel: kernel function to be integrated against the Limber power spectrum, instance of
     :class:`KernelFunction`.
    """
    angular_config: "AngularConfig"
    forecast_config: "ForecastConfig"
    kernel: "kfs.KernelFunction"
    c_lij: "np.ndarray"
    z_array: "np.ndarray"
    cosmology: "Cosmology"

    def __init__(self, probe1: "str" = None, probe2: "str" = None, kernel: "kfs.KernelFunction" = None,
                 forecast_config: "ForecastConfig" = None, angular_config: "AngularConfig" = None):
        self.angular_config = angular_config
        self.forecast_config = forecast_config
        self.kernel = kernel
        self.probe1 = probe1
        self.probe2 = probe2
        self.l_bin_centers = None
        self.l_bin_widths = None
        self.limber_power_spectrum_l_z = None
        self.cosmology = None
        self.c_lij = None
        self._attrs_excluded_from_equality = set()
        if self.kernel is not None:
            if self.probe1 is not None:
                assert self.probe1 == self.kernel.probe1
            else:
                self.probe1 = self.kernel.probe1
            if self.probe2 is not None:
                assert self.probe2 == self.kernel.probe2
            else:
                self.probe2 = self.kernel.probe2
            self.cosmology = self.kernel.cosmology

        if self.forecast_config is not None:
            self.l_bin_centers, self.l_bin_widths = self.forecast_config.getMultipoleArraysForProbeComb(self.obs_key)

    @property
    def is_auto_correlation(self) -> "bool":
        return self.probe1 == self.probe2

    @property
    def ells(self) -> "np.ndarray":
        return self.l_bin_centers

    @property
    def power_spectrum(self) -> "ps.PowerSpectrum":
        """Reference to the power spectrum possessed by the cosmology attribute.
        """
        return self.cosmology.power_spectrum

    @property
    def obs_key(self) -> "str":
        try:
            return self.kernel.obs_key
        except AttributeError:
            return gu.get_probes_combination_key(self.probe1, self.probe2)

    @property
    def weight1(self) -> "wfs.WeightFunction":
        """Reference to first weight function making up the kernel"""
        return self.kernel.weight1

    @property
    def weight2(self) -> "wfs.WeightFunction":
        """Reference to second weight function making up the kernel"""
        return self.kernel.weight2

    @property
    def z_array(self) -> "np.ndarray":
        """Reference to the `z_grid` attribute of the possessed `cosmology` instance.
        """
        try:
            return self.cosmology.z_grid
        except AttributeError:
            return self.kernel.z_grid

    @property
    def n_z(self) -> "int":
        return self.z_array.size

    @property
    def n_ell(self) -> "int":
        return len(self.l_bin_centers)

    @property
    def n_i(self) -> "int":
        return self.weight1.n_bins

    @property
    def n_j(self) -> "int":
        return self.weight2.n_bins

    @property
    def integ_method(self) -> "str":
        return self.angular_config.cl_integration_method

    @property
    def verbose(self):
        return self.forecast_config.verbose

    def getWeightFunction(self, probe: "str") -> "wfs.WeightFunction":
        if probe == self.probe1:
            return self.weight1
        elif probe == self.probe2:
            return self.weight2
        else:
            raise KeyError(probe)

    def getLimberRedshiftIntegrand(self, i: "int", j: "int") -> "np.ndarray":
        return self.limber_power_spectrum_l_z * self.kernel.k_ijz[i, j, :]

    def evaluateAngularCorrelation(self) -> "None":
        """
        Method for evaluating and storing the Cl.
        """
        self.c_lij = np.zeros((self.n_ell, self.n_i, self.n_j))

        if self.verbose:
            logger.info(f'Computing cl integrand')
            logger.info(f'Doing cl integral, integration method: {self.integ_method}')
        if self.is_auto_correlation:
            probe = self.probe1
            if probe != "SpectroscopicGalaxy":
                compute_off_diagonal = True
            else:
                probe_cfg = self.forecast_config.probe_configs["SpectroscopicGalaxy"]
                compute_off_diagonal = probe_cfg.specific_settings["compute_gcsp_cl_offdiag"]

            for myi, myj in itertools.combinations_with_replacement(range(self.n_i), 2):
                integrand = self.getLimberRedshiftIntegrand(myi, myj)
                if myi != myj and not compute_off_diagonal:
                    if self.verbose:
                        logger.info(f'Skipping off-diagonal element {myi}-{myj}')
                else:
                    integral = self.computeClIntegral(integrand=integrand, axis=-1)
                    self.c_lij[:, myi, myj] = integral
                    self.c_lij[:, myj, myi] = self.c_lij[:, myi, myj]
        else:
            for myi, myj in itertools.product(range(self.n_i), range(self.n_j)):
                integrand = self.getLimberRedshiftIntegrand(myi, myj)
                integral = self.computeClIntegral(integrand=integrand, axis=-1)
                self.c_lij[:, myi, myj] = integral

    def computeClIntegral(self, integrand: "np.ndarray", axis: "int") -> "np.ndarray":
        """
        Method for computing a single Cl_ij.
        """
        method = self.angular_config.cl_integration_method
        if method.startswith("newton-cotes"):
            order = int(method.split('_')[-1])
            integrator = CompositeNewtonCotesIntegrator(x=self.z_array, y=integrand, order=order, axis=axis)
            result = integrator.computeIntegral()
        elif method == 'simpson':
            result = integrate.simps(y=integrand, x=self.z_array, axis=axis)
        elif method == 'romberg':
            dx = self.z_array[1] - self.z_array[0]
            two_power = np.log2(self.n_z - 1)
            if two_power != int(two_power):
                raise ValueError(f"Invalid number of samples {self.n_z} for Romberg integration, "
                                 f"should be of the form 2^k + 1, with k integer")
            result = integrate.romb(y=integrand, dx=dx, axis=axis)
        elif method == 'quad':
            result = np.zeros(integrand.shape[0])
            for ell_idx in range(len(result)):
                spl = interpolate.interp1d(x=self.z_array, y=integrand[ell_idx], kind='linear')
                result[ell_idx] = integrate.quad(spl, a=self.z_array[0], b=self.z_array[-1])[0]
        else:
            raise ClError(f"unrecognized integration method: {self.integ_method}")

        return result

    def loadCosmology(self, pmm_file: "Union[str, Path]", load_power_spectrum: "bool" = True) -> "None":
        self.cosmology = Cosmology.fromHDF5(pmm_file, load_power_spectrum=load_power_spectrum)

    def setUp(self) -> "None":
        if self.probe1 != self.probe2:
            self.l_bin_centers, self.l_bin_widths = self.forecast_config.getMultipoleArraysForProbeComb(self.obs_key)
        else:
            self.l_bin_centers = self.forecast_config.probe_configs[self.probe1].l_bin_centers
            self.l_bin_widths = self.forecast_config.probe_configs[self.probe1].l_bin_widths
        self.evaluateLimberApproximatedPowerSpectrum()
        if not self.kernel.is_evaluated:
            self.kernel.evaluateOverRedshiftGrid(self.z_array)

    def evaluateLimberApproximatedPowerSpectrum(self) -> "None":
        pmm_logspline = interpolate.RectBivariateSpline(self.z_array,
                                                        np.log10(self.power_spectrum.k_grid),
                                                        np.log10(self.power_spectrum.nonlin_p_mm_z_k),
                                                        kx=self.angular_config.limber_spline_kx,
                                                        ky=self.angular_config.limber_spline_ky,
                                                        s=self.angular_config.limber_spline_s)
        self.limber_power_spectrum_l_z = np.zeros((len(self.l_bin_centers), len(self.z_array)))
        k_lz = np.expand_dims((self.l_bin_centers + 0.5), 1) / self.cosmology.r_z
        for (z_idx, myz) in enumerate(self.z_array):
            self.limber_power_spectrum_l_z[:, z_idx] = pmm_logspline(myz, np.log10(k_lz[:, z_idx]))
        self.limber_power_spectrum_l_z = 10 ** self.limber_power_spectrum_l_z
        if 'Void' in self.obs_key:
            self.applyVoidsCutToLimberPowerSpectrum()

    def applyVoidsCutToLimberPowerSpectrum(self) -> "None":
        smooth_array = np.zeros(self.limber_power_spectrum_l_z.shape)
        probe_cfg = self.forecast_config.probe_configs["Void"]
        k_void_cut = probe_cfg.specific_settings["void_kcut_invMpc"]
        k_void_cut_width = probe_cfg.specific_settings["void_kcut_width_invMpc"]
        k_min = k_void_cut - k_void_cut_width
        k_max = k_void_cut + k_void_cut_width
        k_lz = np.expand_dims((self.l_bin_centers + 0.5), 1) / self.cosmology.r_z
        for z_idx, myz in enumerate(self.z_array):
            smooth_array[:, z_idx] = 1 - nu.smoothstep(x=k_lz[:, z_idx], x_min=k_min, x_max=k_max, N=3)
        self.limber_power_spectrum_l_z *= smooth_array

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", probe1: "str", probe2: "str",
                 root: "str" = '/') -> "AngularCoefficient":
        h5cl = H5Cl(probe1, probe2)
        return h5cl.load(file, root)

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = '/') -> "None":
        h5cl = H5Cl(self.probe1, self.probe2)
        h5cl.save(self, file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", probe1: "str" = None, probe2: "str" = None,
                     root: "str" = '/') -> "None":
        if probe1 is None:
            probe1 = self.probe1
        if probe2 is None:
            probe2 = self.probe2
        h5cl = H5Cl(probe1, probe2)
        h5cl.openFile(file=file, mode='r', root=root)
        h5cl.writeToObject(self)
        h5cl.closeFile()

    def __eq__(self, other: "AngularCoefficient") -> "bool":
        return all([
            np.all(self.c_lij == other.c_lij),
            np.all(self.l_bin_centers == other.l_bin_centers),
            np.all(self.l_bin_widths == other.l_bin_widths),
        ])

    def plot(self, what: "str" = "cl", axes: "np.ndarray[plt.Axes]" = None,
             **kwargs) -> "Tuple[plt.Figure, np.ndarray[plt.Axes]]":
        kwds = {
            "label": self.obs_key,
            "triangular": self.is_auto_correlation
        }
        kwds.update(kwargs)

        if what == "cl":
            if "logx" not in kwds:
                kwds["logx"] = True
            x = self.l_bin_centers
            y = np.transpose(self.c_lij, axes=(1, 2, 0))
            fig, axes = grid_plot(x=x, y=y, axes=axes, **kwds)
        elif what == "kernel":
            x = self.z_array
            y = self.kernel.k_ijz
            fig, axes = grid_plot(x=x, y=y, axes=axes, **kwds)
        else:
            raise ValueError(f"Unrecognized option {what}")

        return fig, axes


class AngularCoefficientsCollector:
    obs_list: "List[str]"
    obs_combinations: "List[Tuple[str]]"
    cl_file_dataset: "h5py.File"
    cosmology: "Cosmology"
    fiducial_cosmology: "Cosmology"
    l_bin_centers: "np.ndarray"
    forecast_config: "ForecastConfig"
    cl_dict: "Dict[str, AngularCoefficient]"
    weight_dict: "Dict[str, wfs.WeightFunction]"
    phys_params: "PhysicalParametersCollection"

    def __init__(self, phys_params: "PhysicalParametersCollection" = None, cosmology: "Cosmology" = None,
                 fiducial_cosmology: "Cosmology" = None, forecast_config: "ForecastConfig" = None,
                 angular_config: "AngularConfig" = None, full_output: "bool" = False):
        self.phys_params = phys_params
        self.forecast_config = forecast_config
        self.angular_config = angular_config
        self.cl_dict = {}
        self.cosmology = cosmology
        self.fiducial_cosmology = fiducial_cosmology
        self.z_grid = None
        self.weight_dict = {}
        self.full_output = full_output
        self._attrs_excluded_from_equality = set()
        self.ready_to_compute = False
        if self.cosmology is not None:
            self.z_grid = self.cosmology.z_grid

    @property
    def probes(self) -> "List[str]":
        try:
            return self.forecast_config.present_probes
        except AttributeError:
            return list(self.weight_dict.keys())

    @property
    def probes_combinations(self) -> "List[Tuple[str]]":
        try:
            return self.forecast_config.probes_combinations
        except AttributeError:
            pass

    @property
    def short_keys(self) -> "List[str]":
        return [ClKey(key).toShortKey().key for key in self.cl_dict]

    @property
    def verbose(self):
        return self.forecast_config.verbose

    def __getitem__(self, key: "str") -> "AngularCoefficient":
        return self.cl_dict[ClKey(key).toLongKey().key]

    def evaluateAngularCoefficients(self) -> "None":
        if not self.ready_to_compute:
            raise ClError("Must do setup before computing Cls!")
        if self.verbose:
            logger.info('Evaluating angular correlation coefficients')
        for obs_key in self.cl_dict:
            if self.verbose:
                logger.info(f'Computing Cl {obs_key}')
            t0 = time.time()
            self.cl_dict[obs_key].evaluateAngularCorrelation()
            tf = time.time()
            if self.verbose:
                logger.info(f'Cl {obs_key} elapsed time: {fm.string_time_format(tf-t0)}')

    def setUp(self, densities: "Dict[str, RedshiftDensity]"):
        if not isinstance(self.z_grid, np.ndarray):
            raise TypeError(f'invalid redshift grid {self.z_grid} for cl computation setup')
        for probe in self.probes:
            probe_cfg = self.forecast_config.probe_configs[probe]
            nuisance_params = self.phys_params.getNuisanceParametersForProbe(probe)
            w = wfs.weight_function_for_probe(probe, probe_config=probe_cfg,
                                              nuisance_params=nuisance_params,
                                              cosmology=self.cosmology,
                                              fiducial_cosmology=self.fiducial_cosmology,
                                              angular_config=self.angular_config)
            w.density = densities[probe]
            w.setUp()
            w.evaluateOverRedshiftGrid(self.z_grid)
            self.weight_dict[probe] = w
        for probe_1, probe_2 in self.probes_combinations:
            obs_key = gu.get_probes_combination_key(probe_1, probe_2)
            kernel = kfs.KernelFunction(probe_1, probe_2,
                                        weight1=self.weight_dict[probe_1], weight2=self.weight_dict[probe_2],
                                        cosmology=self.cosmology)
            self.cl_dict[obs_key] = AngularCoefficient(probe1=probe_1, probe2=probe_2, kernel=kernel,
                                                       forecast_config=self.forecast_config,
                                                       angular_config=self.angular_config)
            self.cl_dict[obs_key].setUp()

        self.ready_to_compute = True

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = '/') -> "None":
        h5cl_coll = H5ClCollection()
        h5cl_coll.save(self, file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", root: "str" = '/') -> "None":
        h5cl_coll = H5ClCollection()
        h5cl_coll.openFile(file=file, mode='r', root=root)
        h5cl_coll.writeToObject(self)
        h5cl_coll.closeFile()

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", root: "str" = '/') -> "AngularCoefficientsCollector":
        h5cl_coll = H5ClCollection()
        return h5cl_coll.load(file, root)

    def __eq__(self, other: "AngularCoefficientsCollector") -> "bool":
        return all([self.cl_dict[key] == other.cl_dict[key] for key in self.cl_dict])
        

class H5Cl(AbstractH5FileIO):
    def __init__(self, probe1: "str" = None, probe2: "str" = None, **kwargs):
        super(H5Cl, self).__init__(**kwargs)
        self.probe1 = probe1
        self.probe2 = probe2
        self.builder_func = AngularCoefficient

    @property
    def main_path_rel_to_root(self) -> "str":
        if isinstance(self.probe1, str) and isinstance(self.probe2, str):
            return gu.get_probes_combination_key(self.probe1, self.probe2)
        else:
            raise TypeError(f'probe1 and probe2 must be both str, not {type(self.probe1)} and {type(self.probe2)}')

    @property
    def absolute_main_path(self) -> "str":
        return join(self.root_path, self.main_path_rel_to_root)

    def readBuildingData(self) -> "None":
        self.build_data['probe1'] = self.probe1
        self.build_data['probe2'] = self.probe2

    def writeToObject(self, obj: "AngularCoefficient") -> "None":
        super(H5Cl, self).writeToObject(obj)
        if self.main_path_rel_to_root is None:
            raise ValueError('cannot load Cl from file without knowing main path')
        main_grp = self.root[f'{self.main_path_rel_to_root}']
        obj.c_lij = main_grp['c_lij'][()]
        obj.l_bin_centers = main_grp['l_bin_centers'][()]
        if 'l_bin_widths' in main_grp:
            obj.l_bin_widths = main_grp['l_bin_widths'][()]
        else:
            logger.warning('BEWARE: l_bin_widths absent from Cls file')
        ker_path = f'{self.absolute_main_path}/kernel'
        if ker_path in self.root:
            h5k = kfs.H5Kernel()
            obj.kernel = h5k.load(self.file_path, root=ker_path)
            w1_root_path = self.findGroup(f"weight_functions/{self.probe1}")[0]
            w2_root_path = self.findGroup(f"weight_functions/{self.probe2}")[0]
            h5w1 = wfs.H5WeightFunction()
            h5w2 = wfs.H5WeightFunction()
            obj.kernel.weight1 = h5w1.load(self.file_path, root=w1_root_path)
            obj.kernel.weight2 = h5w2.load(self.file_path, root=w2_root_path)

    def writeObjectToFile(self, obj: "AngularCoefficient") -> "None":
        super(H5Cl, self).writeObjectToFile(obj)
        self.createDataset(name=f'{self.main_path_rel_to_root}/c_lij', data=obj.c_lij)
        self.createDataset(name=f'{self.main_path_rel_to_root}/l_bin_centers', data=obj.l_bin_centers)
        if obj.l_bin_widths is not None:
            self.createDataset(name=f'{self.main_path_rel_to_root}/l_bin_widths', data=obj.l_bin_widths)
        else:
            logger.warning(f'BEWARE: l_bin_widths absent from Cl {obj.obs_key}')
        if obj.kernel is not None:
            h5k = kfs.H5Kernel()
            h5k.save(obj.kernel, self.file_path, root=f'{self.absolute_main_path}/kernel')
        if obj.weight1 is not None:
            w1_root_path = f'weight_functions/{self.probe1}'
            if w1_root_path not in self._h5file:
                h5w = wfs.H5WeightFunction()
                h5w.save(obj.weight1, self.file_path, root=w1_root_path)
            w2_root_path = f'weight_functions/{self.probe2}'
            if w2_root_path not in self._h5file:
                h5w = wfs.H5WeightFunction()
                h5w.save(obj.weight2, self.file_path, root=w2_root_path)


class H5ClCollection(AbstractH5FileIO):
    def __init__(self, **kwargs):
        super(H5ClCollection, self).__init__(**kwargs)
        self.builder_func = AngularCoefficientsCollector

    def writeToObject(self, obj: "AngularCoefficientsCollector") -> "None":
        super(H5ClCollection, self).writeToObject(obj)
        if 'z_grid' in self.root:
            obj.z_grid = self.root['z_grid'][()]
        elif 'z_array' in self.root:
            obj.z_grid = self.root['z_array'][()]

        if 'weight_functions' in self.root:
            w_grp = self.root['weight_functions']
            w_grp_path = w_grp.name
            for w_probe in w_grp:
                h5w = wfs.H5WeightFunction()
                obj.weight_dict[w_probe] = h5w.load(file=self.file_path, root=f'{w_grp_path}/{w_probe}')
        else:
            logger.warning('weight functions absent from file, skipping them')

        cls_grp = self.root['cls']
        cls_grp_path = cls_grp.name
        obj.weight_dict = {}
        for cl_kind in cls_grp:
            p1, p2 = gu.get_probes_from_comb_key(cl_kind)
            h5cl = H5Cl(probe1=p1, probe2=p2)
            cl_obj = h5cl.load(file=self.file_path, root=f'{cls_grp_path}')
            cl_obj: "AngularCoefficient"
            if p1 not in obj.weight_dict:
                obj.weight_dict[p1] = cl_obj.weight1
            if p2 not in obj.weight_dict:
                obj.weight_dict[p2] = cl_obj.weight2

            obj.cl_dict[cl_kind] = cl_obj

    def writeObjectToFile(self, obj: "AngularCoefficientsCollector") -> "None":
        super(H5ClCollection, self).writeObjectToFile(obj)
        if obj.z_grid is not None:
            self.createDataset(name='z_grid', data=obj.z_grid)
        cls_grp_path = join(self.root_path, 'cls')
        self.createGroup(name=cls_grp_path)
        for cl_kind, cl in obj.cl_dict.items():
            cl.saveToHDF5(self.file_path, root=cls_grp_path)

        weights_path = join(self.root_path, 'weight_functions')
        for w_probe, w in obj.weight_dict.items():
            w_path = f'{weights_path}/{w_probe}'
            if w_path not in self._h5file:
                w.saveToHDF5(self.file_path, root=w_path)



