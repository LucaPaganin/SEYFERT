from pathlib import Path
from seyfert.cosmology.redshift_density import RedshiftDensity
from abc import ABC, abstractmethod
from scipy import integrate, interpolate
import numpy as np
import re
from os.path import join
import logging
import sys
from typing import Union, Dict

from seyfert.cosmology.parameter import PhysicalParameter
from seyfert.config.main_config import AngularConfig
from seyfert.cosmology.cosmology import Cosmology
from seyfert.cosmology.bias import Bias
from seyfert.file_io.hdf5_io import AbstractH5FileIO
from seyfert.config.probe_config import ProbeConfig
from seyfert.utils import filesystem_utils as fsu

this_module = sys.modules[__name__]
logger = logging.getLogger(__name__)


class WeightFunction(ABC):
    cosmology: "Cosmology"
    density: "RedshiftDensity"
    w_bin_z: "np.ndarray"
    z_grid: "np.ndarray"
    probe_config: "ProbeConfig"
    bias: "Bias"
    nuisance_params: "Dict[str, PhysicalParameter]"

    def __init__(self,
                 probe_config: "ProbeConfig" = None, nuisance_params: "Dict[str, PhysicalParameter]" = None,
                 cosmology: "Cosmology" = None, fiducial_cosmology: "Cosmology" = None,
                 angular_config: "AngularConfig" = None):
        self.w_bin_z = None
        self.shot_noise_factor = 1
        self.cosmology = cosmology
        self.fiducial_cosmology = fiducial_cosmology
        self.probe_config = probe_config
        self.angular_config = angular_config
        self.nuisance_params = nuisance_params
        self.density = None
        self.z_grid = None
        self.bias = None
        self._attrs_excluded_from_equality = {'cosmology',
                                              'fiducial_cosmology',
                                              'forecast_config',
                                              'angular_config'}

    @property
    def is_evaluated(self) -> "bool":
        return self.w_bin_z is not None

    @property
    def probe(self):
        return probe_from_weight_function_cls_name(self.__class__.__name__)

    @property
    def z_min(self):
        return self.density.z_min

    @property
    def z_max(self):
        return self.density.z_max

    @property
    def z_bin_centers(self):
        return self.density.z_bin_centers

    @property
    def z_bin_edges(self):
        return self.density.z_bin_edges

    @property
    def n_bins(self):
        if hasattr(self.density, 'n_bins'):
            return self.density.n_bins
        else:
            return self.w_bin_z.shape[0]

    @property
    def n_i_z(self) -> "np.ndarray":
        return self.density.norm_density_iz

    @property
    def H_z(self) -> "np.ndarray":
        return self.cosmology.H_z

    @property
    def r_tilde_z(self) -> "np.ndarray":
        return self.cosmology.r_tilde_z

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", root: "str" = '/') -> "WeightFunction":
        h5w = H5WeightFunction()
        w = h5w.load(file, root)
        return w

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = '/') -> "None":
        h5w = H5WeightFunction()
        h5w.save(self, file, root)

    def loadFromHDF5(self, file: Union[str, Path], root: str = '/') -> "None":
        h5w = H5WeightFunction()
        h5w.openFile(file, root)
        h5w.writeToObject(self)
        h5w.closeFile()

    def setUp(self):
        if self.density is None:
            self.density = RedshiftDensity.fromHDF5(self.probe_config.density_init_file)
            self.density.setUp()

    @abstractmethod
    def evaluateOverRedshiftGrid(self, z_grid: "np.ndarray"):
        self.z_grid = z_grid
        if self.density.norm_density_iz is None:
            self.density.evaluate(self.z_grid)
        self.w_bin_z = np.zeros((self.n_bins, len(self.z_grid)))

    def __eq__(self, other: "WeightFunction") -> bool:
        return np.all(self.z_grid == other.z_grid) and np.all(self.w_bin_z == other.w_bin_z)


# noinspection PyTypeChecker
class LensingWeightFunction(WeightFunction):

    def __init__(self, **kwargs):
        super(LensingWeightFunction, self).__init__(**kwargs)
        self.lensing_efficiency_integration_method = 'simpson'
        self.sigma_epsilon = 0.3
        self.shot_noise_factor = self.sigma_epsilon ** 2
        if self.angular_config is not None:
            self.lensing_efficiency_integration_method = self.angular_config.lensing_efficiency_integration_method

    @property
    def Omm(self) -> "float":
        return self.cosmology.Omm

    @property
    def c_km_s(self) -> "float":
        return self.cosmology.c_km_s

    @property
    def H0(self) -> "float":
        return self.cosmology.H0

    def computeLensingEfficiencyAtBin(self, i: int, z: Union[float, np.ndarray]) -> "Union[float, np.ndarray]":
        if self.lensing_efficiency_integration_method == "quad":

            def integrand(x, bin_idx: int, start_z: float):
                return self.density.computeNormalizedDensityAtBinAndRedshift(bin_idx, x) * \
                       (1 - self.cosmology.computeComovingDistance(start_z) / self.cosmology.computeComovingDistance(x))

            if isinstance(z, float):
                result, _ = integrate.quad(integrand, z, self.z_max, args=(i, z))
            elif isinstance(z, np.ndarray):
                result = np.zeros(z.shape)
                for z_idx, myz in enumerate(z):
                    result[z_idx] = integrate.quad(integrand, myz, self.z_max, args=(i, myz))[0]
            else:
                raise TypeError(f'Expected float or np.ndarray, got {type(z)}')
        elif self.lensing_efficiency_integration_method == "simpson":
            if isinstance(z, float):
                # start to integrate from nearest value in grid
                z_idx = np.argmin(np.abs(z - self.z_grid))
                integ_array = self.density.norm_density_iz[i, :] * \
                              (1 - self.cosmology.r_z[z_idx] / self.cosmology.r_z)
                result = integrate.simps(y=integ_array[z_idx:], x=z[z_idx:])
            elif isinstance(z, np.ndarray):
                result = np.zeros(z.shape)
                for z_idx, myz in enumerate(z):
                    integ_array = self.density.norm_density_iz[i, :] * \
                                  (1 - self.cosmology.r_z[z_idx] / self.cosmology.r_z)
                    result[z_idx] = integrate.simps(y=integ_array[z_idx:], x=z[z_idx:])
            else:
                raise TypeError(f'Expected float or np.ndarray, got {type(z)}')
        else:
            raise ValueError(f'Unsupported integration method {self.lensing_efficiency_integration_method}. '
                             f'Available options are "quad" and "simpson".')
        return result

    def computeIntrinsicAlignmentContribution(self) -> "np.ndarray":
        A_IA = self.nuisance_params['aIA'].current_value
        C_IA = self.nuisance_params['cIA'].current_value
        etaIA = self.nuisance_params['etaIA'].current_value
        betaIA = self.nuisance_params['betaIA'].current_value

        scaled_lum_z, scaled_lum_L = np.loadtxt(fsu.default_data_dir() / 'scaledmeanlum-E2Sa.dat', unpack=True)
        scaled_lum_spline = interpolate.InterpolatedUnivariateSpline(scaled_lum_z, scaled_lum_L, k=1)
        scaled_L_z_grid = scaled_lum_spline(self.z_grid)
        F_IA_z = (1 + self.z_grid) ** etaIA * scaled_L_z_grid ** betaIA

        D_z = self.cosmology.growth_factor_z

        w_IA_i_z = (A_IA * C_IA * F_IA_z * self.Omm * self.H_z * self.n_i_z) / (D_z * self.c_km_s)

        return w_IA_i_z

    def computeLensingEfficiency(self, z: np.ndarray) -> "np.ndarray":
        lensing_efficiency = np.zeros((self.n_bins, len(z)))
        for i in range(self.n_bins):
            lensing_efficiency[i, :] = self.computeLensingEfficiencyAtBin(i, self.z_grid)

        return lensing_efficiency

    def evaluateOverRedshiftGrid(self, z_grid: "np.ndarray") -> None:
        super(LensingWeightFunction, self).evaluateOverRedshiftGrid(z_grid)
        wl_eff = self.computeLensingEfficiency(z_grid)
        # Enjoy numpy broadcasting
        self.w_bin_z = 1.5 * self.Omm * (1 + self.z_grid) * (self.H0 / self.c_km_s) * self.r_tilde_z * wl_eff

        if self.probe_config.specific_settings['include_IA']:
            w_IA_iz = self.computeIntrinsicAlignmentContribution()
            self.w_bin_z -= w_IA_iz


class WeightFunctionWithBias(WeightFunction, ABC):
    bias: "Bias"

    def __init__(self, **kwargs):
        super(WeightFunctionWithBias, self).__init__(**kwargs)
        self.bias = None

    def setUp(self):
        super(WeightFunctionWithBias, self).setUp()
        if self.nuisance_params is not None:
            self.setupBias()
        else:
            logger.info('nuisance parameters not passed, not building bias')

    @abstractmethod
    def setupBias(self) -> "None":
        self.bias = Bias(probe=self.probe, z_bin_edges=self.z_bin_edges)
        self.bias.loadFromHDF5(self.probe_config.bias_init_file)
        for bias_par in self.bias.nuisance_parameters.keys():
            curr_val = self.nuisance_params[bias_par].current_value
            self.bias.nuisance_parameters[bias_par] = curr_val

        if self.bias.model_name in {'piecewise', 'constant', 'euclid_flagship_photo'}:
            self.bias.initBiasModel()

    def evaluateOverRedshiftGrid(self, z_grid: "np.ndarray") -> "None":
        super(WeightFunctionWithBias, self).evaluateOverRedshiftGrid(z_grid)
        self.setupBias()
        self.bias.evaluateBias(z_grid=self.z_grid)
        self.w_bin_z = (self.cosmology.H_z / self.cosmology.c_km_s) * self.density.norm_density_iz * self.bias.b_i_z


class VoidWeightFunction(WeightFunctionWithBias):
    bias: Bias

    def __init__(self, **kwargs):
        super(VoidWeightFunction, self).__init__(**kwargs)

    def setupBias(self) -> None:
        super(VoidWeightFunction, self).setupBias()
        if self.bias.model_name == 'vdn_void':
            self.bias.initBiasModel(cosmology=self.cosmology)
        elif self.bias.model_name == 'fiducial_growth_void':
            self.bias.initBiasModel(fiducial_cosmology=self.fiducial_cosmology)
        else:
            raise NotImplementedError(f'bias model {self.bias.model_name} not implemented')


class GalaxyClusteringWeightFunction(WeightFunctionWithBias, ABC):

    def __init__(self, **kwargs):
        super(GalaxyClusteringWeightFunction, self).__init__(**kwargs)

    def setupBias(self) -> None:
        super(GalaxyClusteringWeightFunction, self).setupBias()


class PhotometricGalaxyWeightFunction(GalaxyClusteringWeightFunction):

    def __init__(self, **kwargs):
        super(PhotometricGalaxyWeightFunction, self).__init__(**kwargs)

    def setupBias(self) -> None:
        super(PhotometricGalaxyWeightFunction, self).setupBias()


class SpectroscopicGalaxyWeightFunction(GalaxyClusteringWeightFunction):

    def __init__(self, **kwargs):
        super(SpectroscopicGalaxyWeightFunction, self).__init__(**kwargs)

    def setupBias(self) -> None:
        super(SpectroscopicGalaxyWeightFunction, self).setupBias()


class H5WeightFunction(AbstractH5FileIO):
    def __init__(self, **kwargs):
        super(H5WeightFunction, self).__init__(**kwargs)
        self.builder_func = weight_function_for_probe

    def readBuildingData(self) -> "None":
        self.build_data['probe'] = self.attrs['probe']

    def writeToObject(self, obj: "WeightFunction") -> "None":
        super(H5WeightFunction, self).writeToObject(obj)
        try:
            obj.z_grid = self.root['z_grid'][()]
            obj.w_bin_z = self.root['w_bin_z'][()]
        except KeyError:
            logger.warning(f'z_grid and w_bin_z absent from weight function file {self.file_path}')
        if 'density' in self.root:
            obj.density = RedshiftDensity.fromHDF5(self.file_path, root=join(self.root_path, 'density'))
        else:
            logger.warning(f'density absent from h5 file')
        if isinstance(obj, WeightFunctionWithBias):
            if 'bias' in self.root:
                obj.bias = Bias.fromHDF5(self.file_path, root=join(self.root_path, 'bias'))
            else:
                logger.warning(f'bias absent from h5 file')

    def writeObjectToFile(self, obj: "WeightFunction") -> "None":
        self.attrs['probe'] = obj.probe
        if obj.z_grid is not None:
            self.createDataset(name='z_grid', data=obj.z_grid)
        if obj.w_bin_z is not None:
            self.createDataset(name='w_bin_z', data=obj.w_bin_z)
        if obj.density is not None:
            obj.density.saveToHDF5(self.file_path, root=join(self.root_path, 'density'))
        else:
            logger.warning(f'density is None, not saving it')
        if isinstance(obj, WeightFunctionWithBias):
            if obj.bias is not None:
                obj.bias.saveToHDF5(self.file_path, root=join(self.root_path, 'bias'))
            else:
                logger.warning(f'bias is None, not saving it')


def weight_cls_name_from_obs(obs_string):
    return f'{obs_string}WeightFunction'


def probe_from_weight_function_cls_name(weight_function_name):
    regex = re.compile('(?P<obs_string>[a-zA-Z]+)WeightFunction')
    match = regex.match(weight_function_name)
    if not match:
        raise Exception(f'Cannot extract observable from density name {weight_function_name} '
                        f'from pattern {regex.pattern}')
    return match.group('obs_string')


def weight_function_for_probe(probe: str, probe_config: "ProbeConfig" = None,
                              nuisance_params: Dict[str, PhysicalParameter] = None,
                              cosmology: Cosmology = None, fiducial_cosmology: Cosmology = None,
                              angular_config: AngularConfig = None) -> WeightFunction:
    w_name = weight_cls_name_from_obs(probe)
    w = getattr(this_module, w_name)(cosmology=cosmology,
                                     nuisance_params=nuisance_params,
                                     probe_config=probe_config,
                                     fiducial_cosmology=fiducial_cosmology,
                                     angular_config=angular_config)
    return w
