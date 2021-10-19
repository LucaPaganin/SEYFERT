"""Module hosting the bias class and the implemented bias models.
"""

from typing import TYPE_CHECKING
from scipy import integrate
from scipy import interpolate
from seyfert.numeric import general
from seyfert.utils import general_utils as gu
import logging
from abc import ABC, abstractmethod
from seyfert.file_io.hdf5_io import AbstractH5FileIO
from pathlib import Path
import numpy as np
import re
from typing import Dict, Union, Set
if TYPE_CHECKING:
    from seyfert.cosmology.cosmology import Cosmology


logger = logging.getLogger(__name__)

BIAS_MODELS = ['piecewise', 'vdn_void', 'fiducial_growth_void']


class BiasError(Exception):
    pass


class Bias:
    """
    Class managing the bias.

    :param model: instance of :class:`.BiasModel`. It defines the model with which to compute the
     bias as a function of the redshift.
    :param nuisance_parameters: a Dict[str, float] of nuisance parameters for the bias. It should contain the updated
     values of the nuisance parameters, since these may be let free to vary.
    :param additional_parameters: a Dict[str, float] of additional parameters for the bias. These are kept constant and not
     let free to vary as it happens for nuisance parameters

    """
    model: "BiasModel"
    nuisance_parameters: "Dict[str, float]"
    additional_parameters: "Dict[str, float]"
    z_grid: "np.ndarray"
    b_i_z: "np.ndarray"
    model_name: "str"
    _attrs_excluded_from_equality: "Set"

    def __init__(self, probe: "str" = None, z_bin_edges: "np.ndarray" = None):
        self.probe = probe
        self.nuisance_parameters = None
        self.additional_parameters = None
        self.z_grid = None
        self.b_i_z = None
        self.model = None
        self.model_name = None
        self.z_bin_edges = z_bin_edges
        self._attrs_excluded_from_equality = {'model'}

    @property
    def has_output(self) -> "bool":
        return self.b_i_z is not None

    def initBiasModel(self, **kwargs) -> "None":
        """Method for initializing the bias model.

        :param kwargs: generic keyword arguments parameters. These depend on the particular model that is being
         instantiated.
        """
        if self.model_name == 'piecewise':
            self.model = PiecewiseBias(name=self.model_name, nuisance_parameters=self.nuisance_parameters,
                                       z_bin_edges=self.z_bin_edges)
        elif self.model_name == 'constant':
            self.model = ConstantBias(name=self.model_name, nuisance_parameters=self.nuisance_parameters,
                                      z_bin_edges=self.z_bin_edges)
        elif self.model_name == 'euclid_flagship_photo':
            self.model = EuclidFlagshipGCphBias(name=self.model_name, nuisance_parameters=self.nuisance_parameters,
                                                z_bin_edges=self.z_bin_edges)
        elif self.model_name == 'vdn_void':
            if 'cosmology' not in kwargs:
                raise BiasError('cosmology is required for Vdn void bias model')
            self.model = VdnVoidBias(cosmology=kwargs['cosmology'], name=self.model_name,
                                     z_bin_edges=self.z_bin_edges,
                                     nuisance_parameters=self.nuisance_parameters,
                                     additional_parameters=self.additional_parameters)

        elif self.model_name == 'fiducial_growth_void':
            if 'fiducial_cosmology' not in kwargs:
                raise BiasError('fiducial cosmology is required for fiducial growth void bias model')
            self.model = FiducialGrowthVoidBias(cosmology=kwargs['fiducial_cosmology'], name=self.model_name,
                                                nuisance_parameters=self.nuisance_parameters,
                                                z_bin_edges=self.z_bin_edges)
        else:
            raise KeyError(f"Invalid bias model name {self.model_name}")

    def evaluateBias(self, z_grid: "np.ndarray") -> "None":
        self.z_grid = z_grid
        self.b_i_z = self.model.computeBias(self.z_grid)

    @staticmethod
    def fromHDF5(file: "Union[str, Path]", root: "str" = '/') -> "Bias":
        h5bias = H5Bias()
        return h5bias.load(file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", root: "str" = '/') -> "None":
        h5bias = H5Bias()
        h5bias.openFile(file=file, mode='r', root=root)
        h5bias.writeToObject(self)
        h5bias.closeFile()

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = '/') -> "None":
        h5bias = H5Bias()
        h5bias.save(self, file, root)

    def __eq__(self, other: "Bias") -> "bool":
        return gu.compare_objects(self, other, self._attrs_excluded_from_equality)


class H5Bias(AbstractH5FileIO):
    def __init__(self, **kwargs):
        super(H5Bias, self).__init__(**kwargs)
        self.builder_func = Bias

    def writeObjectToFile(self, obj: "Bias") -> "None":
        if obj.z_bin_edges is not None:
            self.createDataset(name='z_bin_edges', data=obj.z_bin_edges)
        self.attrs['probe'] = obj.probe
        bias_model_grp = self.createGroup(name='bias_model')
        bias_model_grp.attrs['model_name'] = obj.model_name
        add_params_grp = self.createGroup(name='additional_parameters', base_grp=bias_model_grp)
        self.createDatasets(obj.additional_parameters, base_grp=add_params_grp)
        nuis_params_grp = self.createGroup(name='nuisance_parameters', base_grp=bias_model_grp)
        self.createDatasets(obj.nuisance_parameters, base_grp=nuis_params_grp)
        if obj.z_grid is not None:
            self.createDataset(name='z_grid', data=obj.z_grid)
        if obj.b_i_z is not None:
            self.createDataset(name='b_i_z', data=obj.b_i_z)

    def writeToObject(self, obj: "Bias") -> "None":
        super(H5Bias, self).writeToObject(obj)
        obj.probe = self.attrs['probe']
        obj.model_name = self.root['bias_model'].attrs['model_name']
        if 'z_bin_edges' in self.root:
            obj.z_bin_edges = self.root['z_bin_edges'][()]
        obj.additional_parameters = {
            key: dset[()] for key, dset in self.root['bias_model/additional_parameters'].items()
        }
        obj.nuisance_parameters = {
            key: dset[()] for key, dset in self.root['bias_model/nuisance_parameters'].items()
        }
        try:
            obj.z_grid = self.root['z_grid'][()]
            if 'b_i_z' in self.root:
                obj.b_i_z = self.root['b_i_z'][()]
        except KeyError:
            pass


class BiasModel(ABC):
    z_bin_edges: "np.ndarray"
    cosmology: "Cosmology"

    def __init__(self, name: "str" = None, z_bin_edges: "np.ndarray" = None,
                 cosmology: "Cosmology" = None,
                 nuisance_parameters: "Dict[str, float]" = None,
                 additional_parameters: "Dict[str, float]" = None):
        self.model_name = name
        self.nuisance_parameters = nuisance_parameters
        self.additional_parameters = additional_parameters
        self._attrs_excluded_from_equality = set()
        self.z_bin_edges = z_bin_edges
        self.cosmology = cosmology

    @property
    def z_bin_centers(self) -> "np.ndarray":
        return (self.z_bin_edges[:-1] + self.z_bin_edges[1:]) / 2

    @property
    def n_bins(self) -> "int":
        return len(self.z_bin_edges) - 1

    @abstractmethod
    def computeBias(self, z_grid: "np.ndarray") -> "np.ndarray":
        pass

    def __eq__(self, other: "BiasModel"):
        return gu.compare_objects(self, other, self._attrs_excluded_from_equality)


class ConstantBias(BiasModel):
    def __init__(self, **kwargs):
        super(ConstantBias, self).__init__(**kwargs)

    def computeBias(self, z_grid: "np.ndarray") -> "np.ndarray":
        b_i = np.array(list(self.nuisance_parameters.values()))
        if len(b_i) != self.n_bins:
            raise BiasError(f'number of bias values {len(b_i)} differs from n_bins {self.n_bins}')
        return np.repeat(b_i[:, np.newaxis], len(z_grid), axis=1)


class PiecewiseBias(BiasModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def computeBias(self, z_grid: "np.ndarray") -> "np.ndarray":
        cond_list, vals_list = self.getConditionAndValuesLists(z_grid)
        b_z = np.piecewise(z_grid, cond_list, vals_list)
        b_i_z = np.repeat(b_z[np.newaxis, :], self.n_bins, axis=0)

        return b_i_z

    def getSortedBiasValues(self) -> "np.ndarray":
        par_names = list(self.nuisance_parameters.keys())
        digit_regex = re.compile(r'\d+')

        def _get_bias_idx(s) -> "int":
            match = digit_regex.search(s)
            return int(s[match.start():match.end()])

        return np.array([self.nuisance_parameters[name] for name in sorted(par_names, key=_get_bias_idx)])

    def getConditionAndValuesLists(self, z_grid: "np.ndarray"):
        b_i = self.getSortedBiasValues()
        if len(b_i) != self.n_bins:
            raise BiasError(f'number of bias values {len(b_i)} differs from n_bins {self.n_bins}')
        cond_list = [(z_grid >= self.z_bin_edges[i]) & (z_grid <= self.z_bin_edges[i+1]) for i in range(self.n_bins)]
        vals_list = list(b_i)
        # Constant extrapolation out of the grid in case the grid is broader than bin edges
        # on the left
        cond_list.insert(0, z_grid < self.z_bin_edges[0])
        vals_list.insert(0, b_i[0])
        # on the right
        cond_list.append(z_grid > self.z_bin_edges[-1])
        vals_list.append(b_i[-1])

        return cond_list, vals_list


class EuclidFlagshipGCphBias(BiasModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def Aph(self) -> "float":
        return self.nuisance_parameters["Aph"]

    @property
    def Bph(self) -> "float":
        return self.nuisance_parameters["Bph"]

    @property
    def Cph(self) -> "float":
        return self.nuisance_parameters["Cph"]

    @property
    def Dph(self) -> "float":
        return self.nuisance_parameters["Dph"]

    def computeBias(self, z_grid: "np.ndarray") -> "np.ndarray":
        b_z = self.Aph + self.Bph / (1 + np.exp(-self.Cph * (z_grid - self.Dph)))
        b_i_z = np.repeat(b_z[np.newaxis, :], self.n_bins, axis=0)

        return b_i_z


class FiducialGrowthVoidBias(BiasModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def computeBias(self, z_grid: "np.ndarray") -> "np.ndarray":
        if not np.all(z_grid == self.cosmology.z_grid):
            raise BiasError("Fiducial Cosmology grid not consistent with the provided one for bias computation")

        vb_0 = self.nuisance_parameters['voidbias0']
        fullgrid_b_z = vb_0 / self.cosmology.growth_factor_z
        spline = interpolate.InterpolatedUnivariateSpline(x=z_grid, y=fullgrid_b_z, k=3)
        b_i = spline(self.z_bin_centers)
        b_i_z = np.repeat(b_i[:, np.newaxis], len(z_grid), axis=1)

        return b_i_z


class VdnVoidBias(BiasModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def z_grid(self):
        return self.cosmology.z_grid

    @property
    def R_min_Mpc(self) -> "float":
        return self.additional_parameters['R_min_Mpc']

    @property
    def R_max_Mpc(self) -> float:
        return self.additional_parameters['R_max_Mpc']

    @property
    def n_R(self) -> int:
        return int(self.additional_parameters['R_grid_size'])

    @property
    def delta_c0(self) -> float:
        return self.additional_parameters['delta_c0']

    @property
    def delta_v0(self) -> float:
        return self.additional_parameters['delta_v0']

    def computeBias(self, z_grid: "np.ndarray") -> "np.ndarray":
        if not np.all(z_grid == self.z_grid):
            raise BiasError('provided redshift grid is not the same as the cosmology one')
        R = np.linspace(self.R_min_Mpc, self.R_max_Mpc, self.n_R)
        sigmaR = self.cosmology.computeSigmaR(R)
        sigma_R_z = np.expand_dims(sigmaR, 1)
        delta_c_z = self.delta_c0 / self.cosmology.growth_factor_z
        delta_v_z = self.delta_v0 / self.cosmology.growth_factor_z
        D = np.abs(delta_v_z) / (np.abs(delta_v_z) + delta_c_z)
        x = (D / np.abs(delta_v_z)) * sigma_R_z
        nu = np.abs(delta_v_z) / sigma_R_z

        void_bias_R_z = 1 + (nu ** 2 - 1) / delta_v_z + (delta_v_z * D) / (4 * delta_c_z ** 2 * nu ** 2)
        Vdn_size_func_R_z = self.computeVdnSizeFunction(R, D, x, delta_v_z, sigmaR)

        weighted_bias_z = np.zeros(self.z_grid.shape)
        for z_idx, myz in enumerate(self.z_grid):
            norm_z = integrate.simps(Vdn_size_func_R_z[:, z_idx], R)
            weighted_bias_z[z_idx] = integrate.simps((void_bias_R_z * Vdn_size_func_R_z)[:, z_idx], R)
            weighted_bias_z[z_idx] /= norm_z

        spline = interpolate.InterpolatedUnivariateSpline(x=self.z_grid, y=weighted_bias_z, k=3)
        b_i = spline(self.z_bin_centers)
        b_i_z = np.repeat(b_i[:, np.newaxis], len(self.z_grid), axis=1)
        return b_i_z

    def computeVdnSizeFunction(self, R: "np.ndarray", D: "np.ndarray",
                               x: "np.ndarray", delta_v_z: "np.ndarray", sigmaR: "np.ndarray"):
        RL = R / 1.7
        RL_step = RL[1] - RL[0]
        sigma_RL = np.zeros(self.n_R)
        d_sigma_d_RL = np.zeros(self.n_R)
        p_lin_k_today = self.cosmology.power_spectrum.lin_p_mm_z_k[0, :]
        k_grid = self.cosmology.power_spectrum.k_grid
        for RL_idx, myRL in enumerate(RL):
            sigma_RL[RL_idx] = self.cosmology.computeSigmaR(myRL)
            d_sigma_d_RL[RL_idx] = general.callable_stencil_derivative(f=self.cosmology.sigmaR,
                                                                       x=myRL, step=0.5 * RL_step,
                                                                       P_lin=p_lin_k_today, k=k_grid)
        V_R = 4 * np.pi * R ** 3 / 3
        f_ln_sigma = self.computeVoidMultiplicity(D, x, delta_v_z, sigmaR)
        R_factor = (1 / (R * V_R)) * (-(RL / sigma_RL) * d_sigma_d_RL)
        Vdn_size_func_R_z = f_ln_sigma * np.expand_dims(R_factor, 1)
        return Vdn_size_func_R_z

    @staticmethod
    def voidMultiplicity(delta_v, D, sigma, x) -> "float":
        if x <= 0.276:
            result = np.sqrt(2 / np.pi) * (np.abs(delta_v) / sigma) * np.exp(-delta_v ** 2 / (2 * sigma ** 2))
        else:
            result = 0
            for j in range(1, 5):
                result += j * np.pi * x ** 2 * np.sin(j * np.pi * D) * np.exp(-(j * np.pi * x) ** 2 / 2)
            result *= 2
        return result

    def computeVoidMultiplicity(self, D: "np.ndarray", x: "np.ndarray",
                                delta_v_z: "np.ndarray", sigmaR: "np.ndarray") -> "np.ndarray":
        f_ln_sigma = np.zeros((self.n_R, len(delta_v_z)))
        for R_idx in range(f_ln_sigma.shape[0]):
            for z_idx in range(f_ln_sigma.shape[1]):
                f_ln_sigma[R_idx, z_idx] = self.voidMultiplicity(delta_v_z[z_idx],
                                                                 D[z_idx], sigmaR[R_idx],
                                                                 x[R_idx, z_idx])
        return f_ln_sigma
