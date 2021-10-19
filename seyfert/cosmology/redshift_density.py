from typing import Union, Tuple, Dict
from pathlib import Path
from scipy import interpolate
from scipy import integrate
import numpy as np
from numba import jit
from numba.core.errors import TypingError
import logging
import h5py

from seyfert.file_io.hdf5_io import AbstractH5FileIO
from seyfert.utils import general_utils as gu
from seyfert.utils.type_helpers import TPathLike


class DensityError(Exception):
    pass


logger = logging.getLogger(__name__)


class RedshiftDensity:
    probe: "str"
    dN_dz_dOmega_spline: "interpolate.InterpolatedUnivariateSpline"
    bin_norm_factors: "np.ndarray"
    z_grid: "np.ndarray"
    input_file: "Path"

    def __init__(self, probe: "str" = None):
        self.probe = probe
        self.z_grid = None
        self.norm_density_iz = None
        self.input_z_domain = None
        self.bin_norm_factors = None
        self.dN_dOmega_bins = None
        self.z_bin_edges = None
        self.instrument_response = {}
        self.good_fraction_params = {}
        self.catastrophic_fraction_params = {}
        self.has_niz_from_input = None
        self.input_dN_dz_dOmega = None
        self.dN_dz_dOmega_spline = None
        self.n_iz_splines = None
        self.input_n_iz = None
        self.input_file = None
        self.catalog_f_sky = None
        self.ready_to_evaluate = False
        self._attrs_excluded_from_equality = {'dN_dz_dOmega_spline', 'input_file', 'ready_to_evaluate',
                                              'bin_norm_factors', 'n_iz_splines',
                                              'good_fraction_params', 'catastrophic_fraction_params'}

    @staticmethod
    def fromHDF5(file: "Union[str, Path]", root: "str" = '/') -> "RedshiftDensity":
        h5d = H5Density()
        return h5d.load(file, root)

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = '/') -> None:
        h5d = H5Density()
        h5d.save(self, file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", root: "str" = '/') -> None:
        h5d = H5Density()
        h5d.openFile(file=file, mode='r', root=root)
        h5d.writeToObject(self)
        h5d.closeFile()

    @property
    def z_min(self) -> "float":
        return self.input_z_domain[0]

    @property
    def z_max(self) -> "float":
        return self.input_z_domain[-1]

    @property
    def z_bin_centers(self) -> "np.ndarray":
        return (self.z_bin_edges[:-1] + self.z_bin_edges[1:]) / 2

    @property
    def n_bins(self) -> "int":
        return len(self.z_bin_centers)

    @property
    def shot_noise(self) -> "np.ndarray":
        return 1. / self.dN_dOmega_bins

    def setUp(self) -> "None":
        self.interpolateInput()
        if not self.has_niz_from_input:
            self.good_fraction_params = {
                'amplitude': 1 - self.instrument_response['f_out'],
                'z_mean': self.instrument_response['z_b'],
                'sigma': self.instrument_response['sigma_b'],
                'c': self.instrument_response['c_b'],
            }
            self.catastrophic_fraction_params = {
                'amplitude': self.instrument_response['f_out'],
                'z_mean': self.instrument_response['z_o'],
                'sigma': self.instrument_response['sigma_o'],
                'c': self.instrument_response['c_o'],
            }
            self.evaluateBinNormFactors()

        self.ready_to_evaluate = True

    def evaluateSurfaceDensity(self):
        self.dN_dOmega_bins = np.array([self.computeSurfaceDensityAtBin(i) for i in range(self.n_bins)])

    def evaluate(self, z_grid: np.ndarray) -> "None":
        if not self.ready_to_evaluate:
            raise DensityError("density must be setup before evaluation")
        self.z_grid = z_grid
        self.norm_density_iz = np.zeros((self.n_bins, len(self.z_grid)))
        for i in range(self.norm_density_iz.shape[0]):
            self.norm_density_iz[i] = self.computeNormalizedDensityAtBinAndRedshift(i=i, z=self.z_grid)

    def interpolateInput(self) -> "None":
        if self.has_niz_from_input:
            self.n_iz_splines = [interpolate.InterpolatedUnivariateSpline(x=self.input_z_domain,
                                                                          y=self.input_n_iz[i, :],
                                                                          ext='zeros', k=3)
                                 for i in range(self.input_n_iz.shape[0])]
        else:
            self.dN_dz_dOmega_spline = interpolate.InterpolatedUnivariateSpline(x=self.input_z_domain,
                                                                                y=self.input_dN_dz_dOmega,
                                                                                ext='zeros', k=3)

    def computeTotalGalaxyNumber(self, integ_method="simps") -> "float":
        if not self.ready_to_evaluate:
            raise DensityError("density must be setup before evaluation")

        if integ_method == "simps":
            dNg_dOmega = integrate.simps(x=self.input_z_domain, y=self.input_dN_dz_dOmega)
        elif integ_method == "quad":
            dNg_dOmega, _ = integrate.quad(self.dN_dz_dOmega_spline, self.z_min, self.z_max)
        else:
            raise KeyError(f"Unrecognized method {integ_method}")

        return dNg_dOmega * 4*np.pi*self.catalog_f_sky

    @staticmethod
    @jit(nopython=True)
    def modifiedGaussianResponse(z_p: "Union[float, np.ndarray]", z: "Union[float, np.ndarray]",
                                 z_mean: "float" = None, sigma: "float" = None,
                                 c: "float" = None, amplitude: "float" = None) -> "Union[float, np.ndarray]":
        return (amplitude / (np.sqrt(2 * np.pi) * sigma * (1 + z))) * \
               np.exp(-((z - c * z_p - z_mean) ** 2 / (2 * (sigma * (1 + z)) ** 2)))

    def computeInstrumentResponse(self,
                                  z_p: "Union[float, np.ndarray]",
                                  z: "Union[np.ndarray]") -> "Union[float, np.ndarray]":
        try:
            return self.modifiedGaussianResponse(z_p, z, **self.good_fraction_params) + \
                   self.modifiedGaussianResponse(z_p, z, **self.catastrophic_fraction_params)
        except TypingError as e:
            logger.error(f'Caught exception {type(e)} {e}')
            raise DensityError('Cannot compute instrument response for already convolved density')

    def convolvedNdzdOmegaWithInstrumentResponse(self,
                                                 z: "Union[float, np.ndarray]", i: "int") -> "Union[float, np.ndarray]":
        if self.has_niz_from_input:
            raise DensityError(f'Cannot convolve already convolved density')
        if isinstance(z, float):
            result, _ = integrate.quad(self.computeInstrumentResponse,
                                       self.z_bin_edges[i], self.z_bin_edges[i + 1], args=z)
        elif isinstance(z, np.ndarray):
            result = np.vectorize(lambda x: integrate.quad(self.computeInstrumentResponse,
                                                           self.z_bin_edges[i], self.z_bin_edges[i + 1], args=x)[0])(z)
        else:
            raise TypeError(f'Expected float or np.ndarray as z, got {type(z)}')

        return result * self.dN_dz_dOmega_spline(z)

    def computeBinNormFactor(self, bin_idx: int) -> float:
        result, _ = integrate.quad(self.convolvedNdzdOmegaWithInstrumentResponse, self.z_min, self.z_max, args=bin_idx)
        return result

    def evaluateBinNormFactors(self) -> None:
        self.bin_norm_factors = np.array([self.computeBinNormFactor(i) for i in range(self.n_bins)])

    def computeNormalizedDensityAtBinAndRedshift(self, i: "int",
                                                 z: "Union[float, np.ndarray]") -> "Union[float, np.ndarray]":
        if not self.has_niz_from_input:
            norm_density = self.convolvedNdzdOmegaWithInstrumentResponse(z, i)
            try:
                norm_density /= self.bin_norm_factors[i]
            except FloatingPointError:
                raise ZeroDivisionError(f"Denominator {i} is zero")
        else:
            norm_density = self.n_iz_splines[i](z)
        return norm_density

    def computeSurfaceDensityAtBin(self, i: "int") -> "float":
        if self.has_niz_from_input:
            result = self.dN_dOmega_bins[i]
        else:
            result, _ = integrate.quad(self.dN_dz_dOmega_spline, self.z_bin_edges[i], self.z_bin_edges[i + 1])
        return result

    def __eq__(self, other: "RedshiftDensity") -> "bool":
        return gu.compare_objects(self, other, self._attrs_excluded_from_equality)


class H5Density(AbstractH5FileIO):
    def __init__(self, **kwargs):
        super(H5Density, self).__init__(**kwargs)
        self.builder_func = RedshiftDensity

    def writeToObject(self, obj: "RedshiftDensity") -> "None":
        super(H5Density, self).writeToObject(obj)
        obj.probe = self.attrs['probe']
        self.populateObject(obj, base_grp_path=self.root_path)
        if not obj.has_niz_from_input:
            obj.instrument_response = {
                key: dset[()] for key, dset in self.root['instrument_response'].items()
            }

    def writeObjectToFile(self, obj: "RedshiftDensity") -> "None":
        self.attrs['probe'] = obj.probe
        self.createDataset(name='has_niz_from_input', data=obj.has_niz_from_input)
        self.createDataset(name='input_z_domain', data=obj.input_z_domain)
        self.createDataset(name='z_bin_edges', data=obj.z_bin_edges)
        self.createDataset(name='catalog_f_sky', data=obj.catalog_f_sky)
        if obj.dN_dOmega_bins is not None:
            self.createDataset(name='dN_dOmega_bins', data=obj.dN_dOmega_bins)
        if obj.has_niz_from_input:
            self.createDataset(name='input_n_iz', data=obj.input_n_iz)
        else:
            self.createDataset(name='input_dN_dz_dOmega', data=obj.input_dN_dz_dOmega)
            instr_resp_grp = self.createGroup(name='instrument_response')
            for key, value in obj.instrument_response.items():
                self.createDataset(name=key, data=value, base_grp=instr_resp_grp)
        if obj.z_grid is not None:
            self.createDataset(name='z_grid', data=obj.z_grid)
        if obj.norm_density_iz is not None:
            self.createDataset(name='norm_density_iz', data=obj.norm_density_iz)


def save_densities_to_file(densities: "Dict[str, RedshiftDensity]", file: "TPathLike"):
    for name in densities:
        densities[name].saveToHDF5(file=file, root=name)


def load_densities_from_file(file: "TPathLike") -> "Dict[str, RedshiftDensity]":
    hf = h5py.File(file, mode='r')
    names = list(hf.keys())
    hf.close()
    densities = {}
    for name in names:
        densities[name] = RedshiftDensity.fromHDF5(file=file, root=name)

    return densities


def get_bins_overlap(z1_l: "float", z1_r: "float", z2_l: "float", z2_r: "float") -> "Tuple[float, float]":
    if z1_r <= z1_l:
        raise Exception("Bin 1 extrema are in wrong order")
    if z2_r <= z2_l:
        raise Exception("Bin 2 extrema are in wrong order")
    if z2_l >= z1_r or z1_l >= z2_r:
        overlap = None
    else:
        z_l = max(z2_l, z1_l)
        z_r = min(z2_r, z1_r)
        overlap = (z_l, z_r)

    return overlap


def compute_photoXspectro_shotnoise_ij(nph: "RedshiftDensity", nsp: "RedshiftDensity") -> "np.ndarray":
    assert nph.probe == "PhotometricGalaxy"
    assert nsp.probe == "SpectroscopicGalaxy"
    noise_phsp_ij = np.zeros((nph.n_bins, nsp.n_bins))
    nsp.interpolateInput()
    for i in range(nph.n_bins):
        for j in range(nsp.n_bins):
            overlap = get_bins_overlap(nph.z_bin_edges[i], nph.z_bin_edges[i + 1],
                                       nsp.z_bin_edges[j], nsp.z_bin_edges[j + 1])
            if overlap is not None:
                dNsp_dOmega_overlap = integrate.quad(nsp.dN_dz_dOmega_spline, a=overlap[0], b=overlap[1])[0]
                noise_phsp_ij[i, j] = dNsp_dOmega_overlap / (nph.dN_dOmega_bins[i] * nsp.dN_dOmega_bins[j])

    return noise_phsp_ij
