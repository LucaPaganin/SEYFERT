from pathlib import Path
import numpy as np
from seyfert.cosmology.cosmology import Cosmology
from seyfert.cosmology import weight_functions as wfs
from seyfert.utils import general_utils as gu
from seyfert.file_io.hdf5_io import AbstractH5FileIO
import logging
from typing import Union


logger = logging.getLogger(__name__)


class KernelFunction:
    cosmology: Cosmology
    probe1: str
    probe2: str
    weight1: wfs.WeightFunction
    weight2: wfs.WeightFunction
    k_ijz: np.ndarray

    def __init__(self, probe1: str = None, probe2: str = None,
                 weight1: wfs.WeightFunction = None, weight2: wfs.WeightFunction = None,
                 cosmology: Cosmology = None):
        self.probe1 = probe1
        self.probe2 = probe2
        self.weight1 = weight1
        self.weight2 = weight2
        self.cosmology = cosmology
        self.k_ijz = None
        self.z_grid = None
        self._attrs_excluded_from_equality = set()

        if self.weight1 is not None:
            if self.probe1 is not None:
                assert self.weight1.probe == self.probe1
            else:
                self.probe1 = self.weight1.probe
        if self.weight2 is not None:
            if self.probe2 is not None:
                assert self.weight2.probe == self.probe2
            else:
                self.probe2 = self.weight2.probe

    def __eq__(self, other: "KernelFunction") -> "bool":
        return gu.compare_objects(self, other, self._attrs_excluded_from_equality)

    @property
    def obs_key(self):
        return gu.get_probes_combination_key(self.probe1, self.probe2)

    @property
    def is_evaluated(self) -> bool:
        return self.k_ijz is not None

    def evaluateOverRedshiftGrid(self, z_grid: "np.ndarray", overwrite: bool = False) -> "None":
        if overwrite or not self.is_evaluated:
            self.z_grid = z_grid
            if not self.weight1.is_evaluated:
                self.weight1.evaluateOverRedshiftGrid(self.z_grid)
            if self.probe1 == self.probe2:
                self.weight2.w_bin_z = self.weight1.w_bin_z
            else:
                if not self.weight2.is_evaluated:
                    self.weight2.evaluateOverRedshiftGrid(self.z_grid)
            self.k_ijz = np.expand_dims(self.weight1.w_bin_z, 1) * np.expand_dims(self.weight2.w_bin_z, 0) * \
                         self.cosmology.c_km_s / (self.cosmology.H_z * self.cosmology.r_z**2)
        else:
            logger.info(f'kernel {self.obs_key} already evaluated')

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", root: "str" = "/") -> "KernelFunction":
        h5k = H5Kernel()
        return h5k.load(file, root)

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = "/") -> "None":
        h5k = H5Kernel()
        h5k.save(self, file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", root: "str" = "/") -> "None":
        if self.probe1 is not None:
            logger.warning(f'probe string {self.probe1} will be overwritten with file content')
        if self.probe2 is not None:
            logger.warning(f'probe string {self.probe2} will be overwritten with file content')
        h5k = H5Kernel()
        h5k.openFile(file=file, mode='r', root=root)
        h5k.writeToObject(self)
        h5k.closeFile()


class H5Kernel(AbstractH5FileIO):
    def __init__(self, **kwargs):
        super(H5Kernel, self).__init__(**kwargs)
        self.builder_func = KernelFunction

    def writeToObject(self, obj: "KernelFunction") -> "None":
        super(H5Kernel, self).writeToObject(obj)
        obj.probe1 = self.attrs['probe1']
        obj.probe2 = self.attrs['probe2']
        obj.z_grid = self.root['z_grid'][()]
        obj.k_ijz = self.root['k_ijz'][()]

    def writeObjectToFile(self, obj: "KernelFunction") -> "None":
        super(H5Kernel, self).writeObjectToFile(obj)
        self.attrs.update({'probe1': obj.probe1, 'probe2': obj.probe2})
        self.createDataset(name='z_grid', data=obj.z_grid)
        self.createDataset(name='k_ijz', data=obj.k_ijz)
