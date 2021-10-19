from pathlib import Path
from typing import Union, Dict
import datetime
import h5py
import numpy as np
import itertools
from os.path import join
import logging

from seyfert.cosmology import weight_functions as wfs
from seyfert.cosmology import redshift_density as rds
from seyfert.config.forecast_config import ForecastConfig
from seyfert.config import main_config as mcfg
from seyfert.utils import general_utils as gu
from seyfert.cosmology.c_ells import AngularCoefficient, AngularCoefficientsCollector
from seyfert.utils.workspace import WorkSpace
from seyfert.base_structs.generic_dict import GenericDictInterface
from seyfert.file_io.hdf5_io import AbstractH5FileIO
from seyfert.fisher import fisher_utils as fu

logger = logging.getLogger(__name__)


class DeltaCl:
    probe_1: "str"
    probe_2: "str"
    c_lij: "np.ndarray"
    forecast_config: "ForecastConfig"
    fisher_config: "mcfg.FisherConfig"
    noise_lij: "np.ndarray"
    delta_c_lij: "np.ndarray"

    def __init__(self, name: "str" = None, delta_c_lij: "np.ndarray" = None, ells: "np.ndarray" = None,
                 delta_ells: "np.ndarray" = None):
        self.name = name
        self.delta_c_lij = delta_c_lij
        self.inv_delta_c_lij = None
        self.c_lij = None
        self.l_bin_centers = ells
        self.l_bin_widths = delta_ells
        self.forecast_config = None
        self.fisher_config = None
        self.noise_lij = None
        self.probe_1 = None
        self.probe_2 = None

        if self.delta_c_lij is not None:
            self.inv_delta_c_lij = np.linalg.inv(self.delta_c_lij)

    @classmethod
    def fromCl(cls, cl: "AngularCoefficient", fcfg: "ForecastConfig", fish_cfg: "mcfg.FisherConfig",
               name: "str" = None) -> "DeltaCl":
        deltacl = cls(name=name)

        deltacl.forecast_config = fcfg
        deltacl.fisher_config = fish_cfg
        deltacl.c_lij = cl.c_lij
        deltacl.l_bin_centers = cl.l_bin_centers
        deltacl.l_bin_widths = cl.l_bin_widths
        deltacl.probe_1 = cl.probe1
        deltacl.probe_2 = cl.probe2

        if deltacl.is_auto_correlation and deltacl.name != deltacl.probe_1:
            logger.warning(f"Covariance matrix from cl {cl.obs_key} should be named {deltacl.probe_1}")

        return deltacl

    @property
    def is_auto_correlation(self) -> "bool":
        return isinstance(self.probe_1, str) and self.probe_1 == self.probe_2

    @property
    def is_cross_correlation(self) -> "bool":
        return not self.is_auto_correlation and "+XC" in self.name

    def __eq__(self, other: "DeltaCl") -> "bool":
        return all([
            np.all(self.delta_c_lij == other.delta_c_lij),
            np.all(self.noise_lij == other.noise_lij),
            np.all(self.l_bin_centers == other.l_bin_centers),
            np.all(self.l_bin_widths == other.l_bin_widths)
        ])

    def computeNoise(self) -> "np.ndarray":
        if self.probe_1 != self.probe_2:
            noise_lij = np.zeros(self.c_lij.shape)
        else:
            probe = self.probe_1
            probe_cfg = self.forecast_config.probe_configs[probe]
            w = wfs.weight_function_for_probe(probe, probe_config=probe_cfg)
            w.setUp()
            w.density.evaluateSurfaceDensity()
            shot_noise_values = w.shot_noise_factor * w.density.shot_noise
            noise_ij = np.diag(shot_noise_values)
            noise_lij = np.repeat(noise_ij[np.newaxis, :, :], self.c_lij.shape[0], axis=0)
            if probe == 'SpectroscopicGalaxy' and self.forecast_config.shot_noise_sp_reduced:
                wsp = w
                wph = wfs.weight_function_for_probe("PhotometricGalaxy",
                                                    probe_config=self.forecast_config.probe_configs["PhotometricGalaxy"])
                wph.setUp()
                Nph_tot = wph.density.computeTotalGalaxyNumber()
                Nsp_tot = wsp.density.computeTotalGalaxyNumber()
                noise_lij *= (Nsp_tot/Nph_tot)

        return noise_lij

    def evaluate(self, f_sky: "float" = None) -> "None":
        if self.noise_lij is None:
            logger.info(f"computing shot noise for block {self.name}")
            self.noise_lij = self.computeNoise()
        if f_sky is None:
            f_sky = self.forecast_config.survey_f_sky

        norm_factor = np.sqrt(1 / (self.l_bin_widths * f_sky))
        norm_factor = np.expand_dims(norm_factor, axis=(1, 2))
        self.delta_c_lij = norm_factor * (self.c_lij + self.noise_lij)
        if self.is_auto_correlation:
            self.inv_delta_c_lij = np.linalg.inv(self.delta_c_lij)

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = "/") -> "None":
        h5c = H5DeltaCl(name=self.name)
        h5c.save(self, file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", root: "str" = "/") -> "None":
        h5c = H5DeltaCl(name=self.name)
        h5c.openFile(file, mode='r', root=root)
        h5c.writeToObject(self)
        h5c.closeFile()

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", name: "str", root: "str" = "/") -> "DeltaCl":
        h5c = H5DeltaCl(name=name)
        return h5c.load(file, root)


class DeltaClCollection(GenericDictInterface[str, DeltaCl]):
    forecast_config: "ForecastConfig"
    fisher_cfg: "mcfg.FisherConfig"
    single_blocks: "Dict[str, DeltaCl]"

    def __init__(self, fcfg: "ForecastConfig" = None, fisher_cfg: "mcfg.FisherConfig" = None):
        super().__init__()
        self.forecast_config = fcfg
        self.f_sky = None
        self.fisher_cfg = fisher_cfg
        self.single_blocks = {}

        if self.forecast_config is not None:
            self.f_sky = self.forecast_config.survey_f_sky

    @property
    def shot_noise_file(self) -> "Path":
        return self.forecast_config.shot_noise_file

    @property
    def ell_dict(self) -> "Dict[str, np.ndarray]":
        return {
            key: self.single_blocks[key].l_bin_centers for key in self.single_blocks
        }

    @property
    def delta_c_lij_array_dict(self) -> "Dict[str, np.ndarray]":
        return {
            key: self.single_blocks[key].delta_c_lij for key in self.single_blocks
        }

    def loadInpuDataFromWorkspace(self, ws: "WorkSpace"):
        self.loadFiducialClsFromWorkspace(ws)
        if self.shot_noise_file is not None:
            logger.info(f"Loading shot noise from file {self.shot_noise_file.name}")
            self.loadShotNoiseFromFile(self.shot_noise_file)

    def loadFiducialClsFromWorkspace(self, ws: "WorkSpace"):
        cl_coll = AngularCoefficientsCollector.fromHDF5(ws.getClFile(dvar='central', step=0))
        self.loadFiducialCls(cl_coll)

    def loadFiducialCls(self, cl_coll: "AngularCoefficientsCollector"):
        for key, cl in cl_coll.cl_dict.items():
            logger.info(f"Loading cl {key} data")
            p1, p2 = gu.get_probes_from_comb_key(obs_comb_key=key)
            name = p1 if p1 == p2 else key
            atomic_cov = DeltaCl.fromCl(cl, self.forecast_config, fish_cfg=self.fisher_cfg, name=name)
            self.single_blocks[key] = atomic_cov

    def loadShotNoiseFromFile(self, file: "Union[str, Path]"):
        logger.info("loading shot noise from file")
        hf = h5py.File(file, mode='r')
        for key in self.single_blocks:
            noise_ij = hf[key][()]
            n_ell = len(self.single_blocks[key].l_bin_centers)
            noise_lij = np.repeat(noise_ij[np.newaxis, :, :], n_ell, axis=0)

            self.single_blocks[key].noise_lij = noise_lij

        hf.close()

    def writeShotNoiseToFile(self, file: "Union[str, Path]"):
        hf = h5py.File(file, mode='w')
        hf.attrs['CreationDate'] = datetime.datetime.now().isoformat()
        for key in self.single_blocks:
            noise_ij = self.single_blocks[key].noise_lij[0]
            hf.create_dataset(name=key, data=noise_ij)

        hf.close()

    def evaluateSingleBlocks(self, f_sky: "float" = None):
        for cov_block in self.single_blocks.values():
            logger.info(f"Evaluating covariance matrix for block {cov_block.name}")
            cov_block.evaluate(f_sky=f_sky)
            if cov_block.is_auto_correlation:
                self[cov_block.name] = cov_block

    def buildXCBlocks(self):
        probes = self.forecast_config.present_probes
        l_dict = self.ell_dict
        blocks_dict = self.delta_c_lij_array_dict
        if len(probes) > 1:
            full_xc_name = fu.get_probes_xc_fisher_name(probes=probes)
            logger.info(f"Building covariance block matrix {full_xc_name}")
            xc_ells, xc_cov_lij = fu.buildXCBlockMatrixForProbes(probes=probes, matrix_dict=blocks_dict, l_dict=l_dict)
            self[full_xc_name] = DeltaCl(name=full_xc_name, delta_c_lij=xc_cov_lij, ells=xc_ells)

            if len(probes) > 2:
                for p1, p2 in itertools.combinations(probes, 2):
                    involved_probes = [p1, p2]
                    xc_name = fu.get_probes_xc_fisher_name(probes=involved_probes)
                    logger.info(f"Building covariance block matrix {xc_name}")
                    xc_ells, xc_cov_lij = fu.buildXCBlockMatrixForProbes(probes=involved_probes,
                                                                         matrix_dict=blocks_dict,
                                                                         l_dict=l_dict)
                    self[xc_name] = DeltaCl(name=xc_name, delta_c_lij=xc_cov_lij, ells=xc_ells)

    def saveToHDF5(self, file: "Union[str, Path]", root: "str" = "/") -> "None":
        h5c_coll = H5DeltaClCollection()
        h5c_coll.save(self, file, root)

    def loadFromHDF5(self, file: "Union[str, Path]", root: "str" = "/") -> "None":
        h5c_coll = H5DeltaClCollection()
        h5c_coll.openFile(file, mode='r', root=root)
        h5c_coll.writeToObject(self)
        h5c_coll.closeFile()

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", root: "str" = "/") -> "DeltaClCollection":
        obj = cls()
        obj.loadFromHDF5(file, root)
        return obj


class H5DeltaCl(AbstractH5FileIO):
    def __init__(self, name: "str" = None, **kwargs):
        super(H5DeltaCl, self).__init__(**kwargs)
        self.name = name
        self.builder_func = DeltaCl

    def readBuildingData(self) -> "None":
        self.build_data['name'] = self.attrs['name']

    def writeToObject(self, obj: "DeltaCl") -> "None":
        obj.delta_c_lij = self.root['delta_c_lij'][()]
        obj.l_bin_centers = self.root['l_bin_centers'][()]

        if 'probe_1' in self.attrs:
            obj.probe_1 = self.attrs['probe_1']
        if 'probe_2' in self.attrs:
            obj.probe_2 = self.attrs['probe_2']

        optional_attrs = ["c_lij", "noise_lij", "l_bin_widths", "inv_delta_c_lij"]
        for attr_name in optional_attrs:
            if attr_name in self.root:
                setattr(obj, attr_name, self.root[attr_name][()])
            else:
                continue

    def writeObjectToFile(self, obj: "DeltaCl") -> "None":
        self.attrs['name'] = obj.name
        if obj.probe_1 is not None:
            self.attrs['probe_1'] = obj.probe_1
        if obj.probe_2 is not None:
            self.attrs['probe_2'] = obj.probe_2

        self.createDataset(name="l_bin_centers", data=obj.l_bin_centers)
        self.createDataset(name="delta_c_lij", data=obj.delta_c_lij)

        if obj.inv_delta_c_lij is not None:
            self.createDataset(name="inv_delta_c_lij", data=obj.inv_delta_c_lij)
        if obj.l_bin_widths is not None:
            self.createDataset(name="l_bin_widths", data=obj.l_bin_widths)
        if obj.noise_lij is not None:
            self.createDataset(name="noise_lij", data=obj.noise_lij)
        if obj.c_lij is not None:
            self.createDataset(name="c_lij", data=obj.c_lij)


class H5DeltaClCollection(AbstractH5FileIO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def writeToObject(self, obj: "DeltaClCollection") -> "None":
        for name in self.root['delta_cls']:
            obj[name] = DeltaCl.fromHDF5(self.file_path, name=name, root=join(self.root_path, 'delta_cls', name))
        for single_block_name in self.root['single_blocks']:
            root_path = join(self.root_path, 'single_blocks', single_block_name)
            obj.single_blocks[single_block_name] = DeltaCl.fromHDF5(self.file_path, name=single_block_name,
                                                                    root=root_path)

    def writeObjectToFile(self, obj: "DeltaClCollection") -> "None":
        for name, delta_cl in obj.items():
            delta_cl.saveToHDF5(self.file_path, root=join(self.root_path, 'delta_cls', name))
        for single_block_name, single_block_delta_cl in obj.single_blocks.items():
            root_path = join(self.root_path, 'single_blocks', single_block_name)
            single_block_delta_cl.saveToHDF5(self.file_path, root=root_path)
