import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple, Dict, Union
from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.utils import filesystem_utils as fsu
from seyfert.utils import general_utils as gu
from seyfert.cosmology import c_ells
from seyfert.cosmology.power_spectrum import PowerSpectrum
from seyfert.cosmology.cosmology import Cosmology
import logging
import datetime
from seyfert.utils.formatters import datetime_str_format

logger = logging.getLogger(__name__)


IST_STEM_DICT = {
  "6p3E-3": 0.00625,
  "1p3E-2": 0.01250,
  "1p9E-2": 0.01875,
  "2p5E-2": 0.02500,
  "3p8E-2": 0.03750,
  "5p0E-2": 0.05000,
  "1p0E-1": 0.10000
}
IST_PMM_COSMO_MAP = {
        'Ob': 'Omb',
        'Ode': 'OmDE',
        'Om': 'Omm',
        'h':  'h',
        'ns': 'ns',
        's8': 'sigma8',
        'w0': 'w0',
        'wa': 'wa'
    }
IST_STEM_EPS = np.array([0.00625, 0.0125, 0.01875, 0.025, 0.0375, 0.05, 0.1])
SCAMERA_PROBE_MAP = {
        'GCph': ("PhotometricGalaxy", "PhotometricGalaxy"),
        'GCsp': ("SpectroscopicGalaxy", "SpectroscopicGalaxy"),
        'XC': ("PhotometricGalaxy", "SpectroscopicGalaxy")
    }

IST_FISHER_NAMES_MAP = {
    "WL": "[WL]",
    "GCph": "[GCph]",
    "GCsp": "[GCsp(Pk)]",
    "GCph_WL_XC": "[WL+GCph+XC(WL,GCph)]",
    "GCsp_GCph_WL_XC": "[WL+GCph+XC(WL,GCph)] + [GCsp(Pk)]"
}


class ISTInputReader:
    pars: "PhysicalParametersCollection"

    OUR_PARAM_NAMES = list(IST_PMM_COSMO_MAP.values())

    def __init__(self, pars: "PhysicalParametersCollection" = None):
        self.ist_regex = re.compile('([a-zA-Z0-9]+)_(mn|pl)_eps_([0-9]p[0-9]E-[0-9])')
        self.pars = pars
        if self.pars is not None:
            self.pars.loadStemDisplacements(IST_STEM_EPS)

    def checkISTParams(self):
        our_params = set(IST_PMM_COSMO_MAP.values())
        assert set(self.pars.cosmological_parameters.keys()).issuperset(our_params)
        for key in self.pars:
            if key not in our_params:
                logger.warning(f'param {key} not present in IST params, removing it')
                del self.pars[key]
        assert self.pars.cosmological_parameters['h'].fiducial == 0.67

    def convertISTPowerSpectra(self, src_dir: "Path", dst_dir: "Path",
                               flat: "bool", cosmo_model: "str") -> "None":
        now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        dst_dir = dst_dir.with_name(f'{dst_dir.name}_{now}')
        dst_dir.mkdir()
        logger.info(f'Created dst dir {dst_dir}')
        ist_pmm_files = list(src_dir.glob("*/pkz*.txt"))
        ist_pmm_files.append(src_dir / 'pkz-Fiducial.txt')
        ist_pmm_files.sort()
        for f in ist_pmm_files:
            h = self.pars.cosmological_parameters['h']
            dvar, step = self.parseISTCosmoString(f.name)
            self.pars.updatePhysicalParametersForDvarStep(dvar=dvar, step=step)
            num_eps = self.pars.stem_dict[step] if step in self.pars.stem_dict else 0
            logger.info(f'converting file {f.name}: dvar {dvar}, step {step}, num_eps {num_eps}')
            logger.info(f'h = {h.current_value}')
            z, k, p_lin_zk, p_nonlin_zk = self.readISTPowerSpectrumTextFile(f)
            if dvar == 'h':
                logger.info(f'h has been varied wrt fiducial {h.fiducial}')

            k *= h.current_value
            p_lin_zk /= (h.current_value ** 3)
            p_nonlin_zk /= (h.current_value ** 3)

            z_mask = z > 0
            pmm = PowerSpectrum()
            pmm.z_grid = z[z_mask]
            pmm.k_grid = k
            pmm.lin_p_mm_z_k = p_lin_zk[z_mask, :]
            pmm.nonlin_p_mm_z_k = p_nonlin_zk[z_mask, :]
            pmm.boltzmann_code = 'CAMB'
            cosmo = Cosmology(params=self.pars.cosmological_parameters, flat=flat, model_name=cosmo_model)
            cosmo.z_grid = z[z_mask]
            cosmo.power_spectrum = pmm

            dst_dirname = fsu.get_cosmology_jobdir_name(dvar, step)
            dst_dirpath = dst_dir / dst_dirname
            dst_dirpath.mkdir()
            dst_file = dst_dirpath / 'p_mm.h5'
            cosmo.saveToHDF5(dst_file)
            self.pars.resetPhysicalParametersToFiducial()

        logger.info('Done')

    def convertSCameraDerivativesInput(self, src_dir: "Path", dst_dir: "Path") -> "None":
        now = datetime_str_format(datetime.datetime.now())
        dst_dirname = dst_dir.name
        dst_dir = dst_dir.parent / f'{dst_dirname}_converted_on_{now}'
        dst_dir.mkdir()
        files_dict = self.buildCosmoClsFilesDict(src_dir)
        correct_ij_shapes = {"GCph": (10, 10), "GCsp": (4, 4), "XC": (10, 4)}
        ell_shapes = set()
        for dvar, step in files_dict.keys():
            cl_coll = c_ells.AngularCoefficientsCollector()
            for probe_key, f in files_dict[(dvar, step)].items():
                p1, p2 = SCAMERA_PROBE_MAP[probe_key]
                symmetric = p1 == p2
                ells, cl_s = self.readSCameraClFile(f, symmetric=symmetric)
                ell_shapes.add(ells.size)
                ell_shapes.add(cl_s.shape[0])
                assert cl_s.shape[1:] == correct_ij_shapes[probe_key]

                the_cl = c_ells.AngularCoefficient(probe1=p1, probe2=p2)
                the_cl.l_bin_centers = ells
                the_cl.c_lij = cl_s
                cl_coll.cl_dict[gu.get_probes_combination_key(p1, p2)] = the_cl

            dst_dirpath = dst_dir / fsu.get_cosmology_jobdir_name(dvar, step)
            dst_dirpath.mkdir(parents=True)
            out_file = dst_dirpath / 'cl_PhotometricGalaxy_SpectroscopicGalaxy.h5'
            cl_coll.saveToHDF5(out_file)

        if len(ell_shapes) != 1:
            raise ValueError(f'Too many ell shapes: {" ".join(ell_shapes)}')

    def readSCameraClDerivatives(self, src_dir: "Union[str, Path]") -> "Dict":
        src_dir = Path(src_dir)
        filename_regex = re.compile('dCijGGd([a-zA-Z0-9]+).*-([a-zA-Z0-9]+)-Stef.dat')
        glob_pattern = 'dCijGGd*.dat'
        der_files = list(src_dir.glob(f"*/{glob_pattern}"))
        sc_dcls = {}

        for file in der_files:
            subdir_name = file.parent.name
            match = filename_regex.match(file.name)
            if match:
                dvar_first_occ, dvar_second_occ = match.groups()
                assert dvar_first_occ == dvar_second_occ
                dvar = dvar_first_occ
                del dvar_first_occ
                del dvar_second_occ
                p1, p2 = SCAMERA_PROBE_MAP[subdir_name]
                cl_key = gu.get_probes_combination_key(p1, p2)
                symmetric = p1 == p2
                logger.info(f'Loading dCl_d{dvar} for {cl_key}')
                my_dvar = IST_PMM_COSMO_MAP[dvar]
                l_s, dcl_s = self.readSCameraClFile(file, symmetric=symmetric)
                if my_dvar not in sc_dcls:
                    sc_dcls[my_dvar] = {}
                sc_dcls[my_dvar][cl_key] = dcl_s
            else:
                raise Exception(f'filename {file.name} does not match regex {filename_regex.pattern}')

        return sc_dcls

    def buildCosmoClsFilesDict(self, src_dir: "Path") -> "Dict[Path]":
        ph_files = list((src_dir / 'GCph').glob("*.dat"))
        sp_files = list((src_dir / 'GCsp').glob("*.dat"))
        xc_files = list((src_dir / 'XC').glob("*.dat"))

        def get_cosmo_files_dict(file_list: "List[Path]") -> "Dict":
            _fd = {}
            for f in file_list:
                dvar, step = self.parseISTCosmoString(f.name)
                _fd[(dvar, step)] = f
            return _fd

        ph_cosmo_files = get_cosmo_files_dict(ph_files)
        sp_cosmo_files = get_cosmo_files_dict(sp_files)
        xc_cosmo_files = get_cosmo_files_dict(xc_files)
        assert set(ph_cosmo_files) == set(sp_cosmo_files)
        assert set(ph_cosmo_files) == set(xc_cosmo_files)

        files_dict = {
            cosmo_key: {
                "GCph": ph_cosmo_files[cosmo_key],
                "GCsp": sp_cosmo_files[cosmo_key],
                "XC": xc_cosmo_files[cosmo_key]
            }
            for cosmo_key in ph_cosmo_files
        }

        return files_dict

    def parseISTCosmoString(self, s: "str") -> "Tuple":
        match = self.ist_regex.search(s)
        if match:
            dvar, sign, eps = self.ist_regex.search(s).groups()
            num_eps = IST_STEM_DICT[eps]
            if sign == 'mn':
                num_eps *= -1
            my_dvar = IST_PMM_COSMO_MAP[dvar]
            step = self.pars.inv_stem_dict[num_eps]
        else:
            logger.warning(f'string {s} does not match regex, assigning it to '
                           f'fiducial cosmology (dvar central, step 0)')
            my_dvar = 'central'
            step = 0

        return my_dvar, step

    def readSCameraClFile(self, file: "Union[str, Path]", symmetric: "bool") -> "Tuple[np.ndarray, np.ndarray]":
        file = Path(file)
        lines = [line.strip() for line in file.read_text().splitlines() if line]
        assert lines[0].startswith('#')
        hdr_lines = list(filter(lambda x: x.startswith('#'), lines))
        hdr = " ".join(hdr_lines)

        i, j, shape_ij = self.getIndicesFromSCameraClFileHeader(hdr, symmetric)

        data = np.loadtxt(file)
        ells = data[:, 0]
        cl_data = data[:, 1:]
        clij = np.zeros((len(ells), *shape_ij))
        clij[:, i, j] = cl_data
        if symmetric:
            clij[:, j, i] = cl_data
        if shape_ij[0] < shape_ij[1]:
            clij = np.transpose(clij, axes=(0, 2, 1))

        return ells, clij

    @staticmethod
    def getIndicesFromSCameraClFileHeader(hdr: "str", symmetric: "bool"):
        ij_regex = re.compile('([0-9]+).?([0-9]+)')
        ij_pairs = ij_regex.findall(hdr)
        i = np.array([int(pair[0]) for pair in ij_pairs])
        j = np.array([int(pair[1]) for pair in ij_pairs])
        i_min, i_max = i.min(), i.max()
        j_min, j_max = j.min(), j.max()
        assert i_min == j_min and i_min >= 0
        if symmetric and i_max != j_max:
            raise Exception(f'symmetric is true but i, j have different ranges {i_max} and {j_max}')
        if i_min == 0:
            shape_ij = (i_max + 1, j_max + 1)
        else:
            shape_ij = (i_max, j_max)
            i -= 1
            j -= 1
        return i, j, shape_ij

    @staticmethod
    def readISTPowerSpectrumTextFile(file: "Union[str, Path]") -> "Tuple[np.ndarray]":
        data = pd.read_table(file, delim_whitespace=True, names=['z', 'k', 'p_lin', 'p_nonlin'])
        z = np.unique(data['z'].to_numpy())
        k = np.unique(data['k'].to_numpy())
        p_lin_zk = data['p_lin'].to_numpy().reshape(len(z), len(k))
        p_nonlin_zk = data['p_nonlin'].to_numpy().reshape(len(z), len(k))
        # put z axis in increasing order...
        p_lin_zk = np.flip(p_lin_zk, axis=0)
        p_nonlin_zk = np.flip(p_nonlin_zk, axis=0)

        return z, k, p_lin_zk, p_nonlin_zk

    @staticmethod
    def readISTFisherMatrixFile(file: "Union[str, Path]") -> "Tuple[str, pd.DataFrame]":
        # Get fisher name
        regex = re.compile("EuclidISTF_([WLGCphsX_]+)")
        filename = Path(file).name
        match = regex.search(filename)
        if match:
            ist_name = match.groups()[0].strip('_')
            name = IST_FISHER_NAMES_MAP[ist_name]
        else:
            logger.warning(f"cannot retrieve fisher name from {filename}")
            name = None

        df = pd.read_table(file, delim_whitespace=True)
        name_dict = dict(zip(df.columns, df.columns))
        name_dict["Omegam"] = "Omm"
        name_dict["Omegab"] = "Omb"
        if "Omegade" in name_dict:
            name_dict["Omegade"] = "OmDE"
        bias_regex = re.compile(r"b([0-9]+)")
        for key in name_dict:
            match = bias_regex.match(key)
            if match:
                idx = int(match.groups()[0])
                name_dict[key] = f'bgph{idx - 1}'

        df.rename(name_dict, inplace=True, axis=1)
        df.index = df.columns
        df.sort_index(axis=0, inplace=True)
        df.sort_index(axis=1, inplace=True)

        return name, df
