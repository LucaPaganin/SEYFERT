import numpy as np
import h5py
import logging
import seyfert.utils.filesystem_utils as fsu
from seyfert.utils import general_utils as gu
from typing import Dict, Union
from pathlib import Path
from seyfert.derivatives.differentiator import SteMClDifferentiator, AnalyticClDifferentiator
from seyfert.config.forecast_config import ForecastConfig
from seyfert.cosmology.parameter import PhysicalParametersCollection, PhysicalParameter
from seyfert.utils.workspace import WorkSpace

logger = logging.getLogger(__name__)


class ClDerivative:
    dvar: "str"
    obs_key: "str"
    forecast_config: "ForecastConfig"
    dc_lij: "np.ndarray"
    c_dvar_lij: "np.ndarray"
    l_bin_centers: "np.ndarray"

    def __init__(self, p1: "str" = None, p2: "str" = None, param: "PhysicalParameter" = None,
                 workspace: "WorkSpace" = None):
        self.probe1 = p1
        self.probe2 = p2
        self.param = param
        self.workspace = workspace
        self.l_bin_centers = None
        self.dc_lij = None
        self.c_dvar_lij = None
        self.differentiator = None

    @property
    def method(self) -> "str":
        return self.param.derivative_method

    @property
    def is_cross_correlation(self) -> "bool":
        return self.probe1 != self.probe2

    @property
    def probe_key(self) -> "str":
        return gu.get_probes_combination_key(self.probe1, self.probe2)

    def evaluate(self) -> "None":
        if self.method == 'SteM':
            self.differentiator = SteMClDifferentiator(probe1=self.probe1, probe2=self.probe2,
                                                       param=self.param, workspace=self.workspace,
                                                       vectorized=True)
        elif self.method == 'Analytic':
            self.differentiator = AnalyticClDifferentiator(probe1=self.probe1, probe2=self.probe2,
                                                           param=self.param, workspace=self.workspace)
        else:
            raise NotImplementedError(f'derivative method {self.method} is not implemented')

        self.differentiator.loadClData()
        self.l_bin_centers = self.differentiator.fiducial_cl.l_bin_centers
        self.dc_lij = self.differentiator.computeDerivative()


class ClDerivativeCollector:
    dvar: "str"
    dcl_dict: "Dict[str, ClDerivative]"
    phys_pars: "PhysicalParametersCollection"
    forecast_config: "ForecastConfig"
    param: "PhysicalParameter"

    def __init__(self, dvar: "str" = None, workspace: "WorkSpace" = None):
        self.dvar = dvar
        self.workspace = workspace
        self.dcl_dict = {}
        self.forecast_config = None
        self.phys_pars = None
        self.param = None
        if self.workspace is not None:
            self.forecast_config = self.workspace.getForecastConfiguration()
            self.phys_pars = self.workspace.getParamsCollection()
        if self.phys_pars is not None and self.dvar is not None:
            self.param = self.phys_pars[self.dvar]

    @property
    def ell_dict(self) -> "Dict[str, np.ndarray]":
        return {key: dcl.l_bin_centers for key, dcl in self.dcl_dict.items()}

    @property
    def dc_lij_array_dict(self) -> "Dict[str, np.ndarray]":
        return {key: dcl.dc_lij for key, dcl in self.dcl_dict.items()}

    def __getitem__(self, item: "str") -> "ClDerivative":
        return self.dcl_dict[item]

    def evaluateDerivatives(self) -> "None":
        logger.info(f"Evaluating derivatives w.r.t. {self.param.name}, method {self.param.derivative_method}")
        for key in self.dcl_dict:
            self.dcl_dict[key].evaluate()

    def setUp(self):
        for p1, p2 in self.forecast_config.probes_combinations:
            obs_key = gu.get_probes_combination_key(p1, p2)
            clder = ClDerivative(p1=p1, p2=p2, param=self.param, workspace=self.workspace)
            self.dcl_dict[obs_key] = clder

    def writeToFile(self, file: "Union[str, Path]") -> "None":
        hf = h5py.File(file, "w")
        for obs_key in self.dcl_dict:
            grp = hf.create_group(obs_key)
            grp.create_dataset("dc_lij", data=self.dcl_dict[obs_key].dc_lij, dtype='f8',
                               compression='gzip', compression_opts=9)
            grp.create_dataset("l_bin_centers", data=self.dcl_dict[obs_key].l_bin_centers, dtype='f8',
                               compression='gzip', compression_opts=9)
        hf.close()

    def loadFromFile(self, file: "Union[str, Path]") -> "None":
        self.dcl_dict = {}
        hf = h5py.File(file, 'r')
        for obs_key in hf.keys():
            p1, p2 = gu.get_probes_from_comb_key(obs_key)
            dc_lij = hf[obs_key]["dc_lij"][()]
            ells = hf[obs_key]["l_bin_centers"][()]
            clder = ClDerivative(p1=p1, p2=p2)
            clder.l_bin_centers = ells
            clder.dc_lij = dc_lij
            self.dcl_dict[obs_key] = clder

        hf.close()

    @classmethod
    def fromHDF5(cls, file: "Union[str, Path]", dvar: "str" = None) -> "ClDerivativeCollector":
        dcoll = cls(dvar=dvar)
        dcoll.loadFromFile(file)

        return dcoll


def collect_cl_derivatives(der_dir: "Path") -> "Dict[str, Dict[str, np.ndarray]]":
    dcl_dict = {}
    for dvar_dir in der_dir.iterdir():
        dvar = fsu.get_dvar_from_derivative_jobdir_name(dvar_dir.name)
        der_file = fsu.get_file_from_dir_with_pattern(dvar_dir, "cl_derivatives*h5")
        dcl_coll = ClDerivativeCollector(dvar=dvar)
        dcl_coll.loadFromFile(der_file)
        dcl_dict[dvar] = {
            key: dcl_coll[key].dc_lij for key in dcl_coll.dcl_dict
        }

    return dcl_dict
