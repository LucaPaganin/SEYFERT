from pathlib import Path
import numpy as np
import json
from abc import ABC
import logging
import sys
from typing import Dict, Union

from seyfert.utils import filesystem_utils as fsu
from seyfert.base_structs.generic_dict import GenericDictInterface
from seyfert.utils import general_utils as gu

logger = logging.getLogger(__name__)


class MainConfig(GenericDictInterface, ABC):
    def __init__(self, json_input: "Union[str, Path]" = None, **kwargs):
        super().__init__(**kwargs)
        self.type = self.__class__.__name__
        self["config_type"] = self.type
        self._attrs_excluded_from_equality = set()
        if json_input is not None:
            self.loadFromJSON(json_input)

    def __repr__(self) -> "str":
        return repr(self.base_dict)

    @classmethod
    def fromJSON(cls, file: "Union[str, Path]") -> "MainConfig":
        return cls(json_input=file)

    def loadFromJSON(self, json_file: "Union[str, Path]") -> "None":
        with open(json_file, mode='r') as jsf:
            self.update(json.load(jsf))

    def writeToJSON(self, json_file: "Union[str, Path]") -> "None":
        with open(json_file, mode='w') as jsf:
            json.dump(self._base_dict, jsf, indent=2)


class PowerSpectrumConfig(MainConfig):
    def __init__(self, **kwargs):
        super(PowerSpectrumConfig, self).__init__(**kwargs)

    @property
    def z_min(self) -> "float":
        return self["redshift_grid"]["z_min"]

    @z_min.setter
    def z_min(self, value):
        self["redshift_grid"]["z_min"] = value

    @property
    def z_max(self):
        return self["redshift_grid"]["z_max"]

    @z_max.setter
    def z_max(self, value):
        self["redshift_grid"]["z_max"] = value

    @property
    def n_redshifts(self) -> "int":
        return self["redshift_grid"]["n_redshifts"]

    @n_redshifts.setter
    def n_redshifts(self, value):
        self["redshift_grid"]["n_redshifts"] = value

    @property
    def z_grid(self):
        return np.linspace(self.z_min, self.z_max, self.n_redshifts)

    @property
    def boltzmann_codes_settings(self):
        return self["boltzmann_codes_settings"]

    @property
    def boltzmann_code(self):
        return self.boltzmann_codes_settings["code"]

    @boltzmann_code.setter
    def boltzmann_code(self, value):
        self.boltzmann_codes_settings["code"] = value

    @property
    def camb_settings(self) -> "Dict[str, str]":
        return self.boltzmann_codes_settings["camb"]

    @property
    def class_settings(self) -> "Dict[str, str]":
        return self.boltzmann_codes_settings["class"]

    @property
    def camb_ini_name(self) -> "str":
        return self.camb_settings['ini_filename']

    @camb_ini_name.setter
    def camb_ini_name(self, value):
        self.camb_settings['ini_filename'] = value

    @property
    def camb_ini_path(self) -> "Path":
        return fsu.config_files_dir() / self.camb_ini_name

    def loadFromJSON(self, json_file: "Union[str, Path]") -> "None":
        super(PowerSpectrumConfig, self).loadFromJSON(json_file)


class AngularConfig(MainConfig):
    def __init__(self, **kwargs):
        super(AngularConfig, self).__init__(**kwargs)

    @property
    def cl_integration_settings(self) -> "Dict":
        return self["cl_integration_settings"]

    @property
    def limber_pmm_spline_settings(self) -> "Dict":
        return self["limber_pmm_spline_settings"]

    @property
    def lensing_efficiency_integration_method(self) -> "str":
        return self.cl_integration_settings["lensing_efficiency"]

    @property
    def cl_integration_method(self) -> "str":
        return self.cl_integration_settings["cl"]

    @property
    def limber_spline_kx(self):
        return int(self.limber_pmm_spline_settings["kx"])

    @property
    def limber_spline_ky(self):
        return int(self.limber_pmm_spline_settings["ky"])

    @property
    def limber_spline_s(self):
        return int(self.limber_pmm_spline_settings["s"])


class DerivativeConfig(MainConfig):
    def __init__(self, **kwargs):
        super(DerivativeConfig, self).__init__(**kwargs)

    @property
    def cl_dir(self):
        return self['angular_dir']

    @cl_dir.setter
    def cl_dir(self, value):
        self['angular_dir'] = value


class FisherConfig(MainConfig):
    def __init__(self, **kwargs):
        super(FisherConfig, self).__init__(**kwargs)

    @property
    def shot_noise_phsp_xc_factor(self) -> "float":
        return self['shot_noise_phsp_xc_factor']

    @property
    def fisher_formula(self) -> "str":
        return self['fisher_formula']

    @property
    def cl_derivatives(self) -> "Dict":
        return self['cl_derivatives']

    def getXCDerivativeFlag(self, key: "str") -> "bool":
        try:
            flag = self.cl_derivatives[key]
        except KeyError:
            p1, p2 = gu.get_probes_from_comb_key(key)
            reversed_key = gu.get_probes_combination_key(p2, p1)
            try:
                flag = self.cl_derivatives[reversed_key]
            except KeyError:
                raise KeyError(f"neither {key} or {reversed_key} found into cl derivatives presence flags")

        return flag


ConcreteConfigType = Union[PowerSpectrumConfig, AngularConfig, DerivativeConfig, FisherConfig]


def config_for_task(task: "str", json_input: "Union[str, Path]") -> "ConcreteConfigType":
    acceptable_task_names = {"PowerSpectrum", "Angular", "Derivative", "Fisher"}
    if task not in acceptable_task_names:
        raise ValueError(f'Invalid task name {task}, should be one of {" ".join(acceptable_task_names)}')
    cfg_name = f'{task}Config'
    cfg = getattr(sys.modules[__name__], cfg_name)(json_input=json_input)
    return cfg