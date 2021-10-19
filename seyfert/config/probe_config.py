from typing import TYPE_CHECKING, Tuple
import numpy as np
import logging
from pathlib import Path
from typing import Dict
from seyfert.cosmology.bias import Bias
from seyfert.cosmology.parameter import PhysicalParameter
from seyfert.utils import array_utils
from seyfert.utils.shortcuts import ProbeName

if TYPE_CHECKING:
    from seyfert.file_io.json_io import JSONForecastConfig

logger = logging.getLogger(__name__)


class ProbeConfig:
    name: "ProbeName"
    input_data_dir: "Path"
    l_bin_centers: "np.ndarray"
    l_bin_widths: "np.ndarray"
    density_init_file: "Path"
    bias_init_file: "Path"
    bias_diff_method: "str"
    nuisance_parameters: "Dict[str, PhysicalParameter]"

    def __init__(self, name: "str", input_data_dir: "Path" = None, json_config: "JSONForecastConfig" = None):
        self.name = ProbeName(name)
        self.input_data_dir = input_data_dir
        self.density_init_file = None
        self.bias_init_file = None
        self.bias_diff_method = None
        self.l_bin_centers = None
        self.l_bin_widths = None
        self.nuisance_parameters = {}
        self.marginalize_bias = False
        self.specific_settings = {}
        self._attrs_excluded_from_equality = {'density_init_file', 'bias_init_file'}
        if json_config is not None:
            self.loadFromJSONConfig(json_config)

    @property
    def long_name(self) -> "str":
        return self.name.toLong().name

    @property
    def alias_name(self):
        return self.name.toAlias().name

    @property
    def present_nuisance_parameters(self) -> "Dict[str, PhysicalParameter]":
        return {name: par for name, par in self.nuisance_parameters.items() if par.is_free_parameter}

    def getProbeInitFile(self, filename: "str"):
        input_dirs = [
            self.input_data_dir / self.alias_name,
            self.input_data_dir / self.long_name,
            self.input_data_dir
        ]
        input_file = None
        for input_dir in input_dirs:
            input_file = input_dir / filename
            if input_file.is_file():
                break

        if input_file is None:
            raise FileNotFoundError(f"file {filename} not found in any of {' '.join([str(d) for d in input_dirs])}")

        return input_file

    def loadFromJSONConfig(self, json_config: "JSONForecastConfig", load_nuis_params=False):
        probe_dict = json_config.probes[self.name.toAlias().name]

        self.l_bin_centers, self.l_bin_widths = self.getProbeMultipoleBinning(probe_dict)
        self.density_init_file = self.getProbeInitFile(probe_dict["density_file"])
        if self.long_name != "Lensing":
            self.bias_init_file = self.getProbeInitFile(probe_dict["bias_file"])
            self.marginalize_bias = probe_dict["marginalize_bias_flag"]
            self.bias_diff_method = probe_dict["bias_derivative_method"]

        self.specific_settings = probe_dict["specific_settings"]

        if load_nuis_params:
            self.loadAllNuisanceParameters(probe_dict)

    def loadAllNuisanceParameters(self, probe_dict: "Dict"):
        if self.bias_init_file is not None and self.marginalize_bias is not None:
            # The bias has to be ALWAYS loaded, in order to know its parameters
            bias = Bias.fromHDF5(self.bias_init_file)
            if bias.nuisance_parameters is not None:
                for key, value in bias.nuisance_parameters.items():
                    nuis_param = PhysicalParameter(name=key, fiducial=value, kind=PhysicalParameter.NUISANCE_PAR_STRING,
                                                   probe=self.long_name, is_free_parameter=self.marginalize_bias)
                    if self.bias_diff_method is not None:
                        nuis_param.derivative_method = self.bias_diff_method
                    self.nuisance_parameters[key] = nuis_param

        if "extra_nuisance_parameters" in probe_dict:
            extra_nuisance_pars = [
                PhysicalParameter.from_dict(par_dict) for par_dict in probe_dict["extra_nuisance_parameters"]
            ]
            self.nuisance_parameters.update({
                par.name: par for par in extra_nuisance_pars
            })

    def getProbeMultipoleBinning(self, probe_dict: "Dict") -> "Tuple[np.ndarray, np.ndarray]":
        ell_filename = probe_dict["ell_external_filename"] if "ell_external_filename" in probe_dict else None
        if ell_filename is not None:
            ells_file = self.input_data_dir / ell_filename
            input_ells = np.loadtxt(ells_file)
            l_bin_centers, l_bin_widths = input_ells[:, 0], input_ells[:, 1]
        else:
            l_min, l_max = probe_dict['l_min'], probe_dict['l_max']
            spacing = "Logarithmic" if probe_dict["ell_log_selection"] else "Linear"
            n_ell = probe_dict["log_l_number"] if "log_l_number" in probe_dict else None

            l_bin_centers, l_bin_widths = array_utils.compute_multipoles_arrays(l_min, l_max, spacing=spacing,
                                                                                n_ell=n_ell)

        return l_bin_centers, l_bin_widths

    def __eq__(self, other: "ProbeConfig") -> "bool":
        conds = [
            self.name == other.name,
            np.all(self.l_bin_centers == other.l_bin_centers),
            np.all(self.l_bin_widths == other.l_bin_widths),
            self.density_init_file.name == other.density_init_file.name,
            self.marginalize_bias == other.marginalize_bias,
            self.nuisance_parameters == other.nuisance_parameters,
            self.specific_settings == other.specific_settings
        ]
        if self.bias_init_file is not None and other.bias_init_file is not None:
            conds.append(self.bias_init_file.name == other.bias_init_file.name)
        if self.marginalize_bias is not None and other.marginalize_bias is not None:
            conds.append(self.marginalize_bias == other.marginalize_bias)
        return all(conds)
