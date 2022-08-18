from typing import Dict, Tuple, List, Union, Any, Iterable
import json
import numpy as np
import itertools
import logging
from pathlib import Path
import datetime

import seyfert.utils.array_utils
from seyfert.cosmology.parameter import PhysicalParameter
from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.utils import general_utils as gu
from seyfert.utils import filesystem_utils as fsu
from seyfert.file_io.json_io import JSONForecastConfig
from seyfert.config.probe_config import ProbeConfig
from seyfert.utils.shortcuts import ProbeName
from seyfert.utils import formatters as fm
from seyfert.utils.type_helpers import TPathLike


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    pass


class ForecastConfig:
    input_data_dir: "Path"
    probe_configs: "Dict[str, ProbeConfig]"
    phys_pars: "PhysicalParametersCollection"
    survey_f_sky: "float"
    json_config: "JSONForecastConfig"

    def __init__(self, input_file: "TPathLike" = None, 
                 input_data_dir: "TPathLike" = None, 
                 load_phys_params: "bool" = False):
        self.input_file = input_file
        self.input_data_dir = input_data_dir
        self.probe_configs = {}
        self.shot_noise_file = None
        self.probes_combinations = None
        self.phys_pars = None
        self._attrs_excluded_from_equality = {}
        self.json_config = None
        self.verbose = False

        if self.input_file is not None and self.input_data_dir is not None:
            self.loadConfiguration(input_file=self.input_file, 
                                   input_data_dir=self.input_data_dir, 
                                   load_phys_params=load_phys_params)
        else:
            logger.warning("Missing one of input_data_dir and input_file, forecast configuration not loaded")

    @property
    def scenario(self) -> "str":
        return self.json_config.scenario

    @property
    def n_sp_bins(self) -> "int":
        return self.json_config.n_sp_bins

    @property
    def shot_noise_sp_reduced(self) -> "bool":
        return self.synthetic_opts["shot_noise_sp_reduced"]

    @property
    def synthetic_opts(self) -> "Dict":
        return self.json_config.synthetic_opts

    @property
    def survey_f_sky(self):
        return self.json_config.f_sky

    @property
    def is_universe_flat(self) -> "bool":
        return self.json_config.cosmology["flat"]

    @property
    def present_probes(self) -> "List[str]":
        return list(self.probe_configs.keys())

    @property
    def nuisance_parameters(self) -> "Dict[str, PhysicalParameter]":
        return self.phys_pars.nuisance_parameters

    @property
    def present_cosmological_parameters(self) -> "Dict[str, PhysicalParameter]":
        return self.phys_pars.free_cosmological_parameters

    @property
    def cosmo_pars_fiducials(self) -> "Dict[str, float]":
        return self.phys_pars.cosmo_pars_fiducials

    def loadConfiguration(self, input_file: "TPathLike", input_data_dir: "TPathLike",
                          load_phys_params: "bool" = False) -> "None":
        logger.info("Loading forecast configuration")
        self.json_config = JSONForecastConfig(file=input_file)
        self.input_data_dir = Path(input_data_dir)
        self.loadProbeConfigurationsFromJSONConfig()
        if load_phys_params:
            self.loadPhysicalParametersFromJSONConfig()

    def writeToJSON(self, file: "TPathLike"):
        self.json_config.toJSON(file)

    def getConfigID(self) -> "str":
        return self.json_config.getConfigID()

    def retrieveInputDataDir(self, rundir: "Path"):
        self.input_data_dir = fsu.get_forecast_input_data_dir_from_rundir(rundir)

    def loadProbeConfigurationsFromJSONConfig(self):
        for name, probe_dict in self.json_config.probes.items():
            if probe_dict['presence_flag']:
                long_name = ProbeName(name).toLong().name
                probe_cfg = ProbeConfig(long_name, input_data_dir=self.input_data_dir)
                probe_cfg.loadFromJSONConfig(self.json_config)
                self.probe_configs[long_name] = probe_cfg

        self.probes_combinations = list(itertools.combinations_with_replacement(self.present_probes, 2))

        if "shot_noise_file" in self.json_config.survey:
            shot_noise_file = self.json_config.survey["shot_noise_file"]
            if shot_noise_file is not None:
                self.shot_noise_file = self.input_data_dir / "shot_noise" / shot_noise_file
                if not self.shot_noise_file.is_file():
                    raise FileNotFoundError(self.shot_noise_file)

    def loadPhysicalParametersFromJSONConfig(self):
        self.phys_pars = PhysicalParametersCollection()
        self.phys_pars.loadStemDisplacements(self.json_config.derivative_settings["base_stem_disps"])
        self.phys_pars.is_universe_flat = self.is_universe_flat
        self.phys_pars.update({
            par.name: par for par in [
                PhysicalParameter.from_dict(par_dict) for par_dict in self.json_config.cosmology["parameters"]
            ]
        })
        for pcfg in self.probe_configs.values():
            pcfg.loadAllNuisanceParameters(self.json_config.probes[pcfg.alias_name])
            self.phys_pars.update(pcfg.nuisance_parameters)

    def checkProbesMultipoles(self) -> None:
        for probe_1, probe_2 in self.probes_combinations:
            l_probe_1 = self.probe_configs[probe_1].l_bin_centers
            l_probe_2 = self.probe_configs[probe_2].l_bin_centers
            if len(seyfert.utils.array_utils.compute_arrays_intersection1d([l_probe_1, l_probe_2])) == 0:
                raise ValueError(f'Probes {probe_1} and {probe_2} have incompatible multipole values, '
                                 f'probably one has log scale while the other does not')

    def getMultipoleArraysForProbeComb(self, obs_key: "str") -> "Tuple[np.ndarray, np.ndarray]":
        obs1, obs2 = gu.get_probes_from_comb_key(obs_key)
        ell1, ell2 = self.probe_configs[obs1].l_bin_centers, self.probe_configs[obs2].l_bin_centers
        ell_common = np.intersect1d(ell1, ell2)
        mask_common = np.isin(ell1, ell_common)
        d_ell1, d_ell2 = self.probe_configs[obs1].l_bin_widths, self.probe_configs[obs2].l_bin_widths
        d_ell_common = d_ell1[mask_common]

        if len(ell_common) == 0:
            raise Exception(f"No multipoles in common for probe combination {obs_key}")

        return ell_common, d_ell_common

    def __getitem__(self, item: "str") -> "ProbeConfig":
        return self.probe_configs[item]

    def __eq__(self, other: "ForecastConfig") -> "bool":
        return all([
            self.phys_pars == other.phys_pars,
            self.is_universe_flat == other.is_universe_flat,
            self.probe_configs == other.probe_configs,
            self.survey_f_sky, other.survey_f_sky,
            self.probes_combinations == other.probes_combinations
        ])


class OptionValueError(Exception):
    def __init__(self, opt_name: "str", got: "Any", acceptable: "Iterable[Any]"):
        self.opt_name = opt_name
        self.got = got
        self.add_info_str = f"got {got}, accepted: " + ", ".join(str(x) for x in sorted(acceptable))
        super().__init__(f"Invalid value for option {self.opt_name}. {self.add_info_str}")


class ForecastConfigEditor:
    opts: "Dict[str, Any]"
    accepted_opt_names = {
            "scenario",
            "n_sp_bins",
            "gcph_minus_gcsp",
            "shot_noise_sp_reduced",
            "gcph_only_bins_in_spectro_range"
        }

    def __init__(self, opts: "Dict[str, Any]"):
        self.opts = opts
        self.creation_date = datetime.datetime.now()
        self.config = None

        with open(fsu.config_files_dir() / "basic_forecast_config.json") as jsf:
            self.config = JSONForecastConfig(data=json.load(jsf))
        self.checkOpts()

        self.config.synthetic_opts.update(self.opts)

    def checkOpts(self):
        missing_opts = self.accepted_opt_names - set(self.opts)
        if missing_opts:
            raise Exception(f"Missing options {missing_opts}")

        invalid_name_opts = set(self.opts) - self.accepted_opt_names
        if invalid_name_opts:
            raise Exception(f"Unrecognized options {invalid_name_opts}, acceptable: {self.accepted_opt_names}")

        if self.scenario not in {"optimistic", "pessimistic"}:
            raise OptionValueError(opt_name="scenario", got=self.scenario, acceptable={"optimistic", "pessimistic"})
        if self.n_sp_bins not in {4, 12, 24, 40}:
            raise OptionValueError(opt_name="n_sp_bins", got=self.n_sp_bins, acceptable={4, 12, 24, 40})
        if not isinstance(self.gcph_minus_gcsp, bool):
            raise TypeError(f"gcph_minus_gcsp must be bool, got {type(self.gcph_minus_gcsp)}")
        if not isinstance(self.shot_noise_sp_reduced, bool):
            raise TypeError(f"shot_noise_sp_reduced must be bool, got {type(self.shot_noise_sp_reduced)}")
        if not isinstance(self.gcph_only_bins_in_spectro_range, bool):
            raise TypeError(f"gcph_only_bins_in_spectro_range must be bool, got {type(self.shot_noise_sp_reduced)}")

    @property
    def scenario(self) -> "str":
        return self.opts['scenario']

    @property
    def n_sp_bins(self) -> "int":
        return self.opts['n_sp_bins']

    @property
    def gcph_minus_gcsp(self):
        return self.opts["gcph_minus_gcsp"]

    @property
    def shot_noise_sp_reduced(self):
        return self.opts["shot_noise_sp_reduced"]

    @property
    def gcph_only_bins_in_spectro_range(self):
        return self.opts["gcph_only_bins_in_spectro_range"]

    @property
    def bool_opts(self) -> "Dict[str, bool]":
        return {
            key: value for key, value in self.opts.items() if isinstance(value, bool)
        }

    def writeJSON(self, outdir: "TPathLike", add_datetime=True):
        file_path = Path(outdir) / self.createFileName(add_datetime=add_datetime)
        self.config.toJSON(file_path)

    def createFileName(self, add_datetime=True) -> "str":
        filename = f"{self.scenario}_{self.n_sp_bins}_sp_bins"
        for flag_name, flag in self.bool_opts.items():
            if flag:
                filename += f"_{flag_name}"

        if add_datetime:
            filename += f"_{fm.datetime_str_format(self.creation_date)}"

        filename += ".json"

        return filename

    def updateData(self):
        self.updateMultipoles()
        self.updateGCphDensityAndBias()
        self.updateGCspDensityAndBias()

    def updateMultipoles(self):
        for probe in ["GCph", "GCsp"]:
            self.config.probes[probe]["l_min"] = 10
            self.config.probes[probe]["l_max"] = 750 if self.scenario == "pessimistic" else 3000
            self.config.probes[probe]["log_l_number"] = 100
            self.config.probes[probe]["ell_log_selection"] = True

        if self.scenario == 'pessimistic':
            self.config.WL["ell_external_filename"] = "wl_pess_ells_10_750_log_nell_100_and_750_1500_lin.txt"
        else:
            self.config.WL["ell_external_filename"] = "wl_optm_ells_10_3000_log_nell_100_and_3000_5000_lin.txt"

    def updateGCphDensityAndBias(self):
        if self.gcph_minus_gcsp:
            if self.gcph_only_bins_in_spectro_range:
                raise NotImplementedError("gcph_minus_gcsp and gcph_only_bins_in_spectro_range cannot be both true")
            self.config.GCph["density_file"] = "gcph_dndz_redbook_gcsp_subtracted.h5"
        else:
            if self.gcph_only_bins_in_spectro_range:
                self.config.GCph["density_file"] = "gcph_dndz_redbook_only_bins_in_spectro_range.h5"
                self.config.GCph["bias_file"] = "gcph_bias_piecewise_only_bins_in_spectro_range.h5"
            else:
                self.config.GCph["density_file"] = "gcph_dndz_redbook.h5"
                self.config.GCph["bias_file"] = "gcph_bias_piecewise.h5"

    def updateGCspDensityAndBias(self):
        self.config.GCsp["density_file"] = f"gcsp_dndz_{self.n_sp_bins}_bins.h5"
        self.config.GCsp["bias_file"] = f"gcsp_bias_piecewise_{self.n_sp_bins}_bins.h5"
