import os
import shutil
from typing import Union, Tuple, Set, Dict, List
import json
from pathlib import Path
import logging

from seyfert.utils import filesystem_utils as fsu
from seyfert.config.forecast_config import ForecastConfig
from seyfert.config import main_config as mcfg
from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.utils.type_helpers import TPathLike

logger = logging.getLogger(__name__)


class WorkSpace:
    run_dir: "Path"
    task_dirs: "Dict[str, Path]"

    def __init__(self, run_dir: "TPathLike"):
        self.run_dir = Path(run_dir)
        self.task_dirs = {
            name: self.run_dir / name for name in ["PowerSpectrum", "Angular", "Derivative", "Fisher"]
        }
        self.niz_file = self.run_dir / 'norm_densities.hdf5'
        self.delta_cls_file = self.run_dir / 'delta_cls.h5'
        self.metadata_file = self.run_dir / "run_metadata.json"
        self.auto_f_ells_file = self.run_dir / "auto_fisher_of_ells.h5"
        self.base_fishers_dir = self.run_dir / "base_fishers"
        self.run_metadata = None

    def __repr__(self) -> "str":
        return self.run_dir.name

    @property
    def input_files_dir(self) -> "Path":
        return self.run_dir / 'input_files_dir'

    @property
    def pmm_dir(self) -> "Path":
        return self.task_dirs["PowerSpectrum"]

    @property
    def cl_dir(self) -> "Path":
        return self.task_dirs["Angular"]

    @property
    def der_dir(self) -> "Path":
        return self.task_dirs["Derivative"]

    @property
    def fish_elems_dir(self) -> "Path":
        return self.task_dirs["Fisher"]

    @property
    def final_results_dir(self) -> "Path":
        return self.run_dir / 'final_results'

    def theInputFilesDict(self) -> "Dict":
        with open(self.run_dir / 'input_files_dict.json') as jsf:
            if_dict = json.load(jsf)

        return if_dict

    @staticmethod
    def getDvarStepFromCosmoDirName(dirname: "str") -> "Tuple[str, int]":
        return fsu.get_dvar_step_from_cosmology_jobdir_name(dirname)

    @staticmethod
    def getDvarFromDerivativeDirName(dirname: "str") -> "str":
        return fsu.get_dvar_from_derivative_jobdir_name(dirname)

    @staticmethod
    def getDvarsFromFisherDirName(dirname: "str") -> "Tuple[str]":
        return fsu.get_dvar1_dvar2_from_fisher_jobdir_name(dirname)

    def getForecastConfiguration(self, load_params: "bool" = False,
                                 input_data_dir: "TPathLike" = None) -> "ForecastConfig":

        input_dir = Path(input_data_dir) if input_data_dir is not None else self.getInputDataDirPath()
        fc = ForecastConfig(input_file=self.getForecastConfigFilePath(),
                            input_data_dir=input_dir)
        if load_params:
            fc.loadPhysicalParametersFromJSONConfig()

        return fc

    def getRunMetadata(self) -> "Dict":
        if self.metadata_file.is_file():
            with open(self.metadata_file, mode="r") as jsf:
                metadata = json.load(jsf)
        else:
            logger.warning(f"metadata file {self.metadata_file} is not a file, returning empty dict")
            metadata = {}

        return metadata

    def loadRunMetadata(self):
        self.run_metadata = self.getRunMetadata()

    def collectPmmVariationParams(self) -> "Set[str]":
        return {
            self.getDvarStepFromCosmoDirName(d.name)[0] for d in self.pmm_dir.iterdir() if d.is_dir()
        }

    def collectClVariationParams(self) -> "Set[str]":
        dvars = set()
        for d in self.cl_dir.iterdir():
            if d.is_dir():
                if "central" not in d.name:
                    dvars.add(self.getDvarStepFromCosmoDirName(d.name)[0])

        return dvars

    def collectDerivativeParams(self) -> "Set[str]":
        return {
            self.getDvarFromDerivativeDirName(d.name) for d in self.der_dir.iterdir() if d.is_dir()
        }

    def collectFisherParamsPairs(self) -> "Set[Tuple[str]]":
        return {
            self.getDvarsFromFisherDirName(d.name) for d in self.fish_elems_dir.iterdir() if d.is_dir()
        }

    def getForecastConfigFilePath(self) -> "Path":
        return fsu.get_forecast_config_file_from_rundir(self.run_dir)

    def getInputDataDirPath(self) -> "Path":
        input_data_dir = fsu.get_forecast_input_data_dir_from_rundir(self.run_dir)
        if not input_data_dir.is_dir():
            logger.warning(f"input data dir {input_data_dir} is not a directory")
            input_data_dir = fsu.default_seyfert_input_data_dir()
            logger.warning(f"Trying with {input_data_dir}")

        return input_data_dir

    def getTaskJSONPath(self, task_name: "str") -> "Path":
        input_dict = self.theInputFilesDict()
        try:
            return self.run_dir / input_dict[task_name]
        except KeyError:
            raise KeyError(f"Invalid task name {task_name}")

    def getTaskJSONConfiguration(self, task_name: "str") -> "mcfg.ConcreteConfigType":
        json_file = self.getTaskJSONPath(task_name)

        return mcfg.config_for_task(task_name, json_file)

    def getTasksJSONConfigs(self) -> "Dict[str, mcfg.MainConfig]":
        return {
            task_name: self.getTaskJSONConfiguration(task_name) for task_name in self.task_dirs.keys()
        }

    def getParamsJSONFile(self) -> "Path":
        return fsu.get_params_file_from_rundir(self.run_dir)

    def getParamsCollection(self) -> "PhysicalParametersCollection":
        params_file = fsu.get_params_file_from_rundir(self.run_dir)

        return PhysicalParametersCollection.fromJSON(params_file)

    def getPowerSpectrumFile(self, dvar: "str", step: "int") -> "Path":
        return self.pmm_dir / fsu.get_cosmology_jobdir_name(dvar=dvar, step=step) / 'p_mm.h5'

    def getClFile(self, dvar: "str", step: "int") -> "Path":
        cl_dir = self.cl_dir / fsu.get_cosmology_jobdir_name(dvar=dvar, step=step)

        return fsu.get_file_from_dir_with_pattern(cl_dir, "cl*.h5")

    def getClDerivativeFile(self, dvar: "str") -> "Path":
        cl_der_dir = self.der_dir / fsu.get_derivative_jobdir_name_from_dvar(dvar)

        return fsu.get_file_from_dir_with_pattern(cl_der_dir, "cl_derivatives*.h5")

    def collectClDerivativeFiles(self) -> "Dict[str, Path]":
        dvars = self.collectDerivativeParams()
        cl_der_files = {}
        for dvar in dvars:
            cl_der_files[dvar] = self.getClDerivativeFile(dvar)

        return cl_der_files

    def collectClVariationDirectories(self) -> "Dict[Tuple, Path]":
        cl_var_dirs = {}
        for x in self.cl_dir.glob("dvar_*"):
            dvar, step = self.getDvarStepFromCosmoDirName(x.name)
            if dvar != 'central':
                cl_var_dirs[(dvar, step)] = x

        return cl_var_dirs

    def getResultsDir(self, results_dirname: "str" = "final_results",
                      cosmology: "str" = "w0_wa_CDM", res_subdir_name: "str" = "marg_before") -> "Path":
        results_dir = self.run_dir / results_dirname / cosmology / res_subdir_name
        if not results_dir.is_dir():
            logger.warning(f"Not a directory: {results_dir}")

        return results_dir

    def symlinkToExternalDirs(self, ext_dirs: "Dict", src_relative_to_rundir=True, link_delta_cls=True):
        unrecognized_tasks = set(ext_dirs) - set(self.task_dirs)
        if unrecognized_tasks:
            raise KeyError(f"Unrecognized tasks {unrecognized_tasks}")

        for key, ext_dir in ext_dirs.items():
            if not self.task_dirs[key].exists():
                logger.info(f"Creating symlink to {ext_dir} at {self.task_dirs[key]}")
                self.createSymlink(src=ext_dir, dst=self.task_dirs[key],
                                   src_relative_to_rundir=src_relative_to_rundir)
                if key == "Angular" and link_delta_cls:
                    src_ws = WorkSpace(Path(ext_dir).parent)
                    src_delta_cls = src_ws.delta_cls_file
                    self.createSymlink(src=src_delta_cls, dst=self.delta_cls_file,
                                       src_relative_to_rundir=src_relative_to_rundir)
            else:
                logger.info(f"Directory already exists {self.task_dirs[key]}")

    def createSymlink(self, src, dst, src_relative_to_rundir=True):
        src = Path(src)
        if not src.exists():
            logger.warning(f"cwd is {os.getcwd()}")
            raise FileNotFoundError(src)
        dst_path = Path(dst)
        symlink_src = Path(os.path.relpath(src, self.run_dir)) if src_relative_to_rundir else Path(src)
        dst_path.symlink_to(symlink_src)

    def createInputFilesDir(self, src_input_files: "Dict", phys_pars: "PhysicalParametersCollection",
                            input_data_dir: "Path"):
        input_files_dir = fsu.get_input_files_dir_for_rundir(self.run_dir)
        input_files_dir.mkdir(exist_ok=True)
        input_files_dict = {}

        for name, input_file in src_input_files.items():
            src = Path(input_file)
            dst = input_files_dir / src.name
            shutil.copy(src, dst)
            input_files_dict[name] = os.path.relpath(dst, self.run_dir)

        params_file = input_files_dir / 'params.json'
        phys_pars.writeJSON(params_file)
        input_files_dict['params'] = os.path.relpath(params_file, self.run_dir)
        logger.info(f'Writing physical parameters to JSON {params_file}')
        input_files_dict['input_data_dir'] = str(input_data_dir.resolve())
        with open(fsu.get_input_files_JSON_for_rundir(self.run_dir), mode="w") as jsf:
            json.dump(input_files_dict, jsf, indent=2)
