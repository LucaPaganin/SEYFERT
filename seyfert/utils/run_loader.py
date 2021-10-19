import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import re
from typing import Union, Dict, Callable, Set, Iterable, List
import datetime
import logging
from functools import partial

from seyfert.utils import filesystem_utils as fsu
from seyfert.utils.general_utils import map_nested_dict
from seyfert.numeric.general import pad
from seyfert.cosmology import c_ells
from seyfert.utils.workspace import WorkSpace
from seyfert.cosmology.cosmology import Cosmology
from seyfert.cosmology.power_spectrum import PowerSpectrum
from seyfert.config.forecast_config import ForecastConfig
from seyfert.config.main_config import MainConfig
from seyfert.fisher.fisher_matrix import FisherMatrix
from seyfert.fisher.fisher_results import FisherResultsCollector
from seyfert.utils.formatters import datetime_str_format
from seyfert.file_io import retro_comp

logger = logging.getLogger(__name__)


class RunLoader:
    pmms: "Dict[str, PowerSpectrum]"
    cosmos: "Dict[str, Cosmology]"
    fishers: "Dict[str, FisherMatrix]"
    metadata: "Dict[str, str]"
    results: "FisherResultsCollector"
    workspace: "WorkSpace"
    main_configs: "Dict[str, MainConfig]"

    def __init__(self, run_dir: "Union[str, Path]"):
        self.metadata = None
        self.fc = None
        self.main_configs = None
        self.version = None
        self.phys_pars = None
        self.workspace = WorkSpace(run_dir=run_dir)
        self.pmms = {}
        self.cosmos = {}
        self.cls = {}
        self.dcls = {}
        self.fishers = {}
        self.results = None
        self.z_grid = None
        self.cosmo_keys = None

    @property
    def run_dir(self):
        return self.workspace.run_dir

    @property
    def marg_errs(self):
        return self.results.marg_errs

    def loadRunData(self):
        logger.info(f"loading data for run {self.run_dir.name}")
        logger.info('loading metadata')
        self.loadMetadata()
        logger.info('loading power spectra')
        self.loadPowerSpectra()
        logger.info('loading cosmologies and cls')
        self.loadCosmologiesAndCls()
        logger.info('loading cl derivatives')
        self.loadClDerivatives()
        logger.info('loading fishers')
        self.loadFishers()
        logger.info('loading results')
        self.loadResults()

    def loadMetadata(self):
        try:
            with open(self.run_dir / "run_metadata.json", mode="r") as jsf:
                self.metadata = json.load(jsf)
        except FileNotFoundError:
            logger.warning('metadata not found, skipping')
        vregex = re.compile(r'([0-9]\.[0-9]\.[0-9][a-z]?)(dev)?')
        self.version = vregex.search(self.run_dir.name).groups()[0]

    def loadConfigs(self):
        logger.info(f'Loading forecast config from {self.run_dir}')
        self.fc = ForecastConfig(input_file=self.workspace.getForecastConfigFilePath(),
                                 input_data_dir=self.workspace.getInputDataDirPath())
        logger.info(f'Loading physical parameters from {self.run_dir}')
        self.fc.loadPhysicalParametersFromJSONConfig()
        self.phys_pars = self.fc.phys_pars
        self.main_configs = self.workspace.getTasksJSONConfigs()

    def loadPowerSpectra(self):
        pmm_dir = self.workspace.pmm_dir
        self.pmms = {}
        if pmm_dir.exists():
            for jobdir in [d for d in pmm_dir.iterdir() if d.is_dir()]:
                dvar, step = fsu.get_dvar_step_from_cosmology_jobdir_name(jobdir.name)
                job_key = f'{dvar}_{step}'
                self.pmms[job_key] = PowerSpectrum.fromHDF5(jobdir / 'p_mm.h5')
        else:
            logger.warning(f"power spectra dir {pmm_dir} does not exist, skipping")

    def loadCosmologiesAndCls(self, params: "Iterable[str]" = None):
        cl_dir = self.run_dir / "Angular"
        all_pars = self.workspace.collectClVariationParams()
        pars_to_collect = all_pars
        if params is not None:
            pars_to_collect = set(params)
            if not pars_to_collect.issubset(all_pars):
                intruders = pars_to_collect - all_pars
                raise Exception(f"parameters {', '.join(intruders)} are not present in the directory")

        jb_keys = []
        for jobdir in [d for d in cl_dir.iterdir() if d.is_dir()]:
            dvar, step = fsu.get_dvar_step_from_cosmology_jobdir_name(jobdir.name)
            if dvar in pars_to_collect:
                job_key = f'{dvar}_{step}'
                jb_keys.append(job_key)
                cosmo_file = jobdir / 'cosmology.h5'
                if cosmo_file.exists():
                    self.cosmos[job_key] = retro_comp.read_cosmo_file(cosmo_file, version=self.version,
                                                                      load_power_spectrum=False)
                cl_file = fsu.get_file_from_dir_with_pattern(jobdir, 'cl*.h5')
                coll = retro_comp.read_cls_file(cl_file, version=self.version)
                if self.z_grid is None:
                    self.z_grid = coll.z_grid
                self.cls[job_key] = coll
            else:
                pass
        self.cosmo_keys = np.array(jb_keys)
        if len(self.cosmos) == 0:
            logger.warning(f'cosmologies not present for run {self.run_dir.name}')

    def loadClDerivatives(self):
        der_dir = self.run_dir / "Derivative"
        for jobdir in [d for d in der_dir.iterdir() if d.is_dir()]:
            dvar = fsu.get_dvar_from_derivative_jobdir_name(jobdir.name)
            file = fsu.get_file_from_dir_with_pattern(jobdir, "cl*.h5")
            dcl_data = retro_comp.read_dcl_file(file)
            self.dcls[dvar] = {}
            for cl_key in dcl_data:
                short_key = c_ells.cl_key_long_to_short(cl_key)
                self.dcls[dvar][short_key] = dcl_data[cl_key]

    def loadFishers(self):
        fishers_dir = self.run_dir / 'fisher_matrices'
        for file in fishers_dir.glob("*.hdf5"):
            f_mat = FisherMatrix()
            f_mat.loadFromFile(file)
            f_mat.selectRelevantFisherSubMatrix()
            if f_mat.inverse is None:
                f_mat.evaluateInverse()

            self.fishers[f_mat.name] = f_mat

    def loadResults(self):
        results_dir = self.run_dir / 'results'
        if results_dir.exists():
            results_config = fsu.config_files_dir() / 'results_config.json'
            self.results = FisherResultsCollector(phys_pars=self.phys_pars, analysis_name='test_comparison',
                                                  results_config_file=results_config)
            self.results.loadFromResultsDir(results_dir)
        else:
            logger.warning('results dir not found, skipping')

    def pickle_dump(self, file: "Union[str, Path]"):
        logger.info(f"Dumping run data to {file}")
        with open(file, mode="wb") as f_handle:
            pickle.dump(self, f_handle)

    @staticmethod
    def pickle_load(file: "Union[str, Path]") -> "RunLoader":
        with open(file, mode="rb") as f_handle:
            obj = pickle.load(f_handle)
        return obj


class RunComparison:
    rl1: "RunLoader"
    rl2: "RunLoader"
    metrics: "ComparisonMetrics"
    _metric_func: "Callable"

    def __init__(self, rundir1: "Union[str, Path]" = None, rundir2: "Union[str, Path]" = None,
                 metric_func: "Callable" = pad):
        self.rl1 = None
        self.rl2 = None
        self.metadata = None
        self._metric_func = metric_func
        self.metrics = None

        if rundir1 is not None and rundir2 is not None:
            self.rl1 = RunLoader(rundir1)
            self.rl2 = RunLoader(rundir2)

    def loadRunData(self):
        self.rl1.loadRunData()
        self.rl2.loadRunData()
        self.metadata = {
            'run1': self.rl1.metadata,
            'run2': self.rl2.metadata
        }

    def evaluateMetricsAndTest(self):
        self.metrics = ComparisonMetrics(self.rl1, self.rl2, metric_func=self._metric_func)
        self.metrics.evaluate()
        self.metrics.testAllMetrics()

    def saveComparisonData(self, outdir: "Union[str, Path]", save_run_pickle: "bool" = False):
        now = datetime_str_format(datetime.datetime.now())
        main_outdir = Path(outdir) / f"comp_{self.rl1.version}_vs_{self.rl2.version}_{now}"
        logger.info(f"Creating directory {main_outdir.name}")
        main_outdir.mkdir(exist_ok=True, parents=True)
        with open(main_outdir / "metadata.json", mode="w") as jsf:
            json.dump([
                self.rl1.metadata,
                self.rl2.metadata
            ], jsf, indent=2)

        if save_run_pickle:
            self.rl1.pickle_dump(main_outdir / "run1.pickle")
            self.rl2.pickle_dump(main_outdir / "run2.pickle")

        self.metrics.writeOnDisk(main_outdir)

    def loadComparisonData(self, inputdir: "Union[str, Path]"):
        inputdir = Path(inputdir)
        with open(inputdir / "metadata.json", mode='r') as jsf:
            self.metadata = json.load(jsf)

        self.metrics = ComparisonMetrics()
        self.metrics.loadFromDisk(inputdir)


class ComparisonMetrics:
    _metric: "Callable"

    def __init__(self, rl1: "RunLoader" = None, rl2: "RunLoader" = None, metric_func: "Callable" = None):
        self.rl1 = rl1
        self.rl2 = rl2
        self._metric = metric_func
        self.workdir = None
        self.metrics = {
            'pmms': {},
            'cosmos': {},
            'cls': {},
            'dcls': {},
            'fisher': {},
            'inverse_fisher': {},
            'marg_errs': {}
        }

    @property
    def pmms(self) -> "Dict":
        return self.metrics['pmms']

    @property
    def cosmos(self) -> "Dict":
        return self.metrics['cosmos']

    @property
    def cls(self) -> "Dict":
        return self.metrics['cls']

    @property
    def dcls(self) -> "Dict":
        return self.metrics['dcls']

    @property
    def fisher(self) -> "Dict":
        return self.metrics['fisher']

    @property
    def inverse_fisher(self) -> "Dict":
        return self.metrics['inverse_fisher']

    @property
    def marg_errs(self) -> "Dict":
        return self.metrics['marg_errs']

    def evaluate(self, eval_pmm_metric=False):
        if eval_pmm_metric:
            logger.info("Evaluating Pmm metrics")
            self.evaluatePmmMetrics()
        logger.info("Evaluating cosmology metrics")
        self.evaluateCosmoMetrics()
        logger.info("Evaluating cls metrics")
        self.evaluateClDataMetrics()
        logger.info("Evaluating dcls metrics")
        self.evaluateClDerivativesMetrics()
        logger.info("Evaluating fisher metrics")
        self.evaluateFisherMetrics()
        logger.info("Evaluating marg_errs metrics")
        self.evaluateMargErrsMetrics()

    def evaluatePmmMetrics(self):
        pads_keys = self.getComparisonKeys(self.rl1.pmms, self.rl2.pmms)
        for jobkey in pads_keys:
            self.pmms[jobkey] = {
                'lin': self._metric(self.rl1.pmms[jobkey].lin_p_mm_z_k, self.rl2.pmms[jobkey].lin_p_mm_z_k),
                'nonlin': self._metric(self.rl1.pmms[jobkey].nonlin_p_mm_z_k, self.rl2.pmms[jobkey].nonlin_p_mm_z_k)
            }

    def evaluateCosmoMetrics(self):
        pads_keys = self.getComparisonKeys(self.rl1.cosmos, self.rl2.cosmos)
        for jobkey in pads_keys:
            self.cosmos[jobkey] = {
                'E_z': self._metric(self.rl1.cosmos[jobkey].E_z, self.rl2.cosmos[jobkey].E_z),
                'r_tilde_z': self._metric(self.rl1.cosmos[jobkey].r_tilde_z, self.rl2.cosmos[jobkey].r_tilde_z)
            }

    def evaluateClDataMetrics(self):
        pads_keys = self.getComparisonKeys(self.rl1.cls, self.rl2.cls)
        for jobkey in pads_keys:
            cl_coll1 = self.rl1.cls[jobkey]
            cl_coll2 = self.rl2.cls[jobkey]
            self.cls[jobkey] = self.buildClDataMetrics(cl_coll1, cl_coll2)

    def evaluateClDerivativesMetrics(self):
        pads_keys = self.getComparisonKeys(self.rl1.dcls, self.rl2.dcls)
        for dvar in pads_keys:
            dcls1 = self.rl1.dcls[dvar]
            dcls2 = self.rl2.dcls[dvar]

            cl_kind_pad_keys = set(dcls1.keys()).intersection(set(dcls2.keys()))
            self.dcls[dvar] = {
                cl_key: self._metric(dcls1[cl_key], dcls2[cl_key]) for cl_key in cl_kind_pad_keys
            }

    def evaluateFisherMetrics(self):
        pads_keys = self.getComparisonKeys(self.rl1.fishers, self.rl2.fishers)

        for key in pads_keys:
            self.fisher[key] = self._metric(self.rl1.fishers[key].matrix, self.rl2.fishers[key].matrix)
            self.inverse_fisher[key] = self._metric(self.rl1.fishers[key].inverse, self.rl2.fishers[key].inverse)

    def evaluateMargErrsMetrics(self):
        pads_keys = self.getComparisonKeys(self.rl1.marg_errs, self.rl2.marg_errs)

        for key in pads_keys:
            self.marg_errs[key] = self._metric(self.rl1.marg_errs[key], self.rl2.marg_errs[key])

    def testAllMetrics(self):
        self.checkPadsDict('marg_errs', self.marg_errs, threshold=1e-6)
        self.checkPadsDict('fisher', self.fisher, threshold=1e-5)
        self.checkPadsDict('inverse_fisher', self.inverse_fisher, threshold=1e-5)

    def buildClDataMetrics(self, coll1: "c_ells.AngularCoefficientsCollector",
                           coll2: "c_ells.AngularCoefficientsCollector") -> "Dict":
        data_metrics = {}
        cl_kind_keys = self.getComparisonKeys(coll1.short_keys, coll2.short_keys)
        for key in cl_kind_keys:
            cl_A = coll1[key]
            cl_B = coll2[key]
            data_metrics[key] = {
                'c_lij': self._metric(cl_A.c_lij, cl_B.c_lij),
                'n1_iz': self._metric(cl_A.weight1.density.norm_density_iz, cl_B.weight1.density.norm_density_iz),
                'n2_iz': self._metric(cl_A.weight2.density.norm_density_iz, cl_B.weight2.density.norm_density_iz),
                'w1_iz': self._metric(cl_A.weight1.w_bin_z, cl_B.weight1.w_bin_z),
                'w2_iz': self._metric(cl_A.weight2.w_bin_z, cl_B.weight2.w_bin_z),
                'k_ijz': self._metric(cl_A.kernel.k_ijz, cl_B.kernel.k_ijz)
            }

        return data_metrics

    def loadFromDisk(self, inputdir: "Union[str, Path]"):
        infile_pickle = Path(inputdir) / 'full_metrics.pickle'
        with open(infile_pickle, mode='rb') as f:
            self.metrics = pickle.load(f)

    def writeOnDisk(self, outdir: "Union[str, Path]"):
        outfile_pickle = Path(outdir) / 'full_metrics.pickle'
        with open(outfile_pickle, mode='wb') as f:
            pickle.dump(self.metrics, f)

    @staticmethod
    def checkPadsDict(name: "str", pads_dict: "Dict[str, pd.DataFrame]", threshold: "float"):
        logger.info(f"Testing {name} pads, threshold: {threshold:.1e}")
        conditions = {key: np.all(pads < threshold) for key, pads in pads_dict.items()}
        for key, cond in conditions.items():
            if not cond:
                logger.warning(f"Not all {name} {key} pads are below threshold {threshold:.1e}")
            else:
                logger.info(f"{name} {key} pads OK")

    @staticmethod
    def getComparisonKeys(data1: "Iterable", data2: "Iterable") -> "Set":
        keys1 = set(data1)
        keys2 = set(data2)
        if keys1 == keys2:
            keys = keys1
        else:
            xor1 = keys1 - keys2
            xor2 = keys2 - keys1
            keys = keys1.intersection(keys2)
            logger.warning(f"keys are not equal")
            logger.warning(f"unique keys for dict 1: {', '.join(xor1)}")
            logger.warning(f"unique keys for dict 2: {', '.join(xor2)}")

        return keys

