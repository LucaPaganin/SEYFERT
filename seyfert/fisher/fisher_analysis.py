from typing import Dict, Tuple, List, Union, Iterable, Set
import numpy as np
import math
import pandas as pd
import logging
import copy
from scipy import special
from pathlib import Path

from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.fisher.fisher_matrix import FisherMatrix
from seyfert.utils import tex_utils as txu
from seyfert.utils import general_utils as gu
from seyfert.utils import table_utils as tu
from seyfert.utils.workspace import WorkSpace

logger = logging.getLogger(__name__)


class FisherAnalysisError(Exception):
    pass


class FisherAnalysis:
    fisher_names: "List[str]"
    base_fishers: "Dict[str, FisherMatrix]"
    fisher_matrices: "Dict[str, FisherMatrix]"
    cosmo_pars_fiducials: "Dict[str, float]"
    cosmo_pars_names: "List[str]"
    n_sigma_range: "int"
    marginalized_errors: "pd.DataFrame"
    metadata: "Dict"

    def __init__(self, analysis_name: "str" = None, cosmology_name: "str" = None, fishers_order: "List[str]" = None,
                 fisher_matrices: "Union[Dict[str, FisherMatrix], Iterable[FisherMatrix]]" = None,
                 cosmo_pars_fiducial: "Dict[str, float]" = None, marginalize_nuisance: "bool" = True):
        self.name = analysis_name if analysis_name is not None else "test_analysis"
        self.cosmology_name = cosmology_name
        self.base_fishers = None
        self.fisher_matrices = {}
        self.phys_pars = None
        self.cosmo_pars_fiducials = cosmo_pars_fiducial
        self.n_sigma_range = 3
        self.marginalized_errors = None
        self.relative_marginalized_errors = None
        self.translator = txu.TeXTranslator()
        self.marginalize_nuisance = marginalize_nuisance
        self._fishers_order = None
        self.metadata = {}
        if not isinstance(self.name, str):
            raise TypeError(f'analysis name must be str, got {type(self.name)}')
        if isinstance(fisher_matrices, list):
            if not all([isinstance(f, FisherMatrix) for f in fisher_matrices]):
                raise TypeError("Fishers must be all FisherMatrix instances")
            self.fisher_matrices = {}
        elif isinstance(fisher_matrices, dict):
            if not all([isinstance(f, FisherMatrix) for f in fisher_matrices.values()]):
                raise TypeError("Fishers must be all FisherMatrix instances")
            self.fisher_matrices.update(fisher_matrices)

        if self.fisher_matrices:
            for key in self.fisher_matrices:
                self.fisher_matrices[key].marginalize_nuisance = self.marginalize_nuisance

            self.sliceFishers()

        if fishers_order is not None:
            self.fishers_order = fishers_order

    @classmethod
    def fromFishersDir(cls, fishers_dir: "Union[str, Path]",
                       params: "PhysicalParametersCollection") -> "FisherAnalysis":
        an = cls(cosmo_pars_fiducial=params.free_cosmo_pars_fiducials)
        an.loadBaseFishersFromDir(fishers_dir=fishers_dir)
        an.useBaseFishers()
        an.phys_pars = params

        return an

    @classmethod
    def fromRundir(cls, rundir: "Union[str, Path]", use_base_fishers=True,
                   cosmology: "str" = "w0_wa_CDM", res_subdir_name: "str" = "marg_before") -> "FisherAnalysis":
        ws = WorkSpace(rundir)
        fishers_dir = ws.getResultsDir(cosmology=cosmology, res_subdir_name=res_subdir_name)
        params = ws.getParamsCollection()
        if cosmology == "LCDM":
            del params["w0"]
            del params["wa"]

        an = cls.fromFishersDir(fishers_dir=fishers_dir, params=params)
        fcfg = ws.getForecastConfiguration()
        an.metadata.update(fcfg.synthetic_opts)
        an.name = fcfg.getConfigID()
        if use_base_fishers:
            an.useBaseFishers()

        return an

    @property
    def correlation_matrices(self) -> "Dict[str, pd.DataFrame]":
        return {name: self.fisher_matrices[name].correlation for name in self.fisher_matrices}

    @property
    def cosmo_pars_names(self) -> "List":
        return list(self.cosmo_pars_fiducials.keys())

    @property
    def cosmo_pars_set(self) -> "Set":
        return set(self.cosmo_pars_fiducials)

    @property
    def fishers_order(self) -> "List[str]":
        if self._fishers_order is not None:
            return self._fishers_order
        else:
            return list(self.fisher_matrices)

    @fishers_order.setter
    def fishers_order(self, order: "List[str]"):
        if isinstance(order, list):
            if set(order) != set(self.fisher_matrices):
                raise Exception(f"order of fishers must contain all and only the fisher matrices of the analysis. "
                                f"Passed order is {order}, fishers in the analysis are {list(self.fisher_matrices)}")
            self._fishers_order = order
        else:
            raise TypeError(f"Invalid fishers order type {type(order)}")

    def __getitem__(self, item: "str") -> "FisherMatrix":
        return self.fisher_matrices[item]

    def __repr__(self) -> "str":
        return f"analysis {self.name}, {len(self.fisher_matrices)} fishers"

    def __iter__(self):
        return iter(self.fisher_matrices)

    def toDataFrameRow(self) -> "pd.Series":
        entries = {}
        entries.update(self.metadata)
        entries['analysis'] = self

        return pd.Series(entries)

    def prepareFisherMatrices(self):
        for name in self.fisher_matrices:
            self.fisher_matrices[name].selectRelevantFisherSubMatrix()
            self.fisher_matrices[name].evaluateInverse()
            self.fisher_matrices[name].evaluateCorrelationMatrix()

    def loadFromRundir(self, rundir: "Union[str, Path]", use_base_fishers=True, **kwargs):
        ws = WorkSpace(rundir)
        fishers_dir = ws.getResultsDir(**kwargs)
        self.loadBaseFishersFromDir(fishers_dir=fishers_dir)
        self.metadata = ws.getRunMetadata()
        self.name = self.metadata['run_id']
        self.phys_pars = ws.getParamsCollection()
        if use_base_fishers:
            self.useBaseFishers()

    def loadBaseFishersFromDir(self, fishers_dir: "Union[str, Path]") -> "None":
        self.base_fishers = {}
        fishers_dir = Path(fishers_dir)
        if not fishers_dir.exists():
            raise FileNotFoundError(fishers_dir)
        for fisher_file in fishers_dir.glob(f"fisher*.hdf5"):
            f_mat = FisherMatrix(marginalize_nuisance=self.marginalize_nuisance)
            f_mat.loadFromFile(fisher_file, file_ext='hdf5')
            if self.cosmo_pars_fiducials is not None:
                f_mat._cosmological_parameters = self.cosmo_pars_set

            f_mat.selectRelevantFisherSubMatrix()
            self.base_fishers[f_mat.name] = f_mat

        if len(self.base_fishers) == 0:
            logger.warning(f"no fisher matrices found into {fishers_dir}")

    def sliceBaseFishers(self):
        for key in self.base_fishers:
            self.base_fishers[key].selectRelevantFisherSubMatrix()

    def sliceFishers(self):
        for key in self.fisher_matrices:
            self.fisher_matrices[key].selectRelevantFisherSubMatrix()

    def invertFishers(self):
        for key in self.fisher_matrices:
            self.fisher_matrices[key].evaluateInverse()
            self.fisher_matrices[key].evaluateCorrelationMatrix()

    def marginalizeFishers(self, params: "Union[str, Set[str]]"):
        for key in self.fisher_matrices:
            logger.info(f"Marginalizing Fisher {key} over {params}")
            if params == "all_nuisance":
                to_marg_pars = self.fisher_matrices[key].nuisance_parameters
            else:
                to_marg_pars = params
            missing_pars = to_marg_pars - self.fisher_matrices[key].physical_parameters
            to_marg_pars = to_marg_pars.intersection(self.fisher_matrices[key].physical_parameters)
            if missing_pars:
                logger.warning(f"not all {params} belong to Fisher {key}, marginalizing only over "
                               f"{to_marg_pars}")
            self.fisher_matrices[key].marginalizeParameters(params=to_marg_pars, ret_copy=False)

    def useBaseFishers(self):
        if self.base_fishers is None:
            raise ValueError("base fishers needed, must load them before")
        self.fisher_matrices = {}
        self.fisher_matrices.update(self.base_fishers)

    def evaluateFishersFromAddendsList(self, fisher_addends_list: "List[List[str]]"):
        self.fisher_matrices = {}
        for addend_list in fisher_addends_list:
            if len(addend_list) == 1:
                name = addend_list[0]
                if name in self.base_fishers:
                    self.fisher_matrices[name] = self.base_fishers[name]
                else:
                    raise FisherAnalysisError(f'fisher name {name} not in base fishers')
            else:
                fisher_sum_name = '+'.join(addend_list)
                if not all([addend in self.base_fishers for addend in addend_list]):
                    raise FisherAnalysisError(f'one of the addend fishers {", ".join(addend_list)} is not '
                                              f'in the base fishers: {", ".join(self.base_fishers.keys())}')
                self.fisher_matrices[fisher_sum_name] = self.base_fishers[addend_list[0]].copy()
                for addend in addend_list[1:]:
                    self.fisher_matrices[fisher_sum_name] += self.base_fishers[addend]

    def saveFishersToDisk(self, outdir: "Union[str, Path]", overwrite=False):
        outdir = Path(outdir)
        for key in self.fisher_matrices:
            self.fisher_matrices[key].writeToFile(outdir=outdir, overwrite=overwrite)

    def getSubsetAnalysis(self, fisher_names: "List[str]", recompute_errs=False) -> "FisherAnalysis":
        an = copy.deepcopy(self)

        an.base_fishers = gu.subset_dict(self.base_fishers, fisher_names)
        an.fisher_matrices = gu.subset_dict(self.fisher_matrices, fisher_names)
        an.fishers_order = fisher_names
        if recompute_errs:
            an.evaluateMarginalizedErrors()
            an.evaluateRelativeMarginalizedErrors()
        else:
            an.marginalized_errors = self.marginalized_errors.loc[fisher_names]
            an.relative_marginalized_errors = self.relative_marginalized_errors.loc[fisher_names]

        return an

    def getParFiducialValue(self, name: "str") -> "float":
        return self.cosmo_pars_fiducials[name]

    def evaluateMarginalizedErrors(self) -> "None":
        self.marginalized_errors = pd.concat({key: F.getMarginalizedErrors() for key, F in self.fisher_matrices.items()},
                                             names=['fisher']).unstack()

        self.evaluateRelativeMarginalizedErrors()

    def evaluateRelativeMarginalizedErrors(self) -> "None":
        rel_marg_errs_den = {}
        for par, val in self.cosmo_pars_fiducials.items():
            rel_marg_errs_den[par] = val if val != 0 else 1

        rel_marg_errs_den = pd.Series(rel_marg_errs_den)
        if 'FoM' in self.marginalized_errors.columns:
            rel_marg_errs_den.loc['FoM'] = 1

        self.relative_marginalized_errors = np.abs(self.marginalized_errors / rel_marg_errs_den)

    def computeParameterRangeForObsAtCL(self, name: "str", obs: "str", CL: "float") -> "Tuple[float, float]":
        mean = self.getParFiducialValue(name)
        sigma = self.marginalized_errors.loc[obs, name]
        range_min = mean - np.sqrt(2) * special.erfinv(CL) * sigma
        range_max = mean + np.sqrt(2) * special.erfinv(CL) * sigma

        return range_min, range_max

    def computeGaussianInfoDictForObsAndCL(self, name: "str", obs: "str", CL: "float",
                                           x_min: "float", x_max: "float") -> "Dict":
        mean = self.getParFiducialValue(name)
        sigma = self.marginalized_errors.loc[obs, name]
        x = np.linspace(x_min, x_max, num=300)
        y = np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
        x_cl_min, x_cl_max = self.computeParameterRangeForObsAtCL(name, obs, CL)
        cl_mask = (x >= x_cl_min) & (x <= x_cl_max)
        gaus_dict = {
            'mean': mean, 'sigma': sigma,
            'cl_mask': cl_mask,
            'x': x, 'y': y
        }

        return gaus_dict

    def computeEllipseInfoDictForObsAndCL(self, name_x: "str", name_y: "str", obs: "str", CL: "float") -> "Dict":
        mean_x = self.getParFiducialValue(name_x)
        mean_y = self.getParFiducialValue(name_y)
        C_xx = self.fisher_matrices[obs].inverse.loc[name_x, name_x]
        C_yy = self.fisher_matrices[obs].inverse.loc[name_y, name_y]
        C_xy = self.fisher_matrices[obs].inverse.loc[name_x, name_y]
        cl_factor = np.sqrt(2) * special.erfinv(CL)
        a = cl_factor * np.sqrt((C_xx + C_yy) / 2 + np.sqrt((C_xx - C_yy)**2 / 4 + C_xy**2))
        b = cl_factor * np.sqrt((C_xx + C_yy) / 2 - np.sqrt((C_xx - C_yy)**2 / 4 + C_xy**2))
        angle = np.rad2deg(math.atan2(2*C_xy, (C_xx - C_yy)) / 2)
        ellipse_dict = {
            'xy': (mean_x, mean_y),
            'width':  2*a, 'height': 2*b,
            'angle':  angle
        }
        return ellipse_dict

    def writeAbsoluteAndRelativeMarginalizedErrors(self, outdir: "Union[str, Path]", overwrite=False) -> "None":
        self.writeMarginalizedErrorsToFile(outdir=outdir, overwrite=overwrite)
        self.writeRelativeMarginalizedErrorsToFiles(outdir=outdir, overwrite=overwrite)

    def writeMarginalizedErrorsToFile(self, outdir: "Union[str, Path]", overwrite=False) -> "None":
        outdir = Path(outdir)
        file_stem = f"marginalized_errors_{self.name}"
        tu.write_dataframe_table(self.marginalized_errors, outfile=outdir / f"{file_stem}.xlsx", overwrite=overwrite)
        tu.write_dataframe_table(self.marginalized_errors, outfile=outdir / f"{file_stem}.csv",  overwrite=overwrite)

    def writeRelativeMarginalizedErrorsToFiles(self, outdir: "Union[str, Path]", overwrite=False) -> "None":
        outdir = Path(outdir)
        file_stem = f"relative_marginalized_errors_{self.name}"
        tu.write_dataframe_table(self.relative_marginalized_errors, outfile=outdir / f"{file_stem}.xlsx",
                                 overwrite=overwrite)
        tu.write_dataframe_table(self.relative_marginalized_errors, outfile=outdir / f"{file_stem}.csv",
                                 overwrite=overwrite)

    def writeCorrelationMatricesToFiles(self, outdir: "Union[str, Path]") -> "None":
        for fisher_name in self.fisher_matrices:
            outfile = Path(outdir) / f'{self.name}_correlation_{fisher_name}.csv'
            self.fisher_matrices[fisher_name].correlation.to_csv(outfile, index=True)
