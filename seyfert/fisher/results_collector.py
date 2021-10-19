from typing import List, Dict
import pandas as pd
import logging

from seyfert.fisher.fisher_analysis import FisherAnalysis
from seyfert.utils.dir_explorer import DirectoryExplorer
from seyfert.utils import filesystem_utils as fsu
from seyfert.utils.workspace import WorkSpace
from seyfert.config.forecast_config import ForecastConfig
from seyfert.utils.type_helpers import TPathLike

logger = logging.getLogger(__name__)


class ResultsCollector:
    ws_list: "List[WorkSpace]"
    fcfg_list: "List[ForecastConfig]"
    explorer: "DirectoryExplorer"
    analyses_df: "pd.DataFrame"

    def __init__(self):
        self.ws_list = None
        self.explorer = None
        self.analyses_df = None
        self.fcfg_list = None

    @property
    def root_path(self):
        return self.explorer.root_path

    def loadWorkSpaces(self, root_path: "TPathLike", **kwargs):
        self.explorer = DirectoryExplorer(root_path)
        self.ws_list = self.explorer.selectWorkspaces(**kwargs)
        self.fcfg_list = [
            ws.getForecastConfiguration() for ws in self.ws_list
        ]

    def buildRelativeMarginalizedErrorsTable(self, **kwargs) -> "pd.DataFrame":
        table = self.buildErrorsTable(file_pattern="relative_marginalized_errors*.csv", **kwargs)

        return table

    def buildMarginalizedErrorsTable(self, **kwargs) -> "pd.DataFrame":
        table = self.buildErrorsTable(file_pattern="marginalized_errors*.csv", **kwargs)

        return table

    def buildErrorsTable(self, file_pattern: "str", **kwargs):
        dfs = []
        for i, ws in enumerate(self.ws_list):
            results_dir = ws.getResultsDir(**kwargs)
            rel_errs_file = fsu.get_file_from_dir_with_pattern(results_dir, file_pattern)
            df = pd.read_csv(rel_errs_file, index_col=0)
            if df.index.name is None:
                df.index.name = 'fisher'
            fc = self.fcfg_list[i]
            for name, value in fc.synthetic_opts.items():
                df[name] = value
            dfs.append(df)

        big_df = pd.concat(dfs).reset_index()

        return big_df

    def buildFisherAnalysesDataFrame(self, cosmology: "str" = "w0_wa_CDM",
                                     res_subdir_name: "str" = "marg_before") -> "pd.DataFrame":
        rows = []
        for i, ws in enumerate(self.ws_list):
            an = FisherAnalysis.fromRundir(ws.run_dir, cosmology=cosmology, res_subdir_name=res_subdir_name)
            rows.append([*self.fcfg_list[i].synthetic_opts.values(), an])

        an_df = pd.DataFrame(rows, columns=[*self.fcfg_list[0].synthetic_opts.keys(), 'analysis'])

        return an_df

    def collectAndDumpResults(self, outdir: "TPathLike", cosmology: "str" = "w0_wa_CDM",
                              res_subdir_name: "str" = "marg_before"):
        out_subdir = outdir / cosmology / res_subdir_name
        out_subdir.mkdir(exist_ok=True, parents=True)

        logger.info("Building marginalized errors table")
        marg_errs_df = self.buildMarginalizedErrorsTable(cosmology=cosmology, res_subdir_name=res_subdir_name)
        logger.info("Building relative marginalized errors table")
        rel_marg_errs_df = self.buildRelativeMarginalizedErrorsTable(cosmology=cosmology, res_subdir_name=res_subdir_name)

        marg_errs_df.to_excel(out_subdir / "marginalized_errors.xlsx", index=False)
        rel_marg_errs_df.to_excel(out_subdir / "relative_marginalized_errors.xlsx", index=False)

        logger.info("Building fisher analyses dataframe")
        an_df = self.buildFisherAnalysesDataFrame(cosmology=cosmology, res_subdir_name=res_subdir_name)

        an_df.to_pickle(out_subdir / "fisher_analyses.pickle")

    def loadFisherAnalysesPickle(self, file: "TPathLike", eval_marg_errs=True):
        self.analyses_df = pd.read_pickle(file)

        if eval_marg_errs:
            for an in self.analyses_df['analysis'].values:
                if an.marginalized_errors is None:
                    an.evaluateMarginalizedErrors()
                if an.relative_marginalized_errors is None:
                    an.evaluateRelativeMarginalizedErrors()

    def replaceHybridFishersWith4BinsEquivalents(self):
        indices = ['n_sp_bins', 'scenario', 'shot_noise_sp_reduced', 'gcph_minus_gcsp',
                   'gcph_only_bins_in_spectro_range']
        self.analyses_df.set_index(indices, inplace=True)

        for idx, row in self.analyses_df.iterrows():
            n_sp_bins, *other = idx
            if n_sp_bins != 4:
                an_4_bins: "FisherAnalysis" = self.analyses_df.loc[tuple([4, *other])].analysis
                an_current: "FisherAnalysis" = row.analysis
                for fisher_name in list(an_current.fisher_matrices):
                    if "Pk" in fisher_name:
                        an_current.fisher_matrices[fisher_name] = an_4_bins.fisher_matrices[fisher_name]
                        an_current.marginalized_errors.loc[fisher_name] = an_4_bins.marginalized_errors.loc[fisher_name]

        self.analyses_df.reset_index(inplace=True)

    def getFisherAnalysesSubDf(self, sel_dict: "Dict", sel_fishers: "List[str]" = None) -> "pd.DataFrame":
        query_str = " and ".join(f"{key} == {value}" for key, value in sel_dict.items())
        subdf = self.analyses_df.query(query_str).reset_index()

        if sel_fishers is not None:
            subdf['analysis'] = subdf['analysis'].apply(lambda an: an.getSubsetAnalysis(sel_fishers))

        return subdf

    def getErrorsTableFromAnalysesDataFrame(self, relative: "bool") -> "pd.DataFrame":
        attr_name = "relative_marginalized_errors" if relative else "marginalized_errors"
        errs_dfs = []
        for row in self.analyses_df.itertuples(index=False):
            row_dict = row._asdict()
            an_errs_df = getattr(row.analysis, attr_name).reset_index()
            for key, value in row_dict.items():
                if key != 'analysis':
                    an_errs_df[key] = value

            errs_dfs.append(an_errs_df)

        return pd.concat(errs_dfs)

    def getMarginalizedErrorsTableFromAnalysesDataFrame(self) -> "pd.DataFrame":
        return self.getErrorsTableFromAnalysesDataFrame(relative=False)

    def getRelativeMarginalizedErrorsTableFromAnalysesDataFrame(self) -> "pd.DataFrame":
        return self.getErrorsTableFromAnalysesDataFrame(relative=True)
