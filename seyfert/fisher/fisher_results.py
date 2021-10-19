from typing import Dict, Union
import json
import pandas as pd
import logging
from pathlib import Path
from seyfert.fisher.fisher_analysis import FisherAnalysis
from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.fisher.fisher_plot import FisherPlotter, FisherPlotConfig
from seyfert.utils import filesystem_utils as fsu
from seyfert.utils import formatters as fm

logger = logging.getLogger(__name__)


class FisherResultsCollector:
    fishers_dir: "Path"
    analysis_dict: "Dict[str, FisherAnalysis]"
    plotters_dict: "Dict[str, FisherPlotter]"
    cosmologies_dict: "Dict"
    outdirs_dict: "Dict"

    def __init__(self, phys_pars: "PhysicalParametersCollection" = None, analysis_name: "str" = None,
                 plot_ranges_file: "Union[str, Path]" = None, fishers_dir: "Union[str, Path]" = None,
                 results_config_file: "Union[str, Path]" = None, do_plots: "bool" = True,
                 marginalize_nuisance: "bool" = True):
        self.phys_pars = phys_pars
        self.analysis_name = analysis_name
        self.fishers_dir = fishers_dir
        self.plot_ranges_file = plot_ranges_file
        self.marginalize_nuisance = marginalize_nuisance
        self.analysis_dict = None
        self.plotters_dict = None
        self.cosmologies_dict = None
        self.results_config_dict = None
        self.outdirs_dict = None
        self.cosmo_pars_fiducials = None
        self.do_plots = do_plots
        if self.fishers_dir is not None:
            self.fishers_dir = Path(self.fishers_dir)
        if results_config_file is not None:
            self.loadFromConfigFile(results_config_file)
        if self.phys_pars is not None:
            self.cosmo_pars_fiducials = phys_pars.cosmo_pars_fiducials

    @property
    def fishers_addends_list(self):
        return self.results_config_dict["fishers_addends_list"]

    @property
    def plot_config_dict(self):
        return self.results_config_dict["plot_settings"]

    @property
    def marg_errs(self) -> "Dict":
        return {key: self.analysis_dict[key].marginalized_errors for key in self.analysis_dict}

    @property
    def rel_marg_errs(self) -> "Dict":
        return {key: self.analysis_dict[key].relative_marginalized_errors for key in self.analysis_dict}

    def loadFromConfigFile(self, results_config_file: Union[str, Path] = None) -> None:
        with open(results_config_file, 'r') as jsf:
            self.results_config_dict = json.load(jsf)
        self.cosmologies_dict = {
            key: self.results_config_dict["cosmologies"][key] for key in self.results_config_dict["cosmologies"]
            if fm.str_to_bool(self.results_config_dict["cosmologies"][key]["present"])
        }

    def checkCosmologies(self) -> "None":
        free_cosmo_pars_set = set(self.phys_pars.free_cosmological_parameters)
        for cosmology_name, cosmo_dict in self.cosmologies_dict.items():
            cosmo_set = set(cosmo_dict["pars"])
            if not cosmo_set.issubset(free_cosmo_pars_set):
                raise Exception(f'parameters {", ".join(cosmo_set - free_cosmo_pars_set)} '
                                f'missing in the forecast for {cosmology_name}')

    def setUp(self) -> "None":
        self.checkCosmologies()
        self.setupFisherAnalyses()
        if self.do_plots:
            self.setupFisherPlotters()

    def setupFisherAnalyses(self):
        self.analysis_dict = {}
        for cosmology_name, cosmo_dict in self.cosmologies_dict.items():
            cosmo_pars_fiducials = {name: self.cosmo_pars_fiducials[name] for name in cosmo_dict["pars"]}
            fish_analysis = FisherAnalysis(analysis_name=self.analysis_name,
                                           cosmology_name=cosmology_name,
                                           cosmo_pars_fiducial=cosmo_pars_fiducials,
                                           marginalize_nuisance=self.marginalize_nuisance)
            fish_analysis.loadBaseFishersFromDir(self.fishers_dir)
            fish_analysis.sliceBaseFishers()
            if self.fishers_addends_list:
                fish_analysis.evaluateFishersFromAddendsList(self.fishers_addends_list)
            else:
                fish_analysis.useBaseFishers()
            fish_analysis.prepareFisherMatrices()
            fish_analysis.evaluateMarginalizedErrors()
            fish_analysis.evaluateRelativeMarginalizedErrors()
            self.analysis_dict[cosmology_name] = fish_analysis

    def setupFisherPlotters(self):
        self.plotters_dict = {}
        for cosmology_name, cosmo_dict in self.cosmologies_dict.items():
            fish_analysis = self.analysis_dict[cosmology_name]
            pars_to_plot = cosmo_dict["pars_to_plot"]
            fisher_plot_config = FisherPlotConfig(fishers=fish_analysis.fisher_matrices,
                                                  n_cosmo_pars=len(pars_to_plot))
            fisher_plot_config.loadFromConfigDict(self.plot_config_dict)
            if self.plot_ranges_file is not None:
                fisher_plot_config.loadParameterPlotRangesFromFile(self.plot_ranges_file)
            fisher_plotter = FisherPlotter(fisher_analysis=fish_analysis,
                                           config=fisher_plot_config,
                                           pars_to_plot=pars_to_plot)
            fisher_plotter.setUpDefault()
            self.plotters_dict[cosmology_name] = fisher_plotter

    def writeResultsAndDoPlots(self, outdir: Union[str, Path]) -> None:
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        self.outdirs_dict = {}
        for cosmology_name in self.cosmologies_dict:
            self.createCosmologyOutdirStructure(main_outdir=outdir, cosmology_name=cosmology_name)
            self.writeResultsAndDoPlotsForCosmology(cosmology_name)

    @staticmethod
    def buildCosmologySubdirsDict(cosmology_dir: Path):
        subdir_types = ['errors', 'fishers', 'plots', 'correlations']
        return dict(zip(subdir_types, [cosmology_dir / x for x in subdir_types]))

    def createCosmologyOutdirStructure(self, main_outdir: Path, cosmology_name: str) -> None:
        cosmology_outdir = main_outdir / cosmology_name
        cosmology_outdir.mkdir(exist_ok=True)
        self.outdirs_dict[cosmology_name] = self.buildCosmologySubdirsDict(cosmology_outdir)
        for key, dirpath in self.outdirs_dict[cosmology_name].items():
            dirpath.mkdir(exist_ok=True)

    def writeResultsAndDoPlotsForCosmology(self, cosmology_name: str):
        outdirs = self.outdirs_dict[cosmology_name]
        fish_analysis = self.analysis_dict[cosmology_name]
        logger.info(f'Writing analysis results for cosmology {cosmology_name}')
        fish_analysis.writeMarginalizedErrorsToFile(outdir=outdirs['errors'])
        fish_analysis.writeRelativeMarginalizedErrorsToFiles(outdir=outdirs['errors'])
        fish_analysis.saveFishersToDisk(outdir=outdirs['fishers'])
        fish_analysis.writeCorrelationMatricesToFiles(outdir=outdirs['correlations'])
        if self.do_plots:
            fish_plotter = self.plotters_dict[cosmology_name]
            logger.info(f'Doing triangular plots for cosmology {cosmology_name}')
            fish_plotter.makeTriangularPlot(outdir=outdirs['plots'])
            logger.info(f'Doing correlation plots for {cosmology_name} cosmology')
            fish_plotter.makeAllCorrelationPlots(outdir=outdirs['plots'])
            logger.info(f'Writing plot ranges to file')
            fish_plotter.writeParameterPlotRangesToFile(outfile=Path(outdirs['plots']) / 'plot_ranges.json')

    def loadFromResultsDir(self, results_dir: "Union[str, Path]") -> "None":
        results_dir = Path(results_dir)
        cosmo_dirs = [x for x in results_dir.iterdir() if x.is_dir()]
        self.analysis_dict = {}
        for cosmo_dir in cosmo_dirs:
            cosmology_name = cosmo_dir.name
            subdirs_dict = self.buildCosmologySubdirsDict(cosmo_dir)
            fish_analysis = FisherAnalysis(cosmology_name=cosmology_name)
            fish_analysis.loadBaseFishersFromDir(subdirs_dict["fishers"])
            fish_analysis.useBaseFishers()
            errors_file = fsu.get_file_from_dir_with_pattern(subdirs_dict["errors"], 'errors*.csv')
            relative_errors_file = fsu.get_file_from_dir_with_pattern(subdirs_dict["errors"], 'relative_errors*.csv')
            fish_analysis.marginalized_errors = pd.read_csv(errors_file, index_col=0)
            fish_analysis.relative_marginalized_errors = pd.read_csv(relative_errors_file, index_col=0)
            for corr_file in subdirs_dict["correlations"].iterdir():
                fisher_name = corr_file.stem.split('_')[-1]
                fish_analysis.fisher_matrices[fisher_name].correlation = pd.read_csv(corr_file, index_col=0)
            self.analysis_dict[cosmology_name] = fish_analysis
