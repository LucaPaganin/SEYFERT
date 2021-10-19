#!/usr/bin/env python3

from pathlib import Path
import matplotlib

from seyfert.utils.tex_utils import TeXTranslator


all_params = ['Omb', 'Omm', 'h', 'ns', 'sigma8', 'w0', 'wa']
    
COLORMAPS = {
    3: matplotlib.colors.ListedColormap(['darkgreen', 'orange', 'dodgerblue']),
    4: matplotlib.colors.ListedColormap(['darkgreen', 'dodgerblue', 'orange', 'gold']),
}

transl = TeXTranslator()

import importlib
from seyfert.fisher import results_collector

from seyfert.notebook_helpers import contours

importlib.reload(results_collector)
importlib.reload(contours)


scenario = "optm"
plots_dir = Path(f"/Users/lucapaganin/PhotoSpectroXCorr/notebooks/results_thesis/plots/out_test/{scenario}")
plots_dir.mkdir(exist_ok=True, parents=True)

if scenario == "optm":
    fish_analyses_file = Path("/Users/lucapaganin/spectrophoto/production_runs/latest/optimistic/results_collection/w0_wa_CDM/marg_before/fisher_analyses.pickle")
elif scenario == "pess":
    fish_analyses_file = Path("/Users/lucapaganin/spectrophoto/production_runs/latest/pessimistic/results_collection/w0_wa_CDM/marg_before/fisher_analyses.pickle")
else:
    raise KeyError(scenario)

rcoll = results_collector.ResultsCollector()

rcoll.loadFisherAnalysesPickle(fish_analyses_file)

rcoll.replaceHybridFishersWith4BinsEquivalents()


opts = contours.get_default_opts()

case = "phsp"
subcase = "fullcl_vs_2dx3d"
pars_to_plot = ["w0", "wa"]
# pars_to_plot = all_params

opts["ranges_file"] = contours.get_max_ranges_file(scenario=scenario, case=case, subcase=subcase)
opts["cmap"] = COLORMAPS[3]


for n_bins in [4, 12, 24, 40]:
    print(n_bins)
    sel_dict = {"gcph_minus_gcsp": False, "shotnoise_red": False, "n_bins": n_bins}
    an = contours.get_fisher_analysis_for_case_subcase(case, subcase, rcoll, sel_dict=sel_dict)
    an.evaluateMarginalizedErrors()
    contours.build_plotter_and_do_plot(analysis=an, pars_to_plot=pars_to_plot, plots_dir=plots_dir, **opts)
