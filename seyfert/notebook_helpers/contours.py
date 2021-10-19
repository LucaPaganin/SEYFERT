from typing import Iterable, List, TYPE_CHECKING, Dict, Tuple
from pathlib import Path
import json
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

from seyfert.utils import filesystem_utils as fsu
from seyfert.fisher.fisher_plot import FisherPlotter
from seyfert.fisher.fisher_analysis import FisherAnalysis
from seyfert.plot_utils import misc

if TYPE_CHECKING:
    from seyfert.fisher.results_collector import ResultsCollector
    import numpy as np

with open(fsu.plots_aux_files_dir() / "contour_plots_defs.json") as jsf:
    PLOTS_FISHERS_MAP = json.load(jsf)

tex_dict = {
    'phsp': r"\mathrm{GC_{ph}} \times \mathrm{GC_{sp}}",
    'wlsp': r"\mathrm{WL} \times \mathrm{GC_{sp}}",
    '6x2pt': r"6\times 2 \mathrm{pt}",
    'fullcl': r"\mathrm{full}\, C(\ell)",
    '2dx3d': r"2\mathrm{D}\times 3\mathrm{D}",
    'cl': r"C_{\ell}"
}

tex_dict['fullcl_vs_2dx3d'] = r"%s\, \mathrm{vs}\, %s" % (tex_dict['fullcl'], tex_dict['2dx3d'])

root_dir = Path("/Users/lucapaganin/PhotoSpectroXCorr/notebooks/results_thesis/plots/")


def get_fisher_analysis_for_case_subcase(case, subcase, rcoll: "ResultsCollector", sel_dict: "Dict"):
    add_metadata = dict(case=case, subcase=subcase)

    return get_fisher_analysis_for_fishers(rcoll, sel_dict, sel_fishers=PLOTS_FISHERS_MAP[case][subcase],
                                           add_metadata=add_metadata)


def get_fisher_analysis_for_fishers(rcoll: "ResultsCollector", sel_dict: "Dict", sel_fishers: "List[str]",
                                    add_metadata: "Dict", eval_errs=True) -> "FisherAnalysis":
    an_df = rcoll.getFisherAnalysesSubDf(sel_dict=sel_dict, sel_fishers=sel_fishers)
    if len(an_df) != 1:
        raise Exception("Invalid number of selected fisher analyses. Change your selection dict until only one "
                        "analysis is left.")

    run_id = an_df['run_id'][0]
    an: "FisherAnalysis" = an_df['analysis'][0]
    an.metadata['run_id'] = run_id
    an.metadata['scenario'] = run_id.split("_")[0]
    an.name = run_id
    an.metadata.update(sel_dict)
    an.metadata.update(add_metadata)

    if eval_errs:
        an.evaluateMarginalizedErrors()
        an.evaluateRelativeMarginalizedErrors()

    return an


def build_plotter_and_do_plot(analysis: "FisherAnalysis", pars_to_plot, plots_dir: "Path",
                              **kwargs) -> "Tuple[FisherPlotter, plt.Figure, np.ndarray[plt.Axes]]":
    try:
        config_file = kwargs['config_file']
    except KeyError:
        config_file = None

    plotter = build_plotter(analysis, pars_to_plot, config_file=config_file)

    fig, axes = do_tri_plot(plotter, plots_dir, **kwargs)

    return plotter, fig, axes


def build_plotter(analysis: "FisherAnalysis", pars_to_plot: "Iterable[str]", config_file=None, config_dict=None,
                  params_ticks_fmts=None):

    plotter = FisherPlotter(fisher_analysis=analysis)
    plotter.setParamsToPlot(pars_to_plot=pars_to_plot)

    if config_dict is not None:
        plotter.setPlotConfig(config_dict=config_dict)
    else:
        if config_file is None:
            config_file = get_config_file(pars_to_plot)

        plotter.setPlotConfig(config_file=config_file)

    plotter.setParametersPlotRanges()

    if isinstance(params_ticks_fmts, dict):
        plotter.params_ticks_fmts.update(params_ticks_fmts)

    return plotter


def do_tri_plot(plotter: "FisherPlotter", plots_dir: "Path", print_opts=False, **kwargs):
    kwds = defaultdict(lambda: None)
    # defaults
    kwds.update({
        'save': True, 'closefig': False, 'usetex': False,
        'write_ranges': True, 'ranges_file': None, 'cmap': None,
        'legend_kwargs': {}, 'text_box_kwargs': {}
    })
    # update from kwargs
    kwds.update({key: value for key, value in kwargs.items() if key in kwds})

    if print_opts:
        print("Acceptable options:")
        for key in kwds.items():
            print(key)

    metadata = plotter.analysis.metadata
    scenario = metadata['scenario']
    case = metadata['case']
    subcase = metadata['subcase']
    n_bins = metadata['n_bins']

    plotter.setParametersPlotRanges(ranges_file=kwds['ranges_file'])

    cmap = kwds['cmap']
    if isinstance(cmap, matplotlib.colors.Colormap):
        plotter.config.cmap = cmap
        if len(cmap.colors) < len(plotter.fisher_matrices):
            plotter.config.cmap = matplotlib.cm.get_cmap('rainbow')
    elif isinstance(cmap, str):
        try:
            plotter.config.cmap = matplotlib.cm.get_cmap(cmap)
        except ValueError:
            print(f"unrecognized colormap {cmap}, defaulting to {plotter.config.color_map_name}")

    plotter.config.config_dict['usetex'] = kwds['usetex']

    fig, axes = plotter.makeTriangularPlot(legend=False)

    # Add legend
    legend_title = r'$%s$: $%s$' % (tex_dict[case], tex_dict[subcase])
    legend_kwds = dict(title=legend_title)
    legend_kwds.update(kwds['legend_kwargs'])

    add_handles = []
    # add_handles = [matplotlib.patches.Patch(color='none', label=r"$\mathrm{GC_{sp}}\, bins:\, %s$" % (nbins))]
    plotter.addPlotLegend(add_handles=add_handles, **legend_kwds)

    # Add text box
    x_leg, y_leg = plotter.config.legend_dict['bbox_to_anchor']
    x_text = x_leg + float(plotter.config.text_box_dict["dx_leg"])
    y_text = y_leg + float(plotter.config.text_box_dict["dy_leg"])
    fontsize = plotter.config.legend_fontsize

    misc.add_scenario_n_bins_text_box(pl_obj=fig, x=x_text, y=y_text, scenario=scenario, n_bins=n_bins,
                                      fontsize=fontsize)

    ranges_outfile = root_dir / f"configs/ranges/{scenario}/{case}_{subcase}_{n_bins}_bins_ranges.json"
    plotter.writeParameterPlotRangesToFile(ranges_outfile, overwrite=True)
    if kwds['save']:
        outdir = plots_dir / f"{case}/{subcase}"
        outdir.mkdir(parents=True, exist_ok=True)
        outfile_path = outdir / f"contour_{'_'.join(sorted(plotter.pars_to_plot))}_{n_bins}_bins.pdf"
        fig.savefig(outfile_path, bbox_inches='tight')
    if kwds['closefig']:
        plt.close(fig)

    return fig, axes


def get_default_opts():
    return {
        'ranges_file': None,
        'cmap': None,
        'save': True,
        'closefig': True,
        'write_ranges': True,
        'legend_kwargs': dict(),
        'text_box_kwargs': dict(),
        'usetex': True,
        'config_file': None
    }


def get_config_file(pars_to_plot, manual=True):
    if isinstance(pars_to_plot, (tuple, list)):
        num_params = len(pars_to_plot)
    elif pars_to_plot == 'all':
        num_params = 7
    else:
        raise ValueError(f"Unrecognized value {pars_to_plot} for pars_to_plot")

    cfg_name = f"{num_params}params_config.json"

    cfg_dir = root_dir / "configs"
    if manual:
        cfg_dir = cfg_dir / "manually_adjusted"

    return cfg_dir / cfg_name


def get_max_ranges_file(scenario, case, subcase):
    ranges_file = root_dir / f"configs/ranges/{scenario}/max_ranges/max_ranges_{case}_{subcase}.json"

    return ranges_file
