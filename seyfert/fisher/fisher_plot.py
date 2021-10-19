import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, List, Iterable

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.patches import Ellipse, Patch
from matplotlib import ticker

from seyfert.fisher.fisher_analysis import FisherAnalysis
from seyfert.fisher.fisher_matrix import FisherMatrix
from seyfert.utils import converters as cnv, tex_utils as txu, filesystem_utils as fsu
from seyfert import plot_utils

logger = logging.getLogger(__name__)


class FisherPlotConfig:
    config_dict: "Dict"
    gauss_cl_alpha_dict: "Dict[float, float]"
    ellipse_cl_alpha_dict: "Dict[float, float]"
    plot_ranges: "Dict[str, Tuple[float, float]]"

    def __init__(self,
                 config_file: "Union[str, Path]" = None,
                 n_cosmo_pars: "int" = None):
        self.n_cosmo_pars = n_cosmo_pars
        self.gauss_cl_alpha_dict = None
        self.ellipse_cl_alpha_dict = None
        self.plot_ranges = None
        self.config_dict = None
        self.cmap = None
        self.tex_transl = txu.TeXTranslator()
        if not isinstance(self.n_cosmo_pars, int):
            raise TypeError(f'expected int as n_cosmo_pars, got {type(self.n_cosmo_pars)}')

        if isinstance(config_file, (str, Path)):
            with open(config_file, 'r') as jsf:
                config_dict = json.load(jsf)
                self.loadFromConfigDict(config_dict["plot_settings"])

    @property
    def use_tex(self):
        return self.config_dict['usetex']

    @property
    def dims_dict(self):
        return self.config_dict['dimensions']

    @property
    def fig_size_x_cm(self):
        return self.dims_dict['fig_size_x_cm']

    @property
    def fig_size_y_cm(self):
        return self.dims_dict['fig_size_y_cm']

    @property
    def fig_size_x_inches(self):
        return cnv.CmToInches(self.fig_size_x_cm)

    @property
    def fig_size_y_inches(self):
        return cnv.CmToInches(self.fig_size_y_cm)

    @property
    def horizontal_spacing(self):
        return self.dims_dict['subplots_spacing']['horizontal']

    @property
    def vertical_spacing(self):
        return self.dims_dict['subplots_spacing']['vertical']

    @property
    def graphics_dict(self):
        return self.config_dict['graphics']

    @property
    def dpi(self):
        return self.graphics_dict['dpi']

    @property
    def color_map_name(self):
        return self.graphics_dict['color_map']

    @property
    def axes_dict(self):
        return self.config_dict['axes']

    @property
    def axes_label_size(self):
        return self.axes_dict['label_size']

    @property
    def axes_x_label_pad(self):
        return self.axes_dict['x_label_pad']

    @property
    def axes_y_label_pad(self):
        return self.axes_dict['y_label_pad']

    @property
    def axes_borders_width(self):
        return self.axes_dict['borders_width']

    @property
    def axes_lines_width(self):
        return self.axes_dict['lines_width']

    @property
    def ticks_width(self):
        return self.axes_dict['ticks']['width']

    @property
    def ticks_length(self):
        return self.axes_dict['ticks']['length']

    @property
    def ticks_label_size(self):
        return self.axes_dict['ticks']['label_size']

    @property
    def ticks_label_num_digits(self):
        return self.axes_dict['ticks']['digits']

    @property
    def ticks_number(self):
        return self.axes_dict['ticks']['number']

    @property
    def x_tick_labels_rot(self):
        return self.axes_dict['ticks']['x_tick_labels_rot']

    @property
    def y_tick_labels_rot(self):
        return self.axes_dict['ticks']['y_tick_labels_rot']

    @property
    def legend_dict(self):
        return self.config_dict['legend']

    @property
    def legend_loc(self):
        return self.legend_dict['loc']

    @property
    def legend_bbox_to_anchor(self):
        return self.legend_dict['bbox_to_anchor']

    @property
    def legend_fontsize(self):
        return self.legend_dict['fontsize']

    @property
    def legend_title_fontsize(self):
        return self.legend_dict['title_fontsize']

    @property
    def text_box_dict(self):
        return self.config_dict["text_box"]

    def loadFromConfigDict(self, config_dict: "Dict") -> "None":
        self.config_dict = config_dict
        self.setGraphicsProperties()

    def setGraphicsProperties(self) -> "None":
        self.gauss_cl_alpha_dict = {
            float(cl): alpha for cl, alpha in
            self.graphics_dict["cl_alpha_dict"]["gauss"].items()
        }
        self.ellipse_cl_alpha_dict = {
            float(cl): alpha for cl, alpha in
            self.graphics_dict["cl_alpha_dict"]["ellipse"].items()
        }

        self.cmap = matplotlib.cm.get_cmap(self.color_map_name)

    def loadParameterPlotRangesFromFile(self, ranges_file: "Union[str, Path]") -> "None":
        with open(ranges_file, 'r') as jsf:
            ranges_dict = json.load(jsf)
        self.plot_ranges = ranges_dict

    def getTeXName(self, name: "str", use_aliases=False) -> "str":
        return self.tex_transl.translateToTeX(name, use_aliases=use_aliases)


class FisherPlotter:
    analysis: "FisherAnalysis"
    figure: "plt.Figure"
    axes: "np.ndarray"
    config: "FisherPlotConfig"
    pars_to_plot: "List[str]"

    def __init__(self, pars_to_plot: "Iterable[str]" = None, config: "FisherPlotConfig" = None,
                 fisher_analysis: "FisherAnalysis" = None):
        self.analysis = fisher_analysis
        self.config = config
        self.pars_to_plot = pars_to_plot
        self.figure = None
        self.axes = None
        self.translator = txu.TeXTranslator()
        self._colors_arr = None
        self.params_ticks_fmts = {}

    @property
    def fisher_matrices(self) -> "Dict[str, FisherMatrix]":
        return self.analysis.fisher_matrices

    @property
    def fishers_order(self) -> "List[str]":
        return self.analysis.fishers_order

    @property
    def n_cosmo_pars(self) -> "int":
        return len(self.pars_to_plot)

    @property
    def plot_ranges(self) -> "Dict[str, Tuple[float, float]]":
        return self.config.plot_ranges

    def setUpDefault(self) -> "None":
        self.setParamsToPlot(pars_to_plot='all')
        self.setPlotConfig(use_default=True)
        self.setParametersPlotRanges()

    def setParamsToPlot(self, pars_to_plot: "Union[str, Iterable[str]]" = 'all'):
        logger.info("Setting parameters to plot")
        if self.analysis is None:
            raise TypeError("Cannot set params without Fisher analysis")
        if pars_to_plot == 'all':
            self.pars_to_plot = sorted(list(self.analysis.cosmo_pars_fiducials.keys()))
        else:
            self.pars_to_plot = list(pars_to_plot)

        logger.info(f"Plotting parameters: {', '.join(self.pars_to_plot)}")

    def setParametersPlotRanges(self, ranges_file: "Union[str, Path]" = None) -> "None":
        logger.info("Setting parameters plot ranges")
        self.config.plot_ranges = {}
        if ranges_file is not None:
            ranges_file = Path(ranges_file)
            if not ranges_file.exists():
                raise FileNotFoundError(ranges_file)
            logger.info(f"Loading ranges from {ranges_file}")
            with open(ranges_file, 'r') as jsf:
                self.config.plot_ranges = json.load(jsf)
        else:
            logger.info(f"Computing plot ranges on the fly")
            for name in self.pars_to_plot:
                self.config.plot_ranges[name] = self.computePlotRangeForParameter(name)

    def setPlotConfig(self, config_file: "Union[str, Path]" = None, use_default=False,
                      config_dict: "Dict" = None) -> "None":
        logger.info("Setting plot configuration")
        if use_default:
            config_file = fsu.config_files_dir() / 'results_config.json'
            logger.info(f"Using default config {config_file}")
        else:
            if config_file is None:
                raise ValueError('config file path needed when not using default. If you want to use '
                                 'the default one, call this function as "setPlotConfig(use_default=True)"')

        self.config = FisherPlotConfig(n_cosmo_pars=self.n_cosmo_pars, config_file=config_file)
        if config_dict is not None:
            self.config.config_dict.update(config_dict)

        self.params_ticks_fmts = {name: f"%.{self.config.ticks_label_num_digits}g" for name in self.pars_to_plot}

    def getColorForFisher(self, name: "str"):
        idx = self.fishers_order.index(name)
        if self._colors_arr is None:
            self._colors_arr = np.linspace(0, 1, len(self.fishers_order), endpoint=True)

        return self.config.cmap(self._colors_arr[idx])

    def makeTriangularPlot(self, outdir: "Union[str, Path]" = None, order: "List[str]" = None, legend=True):
        plt.rcParams['text.usetex'] = self.config.use_tex
        fig, axes = plt.subplots(nrows=self.n_cosmo_pars, ncols=self.n_cosmo_pars, dpi=self.config.dpi,
                                 figsize=(self.config.fig_size_x_inches, self.config.fig_size_y_inches),
                                 sharex=False, sharey=False)

        self.figure = fig
        self.axes = axes
        self.figure.subplots_adjust(hspace=self.config.horizontal_spacing,
                                    wspace=self.config.vertical_spacing)

        self.drawContours(order=order)

        if legend:
            self.addPlotLegend(order=order)
        self.setFigureTitle()
        if outdir is not None:
            outfile_name = f'contour_{self.analysis.name}.pdf'
            self.figure.savefig(Path(outdir) / outfile_name)
            plt.close(self.figure)

        return self.figure, self.axes

    def drawContours(self, order: "List[str]" = None):
        if order is None:
            order = self.fishers_order
        for i in range(self.axes.shape[0]):
            for j in range(self.axes.shape[1]):
                name_x, name_y = self.pars_to_plot[j], self.pars_to_plot[i]
                if i < j:
                    self.axes[i, j].axis('off')
                elif i == j:
                    name = self.pars_to_plot[i]
                    for obs in order:
                        self.drawGaussianForParameterAndObs(name, obs, self.axes[i, i])
                elif i > j:
                    for obs in order:
                        self.drawEllipsesForParametersAndObs(name_x, name_y, obs, self.axes[i, j])
                self.setPlotProperties(i, j)

    def makeAllCorrelationPlots(self, outdir: "Union[str, Path]" = None, close_fig: "bool" = True) -> "None":
        for name, correlation_matrix in self.analysis.correlation_matrices.items():
            fig, ax = self.makeSingleCorrelationPlot(name, correlation_matrix)
            if outdir is not None:
                outfile_name = f'correlation_{self.analysis.name}_{name}.pdf'
                outfile_path = Path(outdir) / outfile_name
                fig.savefig(outfile_path)
                if close_fig:
                    plt.close(fig)

    def makeSingleCorrelationPlot(self, name: "str") -> "Tuple[plt.Figure, plt.Axes]":
        fig = plt.figure(figsize=(6.0, 6.0), dpi=150)
        corr = self.fisher_matrices[name].correlation

        ax = plot_utils.heatmap(corr, usetex=True, vmin=-1.0, vmax=1.0,
                                cmap='bwr', alpha=0.9, square=True, cbar_kws={'shrink': 0.825})

        ax.set_title(rf'${self.translator.ProbeNameToTeX(name)}$')

        return fig, ax

    def setPlotProperties(self, i: int, j: "int") -> "None":
        self.setPlotTicks(i, j)
        self.setPlotAxesLabels(i, j)
        self.setPlotRange(i, j)
        for side in self.axes[i, j].spines.keys():
            self.axes[i, j].spines[side].set_linewidth(self.config.axes_borders_width)

    def setPlotAxesLabels(self, i: "int", j: "int") -> "None":
        name_x, name_y = self.pars_to_plot[j], self.pars_to_plot[i]
        if i == j:
            self.axes[i, i].set_yticks([0.0, 1.0])
            self.axes[i, i].yaxis.tick_right()
            self.axes[i, i].set_ylabel(r'$P/P_{\rm max}$', fontsize=self.config.axes_label_size)
            self.axes[i, i].yaxis.set_label_position('right')
        if i == self.axes.shape[0] - 1:
            self.axes[i, j].set_xlabel(rf'${self.translator.CosmoParNameToTeX(name_x)}$',
                                       labelpad=self.config.axes_x_label_pad,
                                       fontsize=self.config.axes_label_size)
        if j == 0 and i > 0:
            self.axes[i, j].set_ylabel(rf'${self.translator.CosmoParNameToTeX(name_y)}$', ha='center', va='center',
                                       labelpad=self.config.axes_y_label_pad, rotation=0,
                                       fontsize=self.config.axes_label_size)

    def setPlotTicks(self, i: "int", j: "int") -> "None":
        self.setXTicks(i, j)
        self.setYTicks(i, j)

        label_bottom = i == self.axes.shape[0] - 1
        label_left = (j == 0 and i > 0)
        self.axes[i, j].tick_params(labelsize=self.config.ticks_label_size,
                                    width=self.config.ticks_width, length=self.config.ticks_length,
                                    labelleft=label_left, labelbottom=label_bottom)

    def setXTicks(self, i: "int", j: "int"):
        name_x = self.pars_to_plot[j]
        ticks_number = int(self.config.ticks_number)
        num_digits = self.config.ticks_label_num_digits
        x_min, x_max = self.plot_ranges[name_x]
        x_ticks = np.linspace(x_min, x_max, num=ticks_number, endpoint=True)
        self.axes[i, j].set_xticks(x_ticks)
        self.axes[i, j].xaxis.set_major_formatter(ticker.FormatStrFormatter(self.params_ticks_fmts[name_x]))
        x_tick_labels = self.axes[i, j].xaxis.get_ticklabels()
        # global properties
        for x_tick_label in x_tick_labels:
            x_tick_label.set_rotation(self.config.x_tick_labels_rot)
        # borders and bulk
        x_tick_labels[0].set_horizontalalignment('left')
        x_tick_labels[-1].set_horizontalalignment('right')
        """
        for x_tick_label in x_tick_labels[1:-1]:
            x_tick_label.set_horizontalalignment('center')
        """

    def setYTicks(self, i: "int", j: "int"):
        name_y = self.pars_to_plot[i]
        ticks_number = int(self.config.ticks_number)
        num_digits = self.config.ticks_label_num_digits
        if i == j:
            self.axes[i, i].set_yticks([0.0, 1.0])
            self.axes[i, i].yaxis.tick_right()
        else:
            y_min, y_max = self.plot_ranges[name_y]
            y_ticks = np.linspace(y_min, y_max, num=ticks_number, endpoint=True)
            self.axes[i, j].yaxis.set_major_formatter(ticker.FormatStrFormatter(self.params_ticks_fmts[name_y]))
            self.axes[i, j].set_yticks(y_ticks)
            y_tick_labels = self.axes[i, j].yaxis.get_ticklabels()
            # global properties
            for y_tick_label in y_tick_labels:
                y_tick_label.set_rotation(self.config.y_tick_labels_rot)
            # borders and bulk
            y_tick_labels[0].set_verticalalignment('bottom')
            y_tick_labels[-1].set_verticalalignment('top')
            """
            for y_tick_label in y_tick_labels[1:-1]:
                y_tick_label.set_verticalalignment('center')
            """

    def setPlotRange(self, i: int, j: "int") -> "None":
        name_x = self.pars_to_plot[j]
        name_y = self.pars_to_plot[i]
        self.axes[i, j].set_xlim(self.plot_ranges[name_x])
        if i != j:
            self.axes[i, j].set_ylim(self.plot_ranges[name_y])

    def getLegendPatch(self, name: "str") -> "Patch":
        tex_label = rf"${self.config.getTeXName(name)}$"
        color = self.getColorForFisher(name)

        return Patch(color=color, label=tex_label)

    def addPlotLegend(self, order: "List[str]" = None, add_handles: "List" = None, **kwargs) -> "None":
        legend_order = order if order is not None else self.fishers_order
        legend_handles = [self.getLegendPatch(name) for name in legend_order]
        if isinstance(add_handles, list):
            legend_handles = add_handles + legend_handles
        kwds = dict(fontsize=self.config.legend_fontsize, loc=self.config.legend_loc,
                    title=self.analysis.name,
                    title_fontsize=self.config.legend_title_fontsize,
                    bbox_to_anchor=self.config.legend_bbox_to_anchor,
                    bbox_transform=self.figure.transFigure)
        kwds.update(kwargs)

        self.figure.legend(handles=legend_handles, **kwds)

    def addInfoTextBox(self, text: "str", x=None, y=None, **kwargs):
        x_leg, y_leg = self.config.legend_bbox_to_anchor
        if x is None:
            x = x_leg + float(self.config.text_box_dict["dx_leg"])
        if y is None:
            y = y_leg + float(self.config.text_box_dict["dy_leg"])

        kwds = {
            'fontsize': self.config.legend_fontsize,
            'transform': self.figure.transFigure
        }
        kwds.update(kwargs)

        plot_utils.misc.add_text_box(pl_obj=self.figure, x=x, y=y, text=text, **kwargs)

    def setFigureTitle(self) -> "None":
        pass

    def writeParameterPlotRangesToFile(self, outfile: "Union[str, Path]", overwrite=False) -> "None":
        outfile = Path(outfile)
        if outfile.exists():
            if overwrite:
                with open(outfile, 'w') as jsf:
                    json.dump(self.plot_ranges, jsf, indent=2)
            else:
                logger.info(f'file {outfile} already exists, not overwriting it')
        else:
            with open(outfile, 'w') as jsf:
                json.dump(self.plot_ranges, jsf, indent=2)

    def computePlotRangeForParameter(self, name: "str") -> "Tuple[float, float]":
        lower_bounds = []
        upper_bounds = []
        for obs in self.fishers_order:
            x_min, x_max = self.analysis.computeParameterRangeForObsAtCL(name, obs, 0.99)
            lower_bounds.append(x_min)
            upper_bounds.append(x_max)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        return lower_bounds.mean(), upper_bounds.mean()

    def drawGaussianForParameterAndObs(self, name: "str", fisher_name: "str", ax: "plt.Axes") -> "None":
        for CL, alpha in self.config.gauss_cl_alpha_dict.items():
            x_min = self.plot_ranges[name][0]
            x_max = self.plot_ranges[name][1]
            gaus_dict = self.analysis.computeGaussianInfoDictForObsAndCL(name, fisher_name, CL, x_min, x_max)
            color = self.getColorForFisher(fisher_name)

            ax.plot(gaus_dict['x'], gaus_dict['y'], color=color, linewidth=self.config.axes_lines_width)
            ax.fill_between(gaus_dict['x'], gaus_dict['y'], where=gaus_dict['cl_mask'], linewidth=0.0,
                            color=color, alpha=alpha)

    def drawEllipsesForParametersAndObs(self, name_x: "str", name_y: "str", fisher_name: "str",
                                        ax: "plt.Axes") -> "None":
        for CL, alpha in self.config.ellipse_cl_alpha_dict.items():
            color = self.getColorForFisher(fisher_name)
            ellipse_info_dict = self.analysis.computeEllipseInfoDictForObsAndCL(name_x, name_y, fisher_name, CL)

            fill_ellipse_kwds = ellipse_info_dict.copy()
            fill_ellipse_kwds.update(dict(facecolor=color, edgecolor=color, fill=True, alpha=alpha, linewidth=0.0))

            cont_ellipse_kwds = ellipse_info_dict.copy()
            cont_ellipse_kwds.update(dict(facecolor=color, edgecolor=color, fill=False,
                                          linewidth=self.config.axes_lines_width))

            ax.add_artist(Ellipse(**fill_ellipse_kwds))
            ax.add_artist(Ellipse(**cont_ellipse_kwds))
