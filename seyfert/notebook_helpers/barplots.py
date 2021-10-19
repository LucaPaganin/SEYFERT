from typing import Dict, List
import matplotlib.pyplot as plt
import json
import pandas as pd

from seyfert.utils.tex_utils import TeXTranslator
from seyfert.utils import filesystem_utils as fsu

transl = TeXTranslator()


def get_bar_plots_defs():
    with open(fsu.plots_aux_files_dir() / "bar_plots_defs.json") as jsf:
        data = json.load(jsf)

    return data


class ErrorBarPlotter:
    df: "pd.DataFrame"
    opts: "Dict"

    def __init__(self, df: "pd.DataFrame"):
        self.df = df
        self.opts = {
            'figure': {
                'side_x_cm': 20,
                'side_y_cm': 14,
                'dpi': 100
            },
            'subplots': {
                'hspace': 0.075,
                'wspace': 0.075,
            },
            'axes': {
                'label_size': 12
            },
            'ticks': {
                'label_size': 12
            },
            'colormap': 'rainbow',
            'legend': {
                'loc': 'upper left',
                'bbox_to_anchor': (1.000, 1.040),
                'title': None,
                'fontsize': 10,
                'title_fontsize': 12,
                'shadow': True,
                'fancybox': False
            },
            'text_box': {
                'coords': (0.580, 0.575),
                'fontsize': 10
            }
        }

        self.validateDataframe()

    def validateDataframe(self):
        self.df = self.df.set_index(["fisher", "n_sp_bins"])

    @property
    def figure_opts(self) -> "Dict":
        return self.opts['figure']

    @property
    def subplots_opts(self) -> "Dict":
        return self.opts['subplots']

    @property
    def axes_opts(self) -> "Dict":
        return self.opts['axes']

    @property
    def ticks_opts(self) -> "Dict":
        return self.opts['ticks']

    @property
    def legend_opts(self):
        return self.opts['legend']

    @property
    def text_box_opts(self) -> "Dict":
        return self.opts['text_box']

    @property
    def legend_title(self):
        return self.legend_opts['title']

    @legend_title.setter
    def legend_title(self, value: "str"):
        if isinstance(value, str):
            self.legend_opts['title'] = value
        else:
            raise TypeError(f"Invalid type {type(value)} for legend title, must be str")

    def sliceResultsForFishersAndNbins(self, fishers: "List[str]", n_bins: "int"):
        sub_df = self.df.loc[fishers].xs(n_bins, level=1)
        sub_df['FoM'] /= sub_df['FoM'].max()

        return sub_df

    def doSingleErrorBarPlot(self, fishers: "List[str]", n_bins: "int", ax: "plt.Axes" = None,
                             do_legend=True, usetex=True) -> "plt.Axes":

        df = self.sliceResultsForFishersAndNbins(fishers, n_bins)

        ax = df.T.plot(kind='barh', ax=ax, colormap=self.opts['colormap'], legend=False)
        ax.set_xscale('log')
        ax.set_xlabel(r"$\sigma_\theta / \theta_{\rm fid}$", fontsize=self.axes_opts['label_size'])
        ax.tick_params(labelsize=self.ticks_opts['label_size'])

        if do_legend:
            ax.legend(**{key: value for key, value in self.opts['legend'].items()})
            transl.translateLegendToTeX(ax.get_legend())

        ax.text(*self.opts['text_box']['coords'], r"$\mathrm{GC_{sp}}(C_{\ell})$ bins: $%s$" % n_bins,
                transform=ax.transAxes, fontsize=self.opts['text_box']['fontsize'])

        if usetex:
            latex_ytick_labels = []
            for label in ax.get_yticklabels():
                text = label.get_text()
                if text == 'FoM':
                    tex_label = r"$\mathrm{FoM}/\mathrm{FoM_{max}}$"
                else:
                    tex_label = r"$%s$" % transl.PhysParNameToTeX(text)

                latex_ytick_labels.append(tex_label)

            ax.set_yticklabels(latex_ytick_labels)

        return ax

    def doPairErrorBarPlot(self, fishers: "List[str]", n_bins1: "int", n_bins2: "int", legend=True):
        side_x_inches = self.figure_opts['side_x_cm'] / 2.54
        side_y_inches = self.figure_opts['side_y_cm'] / 2.54

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(side_x_inches, side_y_inches), sharex=True,
                                dpi=self.figure_opts['dpi'])
        plt.subplots_adjust(hspace=self.subplots_opts['hspace'], wspace=self.subplots_opts['wspace'])
        axs = axs.ravel()

        for ax, n_bins in zip(axs, [n_bins1, n_bins2]):
            do_legend = legend and n_bins == n_bins1
            self.doSingleErrorBarPlot(fishers, n_bins, ax=ax, do_legend=do_legend)

        return fig, axs

    def doGridErrorBarPlot(self, fishers: "List[str]"):
        side_x_inches = self.figure_opts['side_x_cm'] / 2.54
        side_y_inches = self.figure_opts['side_y_cm'] / 2.54

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(side_x_inches, side_y_inches), sharex=True,
                                dpi=self.figure_opts['dpi'])
        plt.subplots_adjust(hspace=self.subplots_opts['hspace'], wspace=self.subplots_opts['wspace'])
        axs = axs.ravel()

        for ax, n_bins in zip(axs, [4, 12, 24, 40]):
            do_legend = n_bins == 12
            self.doSingleErrorBarPlot(fishers, n_bins, ax=ax, do_legend=do_legend)

        axs[1].tick_params(labelleft=False)
        axs[3].tick_params(labelleft=False)

        return fig, axs

    def setSinglePlotOpts(self):
        self.axes_opts['label_size'] = 20
        self.ticks_opts['label_size'] = 20
        self.legend_opts.update({
            'fontsize': 20,
            'title_fontsize': 22,
            'bbox_to_anchor': (1.000, 1.025)
        })
        self.text_box_opts.update({
            'coords': (0.620, 0.600),
            'fontsize': 14
        })
