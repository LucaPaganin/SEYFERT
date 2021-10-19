from typing import List
import matplotlib.pyplot as plt
import pandas as pd

from seyfert.utils.tex_utils import TeXTranslator

transl = TeXTranslator()


def single_plot_fom_gain(diff: "pd.DataFrame", minuend_str: "str",
                         shotnoise_red: "bool", ax, label=None, add_ylabel=True, **kwargs):
    fom_gain = diff.xs(minuend_str, level=0).xs(shotnoise_red, level=0)['FoM']

    if label is None:
        label = "$%s$" % transl.toTeX(minuend_str)

    ax.plot(fom_gain.index, fom_gain.values, label=label, **kwargs)
    ax.set_xlabel(r"$\mathrm{GC_{sp}}$ bins")
    ax.set_xticks(fom_gain.index)
    if add_ylabel:
        ax.set_ylabel(r"FoM % gain")

    return fom_gain, ax


def comparison_plot_fom_gain(diff: "pd.DataFrame", ref: "str", minuends: "List[str]",
                             logscale=True, legend_kwargs=None, legend=True, **kwargs):
    if logscale:
        fig, axs = plt.subplots(figsize=(13, 6), nrows=1, ncols=2, sharey=True)
        plt.subplots_adjust(wspace=4e-2)
        plt.yscale('log')
    else:
        fig, axs = plt.subplots(figsize=(13, 6), nrows=1, ncols=2)
        plt.subplots_adjust(wspace=2e-1)

    for minuend_str in minuends:
        axs[0].set_title(r"$\mathrm{GC_{sp}}$ shotnoise normal", fontsize=16)
        single_plot_fom_gain(diff, minuend_str=minuend_str, shotnoise_red=False, ax=axs[0],
                             **kwargs)
        axs[1].set_title(r"$\mathrm{GC_{sp}}$ shotnoise reduced", fontsize=16)
        single_plot_fom_gain(diff, minuend_str=minuend_str, shotnoise_red=True, ax=axs[1],
                             add_ylabel=False, **kwargs)

    if legend:
        legend_kwds = dict(bbox_to_anchor=[1.0, 0.610])
        if isinstance(legend_kwargs, dict):
            legend_kwds.update(legend_kwargs)
        plt.legend(**legend_kwds)

    plt.suptitle(r"FoM percentage gain relative to $%s$" % transl.toTeX(ref), fontsize=22, va='bottom')

    return fig, axs
