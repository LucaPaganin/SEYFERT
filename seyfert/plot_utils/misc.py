from typing import Tuple, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from seyfert.utils.tex_utils import TeXTranslator
from seyfert.utils import general_utils as gu


translator = TeXTranslator()


def add_scenario_n_bins_text_box(pl_obj: "Union[plt.Figure, plt.Axes]", x: "float", y: "float",
                                 scenario: "str", n_bins: "int" = None, **kwargs):
    scenario = gu.get_scenario_string(scenario)
    text = f"Scenario: {scenario}"
    if n_bins is not None:
        text += "\n"
        text += r"$\mathrm{GC_{sp}}$ bins: $%s$" % n_bins
    add_text_box(pl_obj=pl_obj, x=x, y=y, text=text, **kwargs)


def add_text_box(pl_obj: "Union[plt.Figure, plt.Axes]", x: "float", y: "float", text: "str", **kwargs):
    kwds = {
        "bbox": dict(boxstyle='round, pad=0.5', edgecolor='lightgrey', facecolor='None')
    }
    kwds.update(kwargs)
    pl_obj.text(x, y, text, **kwds)


def set_TeX_xticklabels(ax: "plt.Axes", rotation=0, label_texts=None):
    if label_texts is None:
        label_texts = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticklabels(translator.PhysParsNamesToTeX(label_texts, include_dollar=True), rotation=rotation)


def set_TeX_yticklabels(ax: "plt.Axes", rotation=0, label_texts=None):
    if label_texts is None:
        label_texts = [label.get_text() for label in ax.get_yticklabels()]
    ax.set_yticklabels(translator.PhysParsNamesToTeX(label_texts, include_dollar=True), rotation=rotation)


def heatmap(matrix: "Union[np.ndarray, pd.DataFrame]", title: str = "", invert_yaxis=True, logscale=False,
            usetex=False, xlabelrot=0, ylabelrot=0, **kwargs) -> "plt.Axes":
    if logscale:
        kwargs['norm'] = mcolors.LogNorm()
    ax = sns.heatmap(matrix, **kwargs)
    ax.set_title(title)
    if invert_yaxis:
        ax.axes.invert_yaxis()
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    if isinstance(matrix, pd.DataFrame) and usetex:
        set_TeX_xticklabels(ax, rotation=xlabelrot)
        set_TeX_yticklabels(ax, rotation=ylabelrot)

    return ax


def plot_errs_deltas(deltas: "pd.DataFrame", title="", xlabelrot=0, scenario="None") -> "Tuple[plt.Figure, plt.Axes]":
    npars = len(deltas.columns)
    fig, ax = plt.subplots(figsize=(15, 10))

    have_unmarg_errs = isinstance(deltas.index, pd.MultiIndex)
    marg_deltas = deltas.loc["marg"] if have_unmarg_errs else deltas

    splot = sns.scatterplot(data=marg_deltas.T, alpha=1.0, ax=ax)
    ax.set_title(f"{title}")
    ax.set_xlabel("")
    ax.set_ylabel(r"% difference on $\sigma_i$")
    ax.set_xticklabels([f"${translator.PhysParNameToTeX(name)}$" for name in marg_deltas.columns],
                       rotation=xlabelrot)
    code_legend = ax.get_legend()
    code_legend.legendHandles = code_legend.legendHandles[2:]
    ax.add_artist(code_legend)

    x0 = 1.0000
    y0 = 0.7750
    dx = 0.0750
    dy = 0.0600

    code_legend.set_bbox_to_anchor([x0, y0])

    text = "\n".join([
        f"Flat, scenario: {scenario}"
    ])
    bbox = dict(boxstyle='round, pad=0.5', edgecolor='lightgrey', facecolor='None')
    ax.text(x0 - 1.1*dx, y0 - 2.0*dy, text, transform=ax.figure.transFigure, bbox=bbox, fontsize=16)

    if have_unmarg_errs:
        unmarg_deltas = deltas.loc['unmarg']
        margband = ax.fill_between(range(npars), marg_deltas.min(axis=0).values, marg_deltas.max(axis=0).values,
                                   alpha=0.30, color='lightgrey')
        unmargband = ax.fill_between(range(npars), unmarg_deltas.min(axis=0).values, unmarg_deltas.max(axis=0).values,
                                     alpha=0.45, color='darkgrey')
        band_legend = plt.legend(handles=[mpatches.Patch(color="lightgrey", label="marg"),
                                          mpatches.Patch(color="darkgrey", label="unmarg")],
                                 bbox_transform=fig.transFigure)
        band_legend.set_bbox_to_anchor([x0, y0 - 4.0*dy])
        ax.add_artist(band_legend)

    return fig, ax


def plot_series(series: "pd.Series", usetex=False, kind='scatter', ax=None, **kwargs) -> "Tuple[plt.Figure, plt.Axes]":
    if ax is None:
        ax = plt.gca()
    if kind == 'scatter':
        ax.scatter(x=series.index, y=series.values, **kwargs)
    elif kind == 'line':
        ax.plot(series, **kwargs)
    else:
        raise ValueError(f'Invalid plot kind {kind}, must be "scatter" or "line"')

    if usetex:
        set_TeX_xticklabels(ax, label_texts=series.index)

    return ax
