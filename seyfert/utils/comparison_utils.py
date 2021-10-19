import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seyfert.numeric.general import pad
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ClDiff:
    acronyms = {'PAD': 'Percentile Absolute Difference'}
    comp_df: pd.DataFrame
    cl1: np.ndarray
    cl2: np.ndarray
    cl_pad: np.ndarray
    cl_type: str
    metrics: Dict[str, np.ndarray]
    l_bin_centers: np.ndarray
    ref: np.ndarray
    i_max_ref: int
    j_max_ref: int

    def __init__(self, cl_type: "str",
                 cl1: "np.ndarray" = None, cl2: "np.ndarray" = None,
                 l_values: "np.ndarray" = None):
        self.cl_type = cl_type
        self.cl1 = cl1
        self.cl2 = cl2
        self.ref = None
        self.comp_df = None
        self.cl_pad = None
        self.l_mean_pad = None
        self.i_range = None
        self.j_range = None
        self.metrics = None
        self.l_bin_centers = l_values
        self.i_max_ref = None
        self.j_max_ref = None
        if self.cl1 is not None and self.cl2 is not None:
            self.evaluateMetrics()

    def evaluateMetrics(self):
        self.ref = (self.cl1 + self.cl2) / 2
        self.cl_pad = pad(self.cl1, self.cl2)
        self.i_range = self.cl_pad.shape[1]
        self.j_range = self.cl_pad.shape[2]
        self.metrics = {
            'mean_PAD': np.mean(self.cl_pad, axis=0),
            'sigma_PAD': np.std(self.cl_pad, axis=0)
        }
        if self.i_range == self.j_range:
            idxs = np.indices((self.i_range, self.j_range))
            for m in self.metrics:
                self.metrics[m] = np.where(idxs[0] <= idxs[1], self.metrics[m], np.nan)
        self.evaluateComparisonTable()

    def evaluateComparisonTable(self):
        ref_cl_value = np.max(self.ref)
        idxs_max_ref = np.unravel_index(np.argmax(self.ref), self.ref.shape)
        self.i_max_ref, self.j_max_ref = idxs_max_ref[1:]
        mean_pad = self.metrics['mean_PAD']
        mpad_idxs = np.indices(mean_pad.shape)

        mean_pad = mean_pad.ravel()
        mask = ~np.isnan(mean_pad)
        i_s = mpad_idxs[0].ravel()[mask]
        j_s = mpad_idxs[1].ravel()[mask]
        cl_mean_ratio = np.mean(self.ref / ref_cl_value, axis=0).ravel()[mask]
        mean_pad = mean_pad[mask]

        self.comp_df = pd.DataFrame.from_dict({
            'mean_pad': mean_pad,
            'i': i_s,
            'j': j_s,
            'cl_order_of_mag': cl_mean_ratio,
            '|i-j|': np.abs(i_s - j_s),
        })

    def cutOffMetrics(self, cutoff: float):
        for kind in self.metrics:
            self.metrics[kind] = np.where(self.metrics[kind] < cutoff, self.metrics[kind], np.nan)

    def doHeatMaps(self, outdir: "Path" = None, show: "bool" = True, **kwargs):
        for kind, diff_mat in self.metrics.items():
            self.doHeatMap(kind, diff_mat, outdir=outdir, show=show, **kwargs)

    def doHeatMap(self, kind, diff_matrix,
                  outdir: "Path" = None, show: "bool" = True, **kwargs) -> "Tuple[plt.Figure, plt.Axes]":
        numsize = kwargs.get('numsize', 12)
        titlesize = kwargs.get('titlesize', 20)
        textfmt = kwargs.get('textfmt', '.2f')
        title = kwargs.get('title', f'Cl {self.cl_type} {kind}')
        title = title.replace('_PAD', ' Percentage Absolute Difference')
        add_text = kwargs.get('add_text', "")
        add_text_coords = kwargs.get('add_text_coords', (-0.05, 0.80))
        add_textsize = kwargs.get('add_textsize', 20)
        info_text_coords = kwargs.get('info_text_coords', (0.900, 0.800))
        info_textsize = kwargs.get('info_textsize', 16)

        plt.tight_layout(h_pad=0.10)
        fig = plt.figure()
        ax = sns.heatmap(data=diff_matrix, cmap='cool', annot=True,
                         annot_kws={'fontsize': numsize, 'color': 'black'}, fmt=textfmt)
        ax.invert_yaxis()
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        plt.title(title, fontsize=titlesize)
        m_mean = diff_matrix[~np.isnan(diff_matrix)].mean()
        m_max = diff_matrix[~np.isnan(diff_matrix)].max()
        m_min = diff_matrix[~np.isnan(diff_matrix)].min()
        plt.text(*info_text_coords, f'Mean: {m_mean:.3f}\nMax: {m_max:.3f}\nMin: {m_min:.3f}',
                 bbox=dict(boxstyle='round', facecolor='white', linewidth=1.0, edgecolor='lightgrey'),
                 transform=fig.transFigure, fontsize=info_textsize)
        if add_text:
            plt.text(*add_text_coords, add_text,
                     bbox=dict(boxstyle='round', facecolor='white', linewidth=1.0, edgecolor='lightgrey'),
                     transform=fig.transFigure, fontsize=add_textsize)
        if outdir is not None:
            filename = kwargs.get('filename', f"{kind}_{self.cl_type}.png")
            fig.savefig(outdir / filename, bbox_inches='tight')
        if not show:
            plt.close(fig)

        return fig, ax

    def doPADHeatMap(self, outdir: "Path" = None, show: "bool" = True, **kwargs) -> "Tuple[plt.Figure, plt.Axes]":
        return self.doHeatMap('mean_PAD', self.metrics['mean_PAD'], outdir=outdir, show=show, **kwargs)

    def makeRelPlot(self, x, y, size_var=None) -> "sns.axisgrid.FacetGrid":
        splot = sns.relplot(data=self.comp_df, x=x, y=y, size=size_var)
        splot.tight_layout()
        if size_var is not None:
            plt.setp(splot.legend.get_title(), fontsize=20)
            plt.setp(splot.legend.get_texts(), fontsize=14)
        splot.ax.grid(which='major', linestyle='--')
        splot.ax.set_title(self.cl_type.upper())
        return splot

    def writeMetricsToFiles(self, outdir: Path) -> "None":
        for kind in self.metrics:
            np.save(str(outdir/f'{self.cl_type}_cl_{kind}'), self.metrics[kind])
