from typing import List, Tuple, Dict, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from seyfert.utils.tex_utils import TeXTranslator

if TYPE_CHECKING:
    from seyfert.cosmology.c_ells import AngularCoefficientsCollector
    from seyfert.cosmology.cosmology import Cosmology
    from seyfert.cosmology.parameter import PhysicalParametersCollection

translator = TeXTranslator()


def cl_plot(ell_list: "List[np.ndarray]", cl_data_list: "List[np.ndarray]", labels: "List[str]",
            i_min=0, j_min=0, i_max=9, j_max=9, wspace=0.15, hspace=0.15, logx=False, logy=False, figsize=(20, 20)):
    if not isinstance(ell_list, list):
        raise TypeError("ell data must be list")

    if not isinstance(cl_data_list, list):
        raise TypeError("cl data must be list")

    if not isinstance(labels, list):
        raise TypeError("labels must be list")

    nrows = i_max - i_min + 1
    ncols = j_max - j_min + 1

    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            row_idx = i - i_min
            col_idx = j - j_min
            ax = axes[row_idx, col_idx]
            labelbottom = row_idx == nrows - 1
            if labelbottom:
                ax.set_xlabel(j)
            if col_idx == 0:
                ax.set_ylabel(i)
            if i >= j:
                for x, y, label in zip(ell_list, cl_data_list, labels):
                    ax.plot(x, y[:, i, j], label=label)
                    if logx:
                        ax.set_xscale("log")
                    if logy:
                        ax.set_yscale("log")
            else:
                ax.axis('off')

            ax.tick_params(labelbottom=labelbottom)

    return fig, axes


def grid_plot(x: "np.ndarray", y: "np.ndarray", label: "str" = None, axes: "np.ndarray[plt.Axes]" = None,
              idx_fontsize: "float" = 12, i_label_coords: "Tuple[float]" = None, j_label_coords: "Tuple[float]" = None,
              draw_idx_labels: "bool" = True,
              figsize: "Tuple" = None, hspace: "float" = None, wspace: "float" = None, tick_kwds: "Dict" = None,
              legend_bbox: "Tuple" = None, logx: "bool" = False, logy: "bool" = False,
              triangular: "bool" = False, xlabel: "str" = None, **kwargs) -> "Tuple[plt.Figure, np.ndarray[plt.Axes]]":
    if len(x.shape) != 1:
        raise Exception("x array must be 1D for grid plot")
    if len(y.shape) != 3:
        raise Exception("y array must be 3D for grid plot")
    nrows, ncols = y.shape[0:2]
    if nrows != ncols and triangular:
        raise Exception("Cannot do traingular plot for non-square grid")
    if axes is None:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
    else:
        fig = plt.gcf()

    tick_kwargs = {}
    if tick_kwds is not None:
        tick_kwargs.update(tick_kwds)

    if i_label_coords is None:
        i_label_coords = (-0.60, 0.25)
    if j_label_coords is None:
        j_label_coords = (0.60, 1.10)

    for i in range(nrows):
        for j in range(ncols):
            if triangular and i < j:
                axes[i, j].axis('off')
            else:
                axes[i, j].plot(x, y[i, j], label=label, **kwargs)

                if logx:
                    axes[i, j].set_xscale('log')
                if logy:
                    axes[i, j].set_yscale('log')

                tick_kwargs['labelbottom'] = i == (nrows - 1)

                i_label_cond = draw_idx_labels and j == 0
                j_label_cond = draw_idx_labels and ((i == j) if triangular else (i == 0))

                if i_label_cond:
                    axes[i, j].text(*i_label_coords, f'i = {i}', transform=axes[i, j].transAxes, fontsize=idx_fontsize)
                if j_label_cond:
                    axes[i, j].text(*j_label_coords, f'j = {j}', transform=axes[i, j].transAxes, fontsize=idx_fontsize)

                x_label_cond = i == (nrows - 1)
                if x_label_cond and xlabel is not None:
                    axes[i, j].set_xlabel(xlabel)

                axes[i, j].tick_params(**tick_kwargs, labelsize=12)

    default_bbox_to_anchor = [0.920, 0.850] if not triangular else [0.400, 0.800]

    bbox_to_anchor = legend_bbox if legend_bbox is not None else default_bbox_to_anchor
    plt.legend(bbox_to_anchor=bbox_to_anchor, bbox_transform=fig.transFigure)

    return fig, axes


def plot_weight_funcs(cl_coll: "AngularCoefficientsCollector", cosmo: "Cosmology",
                      phys_pars: "PhysicalParametersCollection"):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16.0 / 2.54, 21.0 / 2.54), sharex=True, dpi=100)
    plt.subplots_adjust(hspace=0.225)

    pl_kwargs = {'linewidth': 1.0}
    legend_fontsize = 7
    # legend_handletextpad = 0.4
    legend_labelspacing = 0.4

    weight_funcs = {
        "GCph": cl_coll.weight_dict["PhotometricGalaxy"],
        "GCsp": cl_coll.weight_dict["SpectroscopicGalaxy"],
        "WL": cl_coll.weight_dict["Lensing"],
    }
    weight_funcs['WL'].nuisance_params = {
        name: phys_pars[name] for name in phys_pars.getNuisanceParametersForProbe("Lensing")
    }
    weight_funcs['WL'].cosmology = cosmo

    w_ia_bin_z = weight_funcs['WL'].computeIntrinsicAlignmentContribution()

    colormap = matplotlib.cm.get_cmap('rainbow')

    for idx, (name, ax) in enumerate(zip(["GCph", "GCsp", "WL"], axs)):
        w = weight_funcs[name]
        leg1_handles = []
        colors = colormap(np.linspace(0, 1, w.n_bins))
        for i in range(w.n_bins):
            line = ax.plot(w.z_grid, w.w_bin_z[i], color=colors[i], label=r"$%s$" % (i + 1), linewidth=1.0)[0]
            leg1_handles.append(line)
            if name == 'WL':
                ax.plot(w.z_grid, w.w_bin_z[i] + w_ia_bin_z[i], color=colors[i], linestyle='--', linewidth=1.0)

        text = r"$%s$" % translator.translateToTeX(name, use_aliases=False)
        ax.text(0.920, 0.850, text, fontsize=16, transform=ax.transAxes, ha='center')
        legend1 = plt.gca().legend(handles=leg1_handles, bbox_to_anchor=(1, -0.02400), bbox_transform=ax.transAxes,
                                   fontsize=legend_fontsize,
                                   loc='lower left', labelspacing=legend_labelspacing)
        plt.gca().add_artist(legend1)

        ax.tick_params(labelsize=11)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        sci_notation_text = ax.yaxis.get_offset_text()
        sci_notation_text.set_fontsize(11)

        ylabelpad = 0 if name == 'WL' else 10
        ax.set_ylabel(r"$W_i(z)[\mathrm{Mpc}^{-1}]$", labelpad=ylabelpad, rotation=90, fontsize=11)

        if name == 'WL':
            solid_line = matplotlib.lines.Line2D([], [], color='black', linestyle='-', linewidth=0.5,
                                                 label='IA included')
            dashed_line = matplotlib.lines.Line2D([], [], color='black', linestyle='--', linewidth=0.5,
                                                  label='IA not included')
            legend2 = plt.gca().legend(handles=[solid_line, dashed_line], bbox_to_anchor=(1, 0.900),
                                       bbox_transform=ax.transAxes,
                                       fontsize=legend_fontsize, loc='lower left', labelspacing=legend_labelspacing)

    return fig, axs