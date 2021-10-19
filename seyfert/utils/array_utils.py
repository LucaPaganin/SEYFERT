from typing import List, Tuple
import numpy as np


def compute_arrays_intersection1d(arrays: "List[np.ndarray]") -> "np.ndarray":
    if len(arrays) == 1:
        arr_intersection = arrays[0]
    else:
        arr_intersection = np.intersect1d(arrays[0], arrays[1])
        for arr in arrays[1:]:
            arr_intersection = np.intersect1d(arr_intersection, arr)

    return arr_intersection


def symmetrize_clij_arr(X: "np.ndarray") -> "np.ndarray":
    return (X + np.transpose(X, axes=(0, 2, 1))) / 2


def compute_arrays_union1d(arrays: "List[np.ndarray]") -> "np.ndarray":
    arr_union = np.union1d(arrays[0], arrays[1])
    for arr in arrays[1:]:
        arr_union = np.union1d(arr_union, arr)

    return arr_union


def compute_ell_bin_widths_from_log_centers(l_bin_centers: "np.ndarray") -> "np.ndarray":
    Dlog_ell = np.log10(l_bin_centers[1]/l_bin_centers[0])

    return 2 * ((10**Dlog_ell - 1)/(10**Dlog_ell + 1)) * l_bin_centers


def compute_multipoles_arrays(l_min: "float", l_max: "float", spacing: "str",
                              n_ell: "int" = None) -> "Tuple[np.ndarray, np.ndarray]":
    if l_min >= l_max:
        raise ValueError(f"Invalid value for l_min {l_min} with l_max {l_max}. l_min must be < l_max")

    if spacing in {"lin", "linear", "Linear"}:
        l_bin_edges = np.arange(l_min, l_max + 1, step=1)
        l_bin_centers = (l_bin_edges[:-1] + l_bin_edges[1:]) / 2
        l_bin_widths = np.ones(l_bin_centers.shape)
    elif spacing in {"log", "logarithmic", "Logarithmic"}:
        if n_ell is None:
            raise ValueError("n_ell is required when doing log spacing!")
        if not isinstance(n_ell, (int, float)):
            raise TypeError("n_ell must be int or float")
        l_bin_edges = np.logspace(np.log10(l_min), np.log10(l_max), num=int(n_ell) + 1, endpoint=True)
        l_bin_centers = (l_bin_edges[:-1] + l_bin_edges[1:]) / 2
        l_bin_widths = np.diff(l_bin_edges)
    else:
        raise KeyError(f"Invalid spacing {spacing}, must be one of lin/linear/Linear or "
                       f"log/logarithmic/Logarithmic")

    return l_bin_centers, l_bin_widths
