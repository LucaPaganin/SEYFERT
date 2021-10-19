from typing import List, Dict, Set, Tuple
import itertools
import numpy as np
from pathlib import Path
import pandas as pd

from seyfert.utils import array_utils
from seyfert.utils import general_utils as gu
from seyfert.utils import filesystem_utils as fsu
from seyfert.fisher import linalg_tools as lt
from seyfert.config.forecast_config import ForecastConfig
from seyfert.fisher.datavector import ClDataVector

ObsComboType = Tuple[str, str]
CovLayoutEntry = Tuple[ObsComboType, ObsComboType]
DataVectorType = List[ObsComboType]

SINGLE_PROBES = ["WL", "GCph", "GCsp"]


def buildXCBlockMatrixForProbes(probes: "List[str]",
                                matrix_dict: "Dict[str, np.ndarray]",
                                l_dict: "Dict[str, np.ndarray]") -> "Tuple[np.ndarray, np.ndarray]":
    relevant_combs = []
    for probe_1, probe_2 in itertools.combinations_with_replacement(probes, 2):
        comb_key = gu.get_probes_combination_key(probe_1, probe_2)
        if comb_key not in matrix_dict:
            comb_key = gu.get_probes_combination_key(probe_2, probe_1)
            if comb_key not in matrix_dict:
                raise KeyError(f"Combination {comb_key} not found into matrix dict keys {' '.join(set(matrix_dict))}")

        relevant_combs.append(comb_key)

    sel_tri_dict = {probe_comb_key: matrix_dict[probe_comb_key] for probe_comb_key in relevant_combs}
    sel_l_dict = {probe_comb_key: l_dict[probe_comb_key] for probe_comb_key in relevant_combs}
    l_common = array_utils.compute_arrays_intersection1d(list(sel_l_dict.values()))
    rows_list = []
    for probe_1 in probes:
        row = []
        for probe_2 in probes:
            probe_comb_key = gu.get_probes_combination_key(probe_1, probe_2)
            if probe_comb_key in sel_tri_dict:
                block = sel_tri_dict[probe_comb_key]
            else:
                probe_comb_key = gu.get_probes_combination_key(probe_2, probe_1)
                block = np.transpose(sel_tri_dict[probe_comb_key], axes=(0, 2, 1))
            # Select only entries with common multipoles
            common_multipoles_mask = np.isin(l_dict[probe_comb_key], l_common)
            block = block[common_multipoles_mask, :, :]
            row.append(block)
        rows_list.append(row)

    return l_common, np.block(rows_list)


def get_probes_xc_fisher_name(probes: "List[str]") -> "str":
    if len(probes) == 1:
        name = probes[0]
    else:
        name_parts = probes + ['XC']
        name = "+".join(name_parts)
    return name


def get_probes_from_fisher_name(fisher_name: str) -> List[str]:
    parts = [x for x in fisher_name.split('+')]
    if 'XC' in parts:
        parts.remove('XC')
    return parts


def get_nuisance_set_for_fisher(fisher_name: "str", forecast_config: "ForecastConfig") -> "Set[str]":
    probes = get_probes_from_fisher_name(fisher_name)
    nuis_pars = set()
    for probe in probes:
        nuis_pars |= set(forecast_config.phys_pars.getFreeNuisanceParametersForProbe(probe))

    return nuis_pars


def invert_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    return pd.DataFrame(np.linalg.inv(df), index=df.index, columns=df.columns)


def load_selected_data_vectors() -> "List[ClDataVector]":
    file = fsu.fisher_aux_files_dir() / "selected_datavectors.txt"

    data_vectors = []

    for line in Path(file).read_text().splitlines():
        line = line.strip()
        if not line.startswith("#") and len(line) > 0:
            data_vectors.append(ClDataVector.fromString(line))

    return data_vectors


def load_a_posteriori_fisher_comb_names() -> "List[str]":
    file = fsu.fisher_aux_files_dir() / "a_posteriori_fisher_combs.txt"
    lines = []
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line.startswith("#") and len(line) > 0:
            lines.append(line)

    return lines


def vecClArray(c_lij_arr: "np.ndarray"):
    return np.stack([
        lt.vec(c_lij_arr[ell_idx]) for ell_idx in range(c_lij_arr.shape[0])
    ])


def vecpClArray(c_lij_arr: "np.ndarray"):
    return np.stack([
        lt.vecp(c_lij_arr[ell_idx]) for ell_idx in range(c_lij_arr.shape[0])
    ])
