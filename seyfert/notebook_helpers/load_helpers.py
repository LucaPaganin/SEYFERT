from typing import TYPE_CHECKING, Union, Dict, List
from pathlib import Path
import re

from seyfert.utils import workspace, dir_explorer
from seyfert.fisher.fisher_analysis import FisherAnalysis


def invert_fishers_and_compute_errors(analysis: "FisherAnalysis"):
    analysis.prepareFisherMatrices()
    analysis.evaluateMarginalizedErrors()
    analysis.evaluateRelativeMarginalizedErrors()


def invert_fishers_in_all_analyses(analyses: "Dict[int, FisherAnalysis]"):
    for nbins in analyses:
        invert_fishers_and_compute_errors(analyses[nbins])


def replace_hybrid_fishers_with_4_bins_equivalents(analyses: "Dict[int, FisherAnalysis]", verbose=False):
    for nbins in analyses:
        an_4_bins = analyses[4]
        fisher_names = analyses[nbins].fisher_matrices.keys()
        for fisher_name in fisher_names:
            if "GCsp(Pk)" in fisher_name:
                if verbose:
                    print(f"replacing fisher {fisher_name} {nbins} bins with its 4 bins equivalent")

                analyses[nbins].fisher_matrices[fisher_name] = an_4_bins.fisher_matrices[fisher_name]


def load_workspaces_for_scenario(scenario: "str", gcph_sub_flag=False,
                                 base_dir=None) -> "Dict[int, workspace.WorkSpace]":

    generic_regex = re.compile(r"^(?P<scenario>optm|pess)_(?P<nbins>[0-9]+)_sp_bins_")
    gcph_sub_gcsp_regex = re.compile(r"gcph_minus_gcsp")

    if base_dir is None:
        base_dir = Path("/Users/lucapaganin/spectrophoto/production_runs/latest")
        base_dir = base_dir / {'optm': 'optimistic', 'pess': 'pessimistic'}[scenario]

    base_selection = lambda x: x.is_dir() and generic_regex.match(x.name)
    if not gcph_sub_flag:
        selection_func = lambda x: base_selection(x) and not gcph_sub_gcsp_regex.search(x.name)
    else:
        selection_func = lambda x: base_selection(x) and gcph_sub_gcsp_regex.search(x.name)

    ws_dict = {}

    for direc in base_dir.iterdir():
        if selection_func(direc):
            match = generic_regex.match(direc.name)
            nbins = int(match.group('nbins'))
            ws_dict[nbins] = workspace.WorkSpace(direc)

    return ws_dict


def load_fisher_analyses_for_scenario(scenario: "str", gcph_sub_flag=False,
                                      base_dir=None, **kwargs) -> "Dict[int, FisherAnalysis]":

    ws_dict = load_workspaces_for_scenario(scenario, gcph_sub_flag=gcph_sub_flag, base_dir=base_dir)

    analyses = {}
    for nbins, ws in ws_dict.items():
        analyses[nbins] = FisherAnalysis.fromRundir(rundir=ws.run_dir, **kwargs)

    return analyses
