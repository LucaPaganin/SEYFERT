from typing import List, Dict
import re
import pandas as pd

from seyfert.utils.tex_utils import TeXTranslator
from seyfert.fisher.fisher_analysis import FisherAnalysis

transl = TeXTranslator()


def remove_spaces(string_list: "List[str]"):
    return [re.sub(r"\s", "", s) for s in string_list]


def get_subset_analyses_for_fishers(analyses: "Dict[int, FisherAnalysis]", fisher_names: "List[str]"):
    filter_names = remove_spaces(fisher_names)

    ans = {
        nbins: all_fish_an.getSubsetAnalysis(filter_names) for nbins, all_fish_an in analyses.items()
    }

    for nbins in ans:
        ans[nbins].prepareFisherMatrices()
        ans[nbins].evaluateMarginalizedErrors()

    return ans
