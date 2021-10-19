from typing import List, Union, Iterable, Dict
import pandas as pd
from pathlib import Path
import logging
import json

from seyfert.utils.tex_utils import TeXTranslator
from seyfert.utils.filesystem_utils import tables_aux_files_dir
from seyfert.numeric import general as nug

logger = logging.getLogger(__name__)
transl = TeXTranslator()

with open(tables_aux_files_dir() / 'tables_defs.json', mode='r') as jsf:
    TABLES_DEFS = json.load(jsf)


def drop_not_numeric_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    to_drop = [
        col for col in df.columns if not any(df[col].dtype == num_type for num_type in {int, float})
    ]

    return df.drop(to_drop, axis=1)


def get_subdf_for_fishers(df: "pd.DataFrame", ordered_names: "List[str]", sort_by: "List[str]" = None,
                          reset_index=False):
    work_df = df.reset_index() if reset_index else df.copy()

    missing = set(ordered_names) - set(work_df['fisher'])
    if missing:
        raise KeyError(f"categories {missing} are not in the fisher column")

    common = set(work_df['fisher']).intersection(set(ordered_names))

    result = work_df[work_df['fisher'].isin(common)]
    fisher_col = pd.Categorical(result['fisher'], categories=ordered_names, ordered=True)
    del result['fisher']
    result.insert(0, 'fisher', fisher_col)
    if sort_by is not None:
        result = result.sort_values(sort_by)

    return result


def write_latex_table(df: "pd.DataFrame", filepath=None, ret_text=True, **kwargs) -> "str":
    default_formatters = {
        key: lambda x: "%.4f" % x for key in df.columns
    }
    default_formatters.update({
        'FoM': lambda x: "%.2f" % x,
        'Delta FoM': lambda x: f"{x:.2f}" if x < 0 else f"+{x:.2f}",
        'Delta FoM (%)': lambda x: f"{x:.2f}%" if x < 0 else f"+{x:.2f}%"
    })
    if isinstance(df.index, pd.MultiIndex):
        default_column_format = r"*{%s}{L}*{%s}{C}" % (len(df.index.levels), len(df.columns))
    else:
        default_column_format = r"*{%s}{C}" % len(df.columns)

    kwds = {
        'formatters': default_formatters,
        'column_format': default_column_format,
        'multirow': True,
        'caption': "Caption",
        'label': "tab:ch4_table"
    }
    kwds.update(kwargs)

    raw_latex_text = df.to_latex(**kwds)
    transl_latex_text = transl.transformRawTeXTable(raw_latex_text, use_aliases=True)

    if filepath is not None:
        Path(filepath).write_text(transl_latex_text)

    retval = transl_latex_text if ret_text else None

    return retval


def get_latex_row(row: "pd.Series", float_fmt="%.4f", ref_fom=None):
    # Add nbins and fisher
    latex_row = ""
    try:
        latex_row += f"{row.loc['nbins']} & "
    except KeyError:
        pass
    latex_row += f"{row.loc['fisher']} & "
    # Add fom
    latex_row += "%.2f & " % row.loc['FoM']
    # Add delta fom if ref fom is given, else put double dash
    if ref_fom is not None:
        delta_fom = 100 * (row.loc['FoM'] - ref_fom) / ref_fom
        str_delta_fom = r"+%.2f\%%" % delta_fom if delta_fom > 0 else r"-%.2f\%%" % delta_fom
        latex_row += f"{str_delta_fom} & "
    else:
        latex_row += "-- & "
    # Add relative marg errors
    latex_row += " & ".join([f"{float_fmt}" % el for el in row[3:]])
    latex_row += r" \\"
    latex_row = transl.translateToTeX(latex_row)

    return latex_row


def write_excel_table(df: "pd.DataFrame", outfile: "Union[str, Path]", overwrite=False, **kwargs):
    kwds = {
        "index": True, "merge_cells": False
    }
    kwds.update(kwargs)
    outfile = Path(outfile)
    if outfile.exists() and not overwrite:
        raise FileExistsError(outfile)
    sheet_name = outfile.stem
    with pd.ExcelWriter(outfile) as writer:
        df.to_excel(writer, sheet_name=sheet_name, **kwds)

        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            writer.sheets[sheet_name].set_column(col_idx, col_idx, column_width)

        writer.save()


def set_results_indices(df: "pd.DataFrame", inplace=True, indices=None) -> "pd.DataFrame":
    if indices is None:
        all_indices = ['fisher', 'n_sp_bins', 'scenario', 'shot_noise_sp_reduced', 'gcph_minus_gcsp',
                       'gcph_only_bins_in_spectro_range']
        indices = [index for index in all_indices if index in df.columns]

    if inplace:
        df.set_index(indices, inplace=True)
        df.sort_index(inplace=True)
        ret_val = None
    else:
        ret_val = df.set_index(indices).sort_index()

    return ret_val


def replace_hybrid_fishers_with_4_bins_equivalents(df: "pd.DataFrame"):
    set_results_indices(df, inplace=True)

    if df.index.names[0] != 'fisher':
        raise KeyError("First index level must be fisher")
    if df.index.names[1] != 'n_sp_bins':
        raise KeyError("Second index level must be n_sp_bins")

    for idx, row in df.iterrows():
        fisher, n_sp_bins, *other = idx
        if "GCsp(Pk)" in fisher and n_sp_bins != 4:
            replacement = df.loc[tuple([fisher, 4, *other])]
            df.loc[idx] = replacement

    df.reset_index(inplace=True)


def load_and_setup_table(file) -> "pd.DataFrame":
    print(f"Reading file {file}")
    df = pd.read_excel(file)
    print("Replacing hybrid fishers with 4 bins equivalents")
    replace_hybrid_fishers_with_4_bins_equivalents(df)

    return df


def default_query_dict() -> "Dict":
    return {
        "scenario": "optimistic",
        "shot_noise_sp_reduced": False,
        "gcph_minus_gcsp": False,
        "gcph_only_bins_in_spectro_range": False
    }


def query_dict(exclude: "Iterable[str]" = None, **kwargs) -> "Dict":
    qdict = default_query_dict()
    qdict.update({key: value for key, value in kwargs.items() if key in qdict})
    if exclude is not None:
        for key in exclude:
            if key in qdict:
                del qdict[key]

    return qdict


def remove_constant_columns_from_df(df: "pd.DataFrame"):
    return df.loc[:, (df != df.iloc[0]).any()]


def select_from_query_dict(df: "pd.DataFrame", query_dict: "Dict", drop_index=False,
                           remove_constant_columns=True) -> "pd.DataFrame":
    unknown_keys = set(query_dict) - set(df.columns)
    if unknown_keys:
        raise KeyError(f"Invalid query keys {unknown_keys}")

    query_pieces = []
    for key, value in query_dict.items():
        if isinstance(value, str):
            value = '"%s"' % value
        query_pieces.append(f"{key} == {value}")

    query_str = " and ".join(query_pieces)
    sel_df = df.query(query_str)
    if drop_index:
        sel_df = sel_df.reset_index(drop=True)
    if remove_constant_columns:
        if len(sel_df) > 1:
            sel_df = remove_constant_columns_from_df(sel_df)

    return sel_df


def compute_percentage_difference(df: "pd.DataFrame", ref_fisher: "str", minuend_fisher: "str") -> "pd.DataFrame":
    minuend = df.loc[minuend_fisher]
    reference = df.loc[ref_fisher]

    return nug.percentage_difference(minuend=minuend, ref=reference)


def compute_simple_difference(df: "pd.DataFrame", ref_fisher: "str", minuend_fisher: "str") -> "pd.DataFrame":
    minuend = df.loc[minuend_fisher]
    reference = df.loc[ref_fisher]

    return minuend - reference


def compute_percentage_differences_table(df: "pd.DataFrame", ref_fisher: "str",
                                         minuend_fishers: "Iterable[str]" = None) -> "pd.DataFrame":
    if minuend_fishers is None:
        if isinstance(df.index, pd.MultiIndex):
            minuend_fishers = df.index.levels[0]
        else:
            raise Exception("Cannot infer minuend fishers for with no multi-index")

    diff = pd.concat({
        minuend: compute_percentage_difference(df, ref_fisher=ref_fisher, minuend_fisher=minuend)
        for minuend in minuend_fishers
    }, names=["fisher"])

    if isinstance(diff, pd.Series):
        diff = diff.unstack()

    return diff


def compute_simple_differences_table(df: "pd.DataFrame", ref_fisher: "str",
                                     minuend_fishers: "Iterable[str]" = None) -> "pd.DataFrame":
    if minuend_fishers is None:
        if isinstance(df.index, pd.MultiIndex):
            minuend_fishers = df.index.levels[0]
        else:
            raise Exception("Cannot infer minuend fishers for with no multi-index")

    diff = pd.concat({
        minuend: compute_simple_difference(df, ref_fisher=ref_fisher, minuend_fisher=minuend)
        for minuend in minuend_fishers
    }, names=["fisher"])

    if isinstance(diff, pd.Series):
        diff = diff.unstack()

    return diff


def get_fom_table_for_fishers(df: "pd.DataFrame", fishers: "List[str]", dfom_map: "Dict"):
    subdf = get_subdf_for_fishers(df, ordered_names=fishers)
    set_results_indices(subdf)

    diffs = {
        "simple": {},
        "perctg": {}
    }
    for minuend in subdf.index.get_level_values('fisher'):
        if minuend in dfom_map:
            ref = dfom_map[minuend]
            diffs["simple"][minuend] = compute_simple_difference(subdf, ref_fisher=ref, minuend_fisher=minuend)
            diffs["perctg"][minuend] = compute_percentage_difference(subdf, ref_fisher=ref, minuend_fisher=minuend)

    diffs["simple"] = pd.concat(diffs["simple"])
    diffs["perctg"] = pd.concat(diffs["perctg"])

    subdf.insert(1, "Delta FoM", diffs['simple']['FoM'])
    subdf.insert(2, "Delta FoM (%)", diffs['perctg']['FoM'])

    return subdf

