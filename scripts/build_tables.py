from pathlib import Path
import pandas as pd

from seyfert.utils import tex_utils
from seyfert.notebook_helpers import table_helpers as th

transl = tex_utils.TeXTranslator()


workdir = Path("/Users/lucapaganin/GoogleDrive/Work/Dottorato/Euclid/PhotoSpectroXCorr/notebooks/results_thesis/tables")

def get_sub_table(df, fishers, delta_fom_ref):
    subdf = df.loc[fishers].round(8).drop_duplicates().reset_index()
    subdf.fisher = pd.Categorical(subdf.fisher, categories=fishers, ordered=True)
    if 'n_sp_bins' in subdf.columns:
        subdf = subdf.set_index(['fisher', 'n_sp_bins'])
    else:
        subdf = subdf.set_index(['fisher'])

    subdf.sort_index(inplace=True)
    add_deltafom_column(subdf, ref_fisher=delta_fom_ref)

    return subdf


def add_deltafom_column(df, ref_fisher):
    if isinstance(df.index, pd.MultiIndex):
        minuend_fishers = list(df.index.levels[0])
    else:
        minuend_fishers = list(df.index)
    diff = th.compute_percentage_differences_table(df, ref_fisher=ref_fisher,
                                                   minuend_fishers=minuend_fishers)
    diff = diff.dropna()
    df.insert(1, 'Delta FoM', diff.FoM)


all_res = th.load_and_setup_table(workdir.parent / "Risultati/1.3.8/w0_wa_CDM/marg_before/relative_marginalized_errors.xlsx")

combos = all_res[list(th.default_query_dict())].drop_duplicates()

outdir = workdir / "output"

for combo in combos.itertuples(index=False):
    qdict = combo._asdict()
    print("Slicing for query: ")
    print(qdict)
    scenario = qdict['scenario']
    out_subdir = outdir.joinpath(*[f"{key}_{value}" for key, value in qdict.items()])
    out_subdir.mkdir(exist_ok=True, parents=True)
    res = th.select_from_query_dict(all_res, qdict, drop_index=True)
    th.set_results_indices(res)
    for case, tabdef in th.TABLES_DEFS.items():
        fishers = tabdef['fishers']
        delta_fom_ref = tabdef['delta_fom_ref']
        subdf = get_sub_table(res, fishers, delta_fom_ref=delta_fom_ref)

        label = f"tab:apxB_{case}_{scenario}"
        caption = rf"Baseline results for {case} in the {scenario} scenario. The $\Delta\FoM$ column " \
                  rf"is computed taking ${transl.toTeX(delta_fom_ref, use_aliases=True)}$ as reference."

        th.write_excel_table(df=subdf, outfile=out_subdir / f"{case}.xlsx", overwrite=True)
        th.write_latex_table(df=subdf, filepath=out_subdir / f"{case}.txt", ret_text=False,
                             multirow=True, label=label, caption=caption)
