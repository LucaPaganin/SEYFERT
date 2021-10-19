from pathlib import Path

from seyfert.notebook_helpers import table_helpers as th
from seyfert.utils import tex_utils

transl = tex_utils.TeXTranslator()

workdir = Path("/Users/lucapaganin/GoogleDrive/Work/Dottorato/Euclid/PhotoSpectroXCorr/notebooks/results_thesis/tables")

all_res = th.load_and_setup_table(workdir.parent / "Risultati/1.3.8/w0_wa_CDM/marg_before/relative_marginalized_errors.xlsx")

qdict = th.query_dict()
subdf = th.select_from_query_dict(all_res, qdict)
fishers = th.TABLES_DEFS['harmonic_vs_hybrid']['fishers']
dfom_map = th.TABLES_DEFS['harmonic_vs_hybrid']['delta_fom_ref']


subdf = th.get_fom_table_for_fishers(subdf, fishers, dfom_map)


