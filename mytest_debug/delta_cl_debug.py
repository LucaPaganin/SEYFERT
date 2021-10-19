from seyfert.main import cl_core
from seyfert.utils.workspace import WorkSpace

ws = WorkSpace("/Users/lucapaganin/spectrophoto/production_runs/1.3.8/SSC/optm_4_sp_bins_ISTF_WL_Flat_powerspectra_1.3.8_2021-10-09T14-15-41/")

delta_cls = cl_core.compute_delta_cls(ws)
