#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import itertools

from seyfert.config.forecast_config import ForecastConfigEditor

main_outdir = Path.home() / f"spectrophoto/config_files/latest/"

loop_dict = {
    "scenario": ["optimistic", "pessimistic"],
    "n_sp_bins": [4, 12, 24, 40],
    "shot_noise_sp_reduced": [True, False],
    "gcph_minus_gcsp": [True, False],
    "gcph_only_bins_in_spectro_range": [False]
}


for combo in itertools.product(*loop_dict.values()):
    opts = dict(zip(loop_dict.keys(), combo))
    print(f"Generating config for {opts}")
    ed = ForecastConfigEditor(opts)
    ed.updateData()
    outdir = main_outdir
    outdir.mkdir(exist_ok=True, parents=True)
    ed.writeJSON(outdir, add_datetime=False)
