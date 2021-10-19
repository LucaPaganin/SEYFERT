import pytest
import datetime

from seyfert.utils import general_utils as gu
from seyfert.utils import formatters as fm


def test_split_run_id():
    rundir_name = 'ph_sp_wl_4_bins_bias_stem_1.3.2_2021-03-07T19-23-56'
    run_id, version, date_string = gu.split_run_id(rundir_name)

    assert run_id == 'ph_sp_wl_4_bins_bias_stem'
    assert version == '1.3.2'
    assert date_string == fm.date_from_str("2021-03-07T19-23-56")


def test_replace_fullnames_with_aliases():

    assert gu.replace_fullnames_with_aliases("Lensing") == "WL"
    assert gu.replace_fullnames_with_aliases("Lensing+PhotometricGalaxy") == "WL+GCph"
    assert gu.replace_fullnames_with_aliases("SpectroscopicGalaxy") == "GCsp"

