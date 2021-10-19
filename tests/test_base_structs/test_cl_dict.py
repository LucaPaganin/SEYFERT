import pytest
import numpy as np
import itertools

from seyfert.base_structs.cl_dict import ClDict
from seyfert.utils import general_utils as gu
from seyfert import PROBE_LONG_TO_ALIAS


class TestClDict:
    @pytest.fixture(autouse=True)
    def setup(self, cl_dict):
        self.cl_dict = ClDict(data_dict=cl_dict)

    def __getitem__(self, item):
        return self.cl_dict[item]

    def test_rev_key(self):
        for key in self.cl_dict.keys():
            rev_key = gu.reverse_probes_comb_key(key)
            assert np.all(self[rev_key] == np.transpose(self[key], axes=(0, 2, 1)))

    def test_alias_to_fullname(self):
        for direct_key, alias_key in PROBE_LONG_TO_ALIAS.items():
            assert self.cl_dict.alias_to_fullname(alias_key) == direct_key

    def test_fullname_to_alias(self):
        for direct_key, alias_key in PROBE_LONG_TO_ALIAS.items():
            assert self.cl_dict.fullname_to_alias(direct_key) == alias_key

    def test_alias_getitem(self):
        for p1, p2 in itertools.combinations_with_replacement(PROBE_LONG_TO_ALIAS.values(), 2):
            tpl_key = (p1, p2)
            p1 = self.cl_dict.alias_to_fullname(p1)
            p2 = self.cl_dict.alias_to_fullname(p2)
            direct_key = gu.get_probes_combination_key(p1, p2)

            if direct_key in self.cl_dict:
                assert np.all(self[tpl_key] == self[direct_key])
            else:
                rev_key = gu.get_probes_combination_key(p2, p1)
                assert np.all(self[tpl_key] == self[rev_key])

