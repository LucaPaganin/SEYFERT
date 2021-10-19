import pytest
from typing import Tuple

from seyfert.utils.filesystem_utils import fisher_aux_files_dir
from seyfert.fisher.datavector import ClDataVector

NAME_VECTOR_MAP = {}
for line in (fisher_aux_files_dir() / "all_datavectors.txt").read_text().splitlines():
    dv = ClDataVector.fromString(line.strip())
    NAME_VECTOR_MAP[dv.name] = dv


@pytest.fixture(scope="class", params=list(NAME_VECTOR_MAP.items()), ids=[name for name in NAME_VECTOR_MAP])
def name_vector(request) -> "Tuple[str, ClDataVector]":
    yield request.param


class TestDatavector:
    name: "str"
    vector: "ClDataVector"

    @pytest.fixture(autouse=True)
    def setup(self, name_vector):
        name, vector = name_vector
        self.name = name
        self.vector = vector

    def test_naming(self):
        assert self.name == self.vector.name

    def test_equality(self):
        assert self.vector == self.vector

    def test_equivalent_vectors(self):
        assert self.vector == self.vector.toShortKeysVector()
        assert self.vector == self.vector.toTupleKeysVector()

    def test_from_name(self):
        assert self.vector == ClDataVector.fromName(self.name)

    def test_to_from_string(self):
        assert self.vector == ClDataVector.fromString(self.vector.toString())

    def test_to_from_brief_string(self):
        assert self.vector == ClDataVector.fromBriefString(self.vector.toBriefString())


INVOLVED_PROBES_MAP = {
    "wlwl": {"WL"},
    "wlwl_phph": {"WL", "GCph"},
    "wlwl_wlph": {"WL"},
    "wlwl_phph_wlph": {"WL", "GCph"},
    "wlph": set(),
    "wlwl_spsp": {"WL", "GCsp"},
    "wlwl_spsp_wlsp": {"WL", "GCsp"},
    "wlwl_wlsp": {"WL"},
    "wlsp": set(),
    "phph": {"GCph"},
    "phph_spsp": {"GCph", "GCsp"},
    "phph_spsp_phsp": {"GCph", "GCsp"},
    "phph_phsp": {"GCph"},
    "phsp": set(),
    "wlph_wlsp": set(),
    "wlsp_phsp": set(),
    "wlwl_phph_spsp_wlph_wlsp_phsp": {"WL", "GCph", "GCsp"},
}


@pytest.fixture(params=list(INVOLVED_PROBES_MAP.items()), ids=[name for name in INVOLVED_PROBES_MAP])
def brief_str_probes(request):
    yield request.param


def test_involved_probes(brief_str_probes):
    brief_str, probes = brief_str_probes
    dv = ClDataVector.fromBriefString(brief_str)
    assert dv.getInvolvedAutoCorrelationProbes() == probes
