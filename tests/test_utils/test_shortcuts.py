import pytest
import logging
import json
import numpy as np
from pathlib import Path

from seyfert.utils import general_utils as gu
from seyfert.utils import shortcuts
# noinspection PyProtectedMember
from seyfert import SEYFERT_PATH

logger = logging.getLogger(__name__)
gu.configure_logger(logger)

with open(Path(SEYFERT_PATH).parent / "tests/data/cl_key_long_to_short.json") as jsf:
    LONG_TO_SHORT_KEY_MAPPING = json.load(jsf)

with open(Path(SEYFERT_PATH).parent / "tests/data/cl_key_long_to_alias.json") as jsf:
    LONG_TO_ALIAS_KEY_MAPPING = json.load(jsf)

with open(Path(SEYFERT_PATH).parent / "tests/data/cl_key_short_to_alias.json") as jsf:
    SHORT_TO_ALIAS_KEY_MAPPING = json.load(jsf)


LONG_KEYS = set(LONG_TO_SHORT_KEY_MAPPING.keys())
SHORT_KEYS = set(LONG_TO_SHORT_KEY_MAPPING.values())
ALIAS_KEYS = set([tuple(ak) for ak in LONG_TO_ALIAS_KEY_MAPPING.values()])

ALL_KEYS = set.union(LONG_KEYS, SHORT_KEYS, ALIAS_KEYS)


class TestClKey:
    @pytest.fixture(params=list(ALL_KEYS), ids=[str(k) for k in ALL_KEYS],
                    autouse=True)
    def setup(self, request):
        self.cl_key = shortcuts.ClKey(request.param)

    def test_to_from_parts(self):
        res_p1, res_p2 = self.cl_key.toParts()
        if self.cl_key.kind == "long":
            exp_p1, exp_p2 = self.cl_key.key.split("_")
        elif self.cl_key.kind == "short":
            exp_p1, exp_p2 = self.cl_key.key[0:2], self.cl_key.key[2:]
        elif self.cl_key.kind == "alias":
            exp_p1, exp_p2 = self.cl_key.key
        else:
            raise Exception(f"Invalid {self.cl_key}")

        assert res_p1 == exp_p1
        assert res_p2 == exp_p2

        new_key = shortcuts.ClKey.fromParts(res_p1, res_p2, kind=self.cl_key.kind)

        assert self.cl_key == new_key

    def testToShort(self):
        short_key = self.cl_key.toShortKey()
        logger.info(f"{self.cl_key} - {short_key}")
        assert self.cl_key.equivalentTo(short_key)

    def testToLong(self):
        long_key = self.cl_key.toLongKey()
        logger.info(f"{self.cl_key} - {long_key}")
        assert self.cl_key.equivalentTo(long_key)

    def testToAlias(self):
        alias_key = self.cl_key.toAliasKey()
        logger.info(f"{self.cl_key} - {alias_key}")
        assert self.cl_key.equivalentTo(alias_key)

    def testReverse(self):
        p1, p2 = self.cl_key.toParts()
        rev_key = shortcuts.ClKey.fromParts(p2, p1, kind=self.cl_key.kind)

        assert self.cl_key.getReverseKey() == rev_key

    def testFromPartsInvalidKind(self):
        p1, p2 = self.cl_key.toParts()
        with pytest.raises(shortcuts.InvalidKeyError):
            _ = shortcuts.ClKey.fromParts(p1, p2, kind="not valid")

    def testIsDiagonal(self):
        p1, p2 = self.cl_key.toParts()

        if p1 == p2:
            assert self.cl_key.is_auto_correlation
        else:
            assert not self.cl_key.is_auto_correlation

    def testFromKey(self):
        new = shortcuts.ClKey(self.cl_key)

        assert new == self.cl_key


SHAPES = {
    "ph": (100, 10),
    "sp": (100, 4),
    "wl": (2100, 10),
    "vd": (100, 10),
}


@pytest.fixture(scope="class")
def smart_ell_dict() -> "shortcuts.SmartKeyDict":
    s_l_dict = shortcuts.SmartKeyDict()
    for k in SHORT_KEYS:
        cl_key = shortcuts.ClKey(k)
        if cl_key.getReverseKey() in s_l_dict:
            continue
        else:
            p1, p2 = cl_key.toParts()
            shp1, shp2 = SHAPES[p1], SHAPES[p2]
            n_ell = min(shp1[0], shp2[0])
            data = np.arange(n_ell)

            s_l_dict[cl_key] = data

    return s_l_dict


@pytest.fixture(scope="class")
def smart_cl_dict() -> "shortcuts.ClSmartKeyDict":
    s_cl_dict = shortcuts.ClSmartKeyDict()
    for k in SHORT_KEYS:
        cl_key = shortcuts.ClKey(k)
        if cl_key.getReverseKey() in s_cl_dict:
            continue
        else:
            p1, p2 = cl_key.toParts()
            shp1, shp2 = SHAPES[p1], SHAPES[p2]
            n_ell = min(shp1[0], shp2[0])
            ni, nj = shp1[1], shp2[1]
            data = np.random.random((n_ell, ni, nj))
            if p1 == p2:
                data = (data + np.transpose(data, axes=(0, 2, 1)))/2

            s_cl_dict[cl_key] = data

    return s_cl_dict


@pytest.fixture(params=["smart_ell_dict", "smart_cl_dict"])
def generic_smart_dict(request) -> "shortcuts.SmartKeyDict":
    return request.getfixturevalue(request.param)


class TestSmartDict:
    def test_alias_key(self, generic_smart_dict):
        for key, value in generic_smart_dict.items():
            for kind in ["long", "short", "alias"]:
                alias = key.transformTo(kind)
                assert np.all(value == generic_smart_dict[alias])
                print(f"Correct alias {alias.key} of {key.key}")

    def test_reverse_key_ell_dict(self, smart_ell_dict):
        for key, value in smart_ell_dict.items():
            assert np.all(value == smart_ell_dict[key.getReverseKey()])
            assert np.all(smart_ell_dict[key.getReverseKey()] == smart_ell_dict.reversed_key_transform(value))

    def test_reverse_key_cl_dict(self, smart_cl_dict):
        for key, value in smart_cl_dict.items():
            assert np.all(smart_cl_dict[key.getReverseKey()] == smart_cl_dict.reversed_key_transform(value))
