from typing import Tuple, Union, Dict
import re
import copy

import numpy as np

from seyfert.base_structs.generic_dict import GenericDictInterface
# noinspection PyProtectedMember
from seyfert import \
    PROBES_LONG_NAMES, PROBES_SHORT_NAMES, PROBES_ALIAS_NAMES, \
    PROBE_SHORT_TO_LONG, PROBE_LONG_TO_SHORT, \
    PROBE_LONG_TO_ALIAS, PROBE_ALIAS_TO_LONG, \
    PROBE_SHORT_TO_ALIAS, PROBE_ALIAS_TO_SHORT

TRawClKey = Union[str, Tuple[str, str]]


_TRANSL_MAPPINGS = {
    "long_to_short": PROBE_LONG_TO_SHORT,
    "short_to_long": PROBE_SHORT_TO_LONG,
    "long_to_alias": PROBE_LONG_TO_ALIAS,
    "alias_to_long": PROBE_ALIAS_TO_LONG,
    "short_to_alias": PROBE_SHORT_TO_ALIAS,
    "alias_to_short": PROBE_ALIAS_TO_SHORT
}


long_cl_key_regex = re.compile(
        r"(PhotometricGalaxy|SpectroscopicGalaxy|Lensing|Void)_(PhotometricGalaxy|SpectroscopicGalaxy|Lensing|Void)")
short_cl_key_regex = re.compile(r"(ph|sp|wl|vd)(ph|sp|wl|vd)")
alias_probe_name_regex = re.compile(r"(WL|GCph|GCsp|V)")


class InvalidKeyError(Exception):
    def __init__(self, message="Invalid key", **kwargs):
        self.kwargs = kwargs
        self.message = message
        self.add_info_str = ", ".join([f"{key} = {value}" for key, value in self.kwargs.items()])
        super().__init__(message)

    def __repr__(self):
        return f"{self.message}: {self.add_info_str}"


class ClKey:
    key: "TRawClKey"
    kind: "str"
    match: "re.Match"

    def __init__(self, key: "Union[TRawClKey, ClKey]", infer_kind=True):
        self.key = None
        self.kind = None
        self.match = None

        if isinstance(key, ClKey):
            self.key = key.key
            self.kind = key.kind
            self.match = key.match
        else:
            self.key = key

            if infer_kind:
                self.kind = self.inferKind()
                if not self.isValidKind(self.kind):
                    raise InvalidKeyError(kind=self.kind)

    def __key(self):
        return self.toShortRawKey()

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self) -> "str":
        return repr(self.key)

    def __eq__(self, other: "ClKey") -> "bool":
        return isinstance(other, self.__class__) and self.equivalentTo(other)

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def copy(self):
        return ClKey(self.key)

    def equivalentTo(self, other: "ClKey") -> "bool":
        transf = self.transformTo(other.kind)

        return transf.key == other.key and transf.kind == other.kind

    @property
    def is_auto_correlation(self) -> "bool":
        p1, p2 = self.toParts()

        return p1 == p2

    @property
    def is_long(self) -> "bool":
        return self.kind == "long"

    @property
    def is_short(self) -> "bool":
        return self.kind == "short"

    @property
    def is_tuple(self) -> "bool":
        return self.kind == "alias"

    def toParts(self) -> "Tuple[str, str]":
        if self.kind is None:
            self.kind = self.inferKind()

        if self.kind == "long" or self.kind == "short":
            parts = self.match.groups()
        elif self.kind == "alias":
            parts = self.key
        else:
            raise InvalidKeyError(kind=self.kind)

        return parts

    @classmethod
    def fromParts(cls, p1: "str", p2: "str", kind: "str") -> "ClKey":
        if not cls.isValidKind(kind):
            raise InvalidKeyError(kind=kind)
        if kind == "long":
            key = f"{p1}_{p2}"
        elif kind == "short":
            key = f"{p1}{p2}"
        elif kind == "alias":
            key = (p1, p2)
        else:
            raise InvalidKeyError(kind=kind)

        cl_key = cls(key)
        if cl_key.kind != kind:
            raise ValueError(f"inferred kind {cl_key.kind} is not equal to passed {kind}")

        return cl_key

    @classmethod
    def fromFisherStringRepr(cls, str_repr: "str"):
        auto_match = alias_probe_name_regex.match(str_repr)
        if auto_match:
            p = auto_match.groups()[0]
            key = cls.fromParts(p, p, kind="alias")
        else:
            pattern = alias_probe_name_regex.pattern
            xc_regex = re.compile(r"XC\(%s,%s\)" % (pattern, pattern))
            xc_match = xc_regex.match(str_repr)
            if not xc_match:
                raise Exception(f"Invalid string repr for ClKey {str_repr}")
            p1, p2 = xc_match.groups()
            key = cls.fromParts(p1, p2, kind="alias")

        return key

    def inferKind(self) -> "str":
        if isinstance(self.key, str):
            long_match = long_cl_key_regex.match(self.key)
            short_match = short_cl_key_regex.match(self.key)
            if long_match:
                kind = "long"
                self.match = long_match
            elif short_match:
                kind = "short"
                self.match = short_match
            else:
                raise KeyError(f"Invalid key {self.key}")
        elif isinstance(self.key, tuple):
            if len(self.key) != 2:
                raise KeyError(f"Invalid key {self.key}")
            kind = "alias"
            p1, p2 = self.key
            if not alias_probe_name_regex.match(p1):
                raise KeyError(f"{p1} is not a valid probe alias")
            if not alias_probe_name_regex.match(p2):
                raise KeyError(f"{p2} is not a valid probe alias")
        else:
            raise KeyError(f"Invalid key {self.key}")

        return kind

    @staticmethod
    def isValidKind(kind) -> "bool":
        return kind in {"long", "short", "alias"}

    def getReverseKey(self) -> "ClKey":
        p1, p2 = self.toParts()

        return ClKey.fromParts(p2, p1, kind=self.kind)

    def transformTo(self, new_kind: "str") -> "ClKey":
        if new_kind == self.kind:
            transformed = copy.deepcopy(self)
        else:
            if not self.isValidKind(new_kind):
                raise InvalidKeyError(kind=new_kind)
            old_kind = self.kind
            old_p1, old_p2 = self.toParts()
            transformation_key = f"{old_kind}_to_{new_kind}"
            new_p1, new_p2 = _TRANSL_MAPPINGS[transformation_key][old_p1], _TRANSL_MAPPINGS[transformation_key][old_p2]
            transformed = self.fromParts(new_p1, new_p2, kind=new_kind)

        return transformed

    def toLongKey(self) -> "ClKey":
        return self.transformTo(new_kind="long")

    def toLongRawKey(self) -> "str":
        return self.toLongKey().key

    def toShortKey(self) -> "ClKey":
        return self.transformTo(new_kind="short")

    def toShortRawKey(self) -> "str":
        return self.toShortKey().key

    def toAliasKey(self) -> "ClKey":
        return self.transformTo(new_kind="alias")

    def toTupleRawKey(self) -> "Tuple[str, str]":
        return self.toAliasKey().key

    def toFisherStringRepr(self, kind: "str" = "alias") -> "str":
        key = self.transformTo(kind) if kind != self.kind else self
        p1, p2 = key.toParts()
        if p1 == p2:
            str_repr = p1
        else:
            str_repr = f"XC({p1},{p2})"

        return str_repr


class SmartKeyDict(GenericDictInterface[ClKey, np.ndarray]):
    def __init__(self, data_dict: "Dict" = None):
        super(SmartKeyDict, self).__init__()
        if isinstance(data_dict, dict):
            self._base_dict.update({ClKey(raw_key): data_dict[raw_key] for raw_key in data_dict})

    def reversed_key_transform(self, X: "np.ndarray") -> "np.ndarray":
        return X

    def searchAlias(self, cl_key: "ClKey"):
        equiv_key = None
        for key in self.keys():
            if key.equivalentTo(cl_key):
                equiv_key = key
                break

        return equiv_key

    def __getitem__(self, cl_key: "Union[ClKey, TRawClKey]"):
        if not isinstance(cl_key, ClKey):
            cl_key = ClKey(cl_key)
        try:
            return self._base_dict[cl_key]
        except KeyError:
            rev_key = cl_key.getReverseKey()

            return self.reversed_key_transform(self._base_dict[rev_key])


class ClSmartKeyDict(SmartKeyDict):
    def reversed_key_transform(self, X: "np.ndarray") -> "np.ndarray":
        return np.transpose(X, axes=(0, 2, 1))


class ProbeName:
    name: "str"
    kind: "str"

    def __init__(self, name: "str"):
        self.name = name
        self.kind = None

        self.inferKind()

    def __eq__(self, other: "ProbeName") -> "bool":
        return self.equivalentTo(other)

    def __repr__(self) -> "str":
        return self.name

    @property
    def is_long(self):
        return self.name in PROBES_LONG_NAMES

    @property
    def is_short(self):
        return self.name in PROBES_SHORT_NAMES

    @property
    def is_alias(self):
        return self.name in PROBES_ALIAS_NAMES

    def inferKind(self):
        if self.is_long:
            self.kind = "long"
        elif self.is_short:
            self.kind = "short"
        elif self.is_alias:
            self.kind = "alias"
        else:
            raise ValueError(f"Invalid probe name {self.name}")

    def transformTo(self, kind: "str") -> "ProbeName":
        if kind == self.kind:
            new_name = self
        else:
            trans_key = f"{self.kind}_to_{kind}"
            mapping = _TRANSL_MAPPINGS[trans_key]
            new_name = ProbeName(mapping[self.name])

        return new_name

    def toShort(self) -> "ProbeName":
        return self.transformTo("short")

    def toLong(self) -> "ProbeName":
        return self.transformTo("long")

    def toAlias(self) -> "ProbeName":
        return self.transformTo("alias")

    def equivalentTo(self, other: "ProbeName"):
        trasf = other.transformTo(self.kind)

        return self.name == trasf.name
