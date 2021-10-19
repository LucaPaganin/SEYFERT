from typing import TYPE_CHECKING, List, Union, Set, Tuple
import re

from seyfert.utils.shortcuts import ClKey

if TYPE_CHECKING:
    from seyfert.utils.shortcuts import TRawClKey


class ClDataVector:
    """
    Class for managing Cl datavector.
    """
    name: "str"
    entries: "List[ClKey]"

    # noinspection PyTypeChecker
    def __init__(self, input_data: "Union[List[TRawClKey], ClDataVector]"):
        self.name = None
        self.entries = None
        self.involved_probes = None

        if isinstance(input_data, ClDataVector):
            self.entries = input_data.entries
        else:
            self.entries = [ClKey(entry) for entry in input_data]

        self.name = self.getNameFromEntries()
        self.involved_probes = self.getInvolvedAutoCorrelationProbes()

    def __repr__(self):
        return f"{self.name}: {repr(self.entries)}"

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, item):
        return self.entries[item]

    def __len__(self):
        return len(self.entries)

    def __eq__(self, other: "ClDataVector") -> "bool":
        return all(entry1 == entry2 for entry1, entry2 in zip(self, other))

    @property
    def is_single_autocorrelation(self) -> "bool":
        return len(self) == 1 and self[0].is_auto_correlation

    def getRawEntries(self) -> "List[TRawClKey]":
        return [entry.key for entry in self]

    def transformEntriesTo(self, kind: "str"):
        new_entries = [entry.transformTo(kind) for entry in self]
        self.entries = new_entries
        self.name = self.getNameFromEntries()

    @classmethod
    def fromName(cls, name: "str"):
        name = name.replace("[", "").replace("]", "")
        entries = [ClKey.fromFisherStringRepr(p) for p in name.split("+")]

        return cls(entries)

    @classmethod
    def fromString(cls, s: "str") -> "ClDataVector":
        name, elements = s.split(":")
        elements = re.findall(r"([a-z]{4})", elements)

        return cls([ClKey(el).toTupleRawKey() for el in elements])

    @classmethod
    def fromBriefString(cls, s: "str"):
        entries = [ClKey(p).toTupleRawKey() for p in s.split("_")]

        return cls(entries)

    def toString(self) -> "str":
        return str(ClDataVector([entry.toShortRawKey() for entry in self]))

    def toBriefString(self) -> "str":
        shv = self.toShortKeysVector()

        return "_".join(entry.key for entry in shv)

    def toShortKeysVector(self) -> "ClDataVector":
        return ClDataVector([entry.toShortRawKey() for entry in self])

    def toTupleKeysVector(self) -> "ClDataVector":
        return ClDataVector([entry.toTupleRawKey() for entry in self])

    def toRawTuples(self) -> "List[Tuple[str, str]]":
        return [k.toTupleRawKey() for k in self]

    def getNameFromEntries(self) -> "str":
        return "[" + "+".join(entry.toFisherStringRepr() for entry in self) + "]"

    def getInvolvedAutoCorrelationProbes(self) -> "Set[str]":
        probes = set()
        for entry in self:
            if entry.is_auto_correlation:
                p1, _ = entry.toTupleRawKey()
                probes.add(p1)

        return probes
