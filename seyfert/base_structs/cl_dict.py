import numpy as np
from typing import Dict, Tuple, Union, List, Any

from seyfert.base_structs.generic_dict import GenericDictInterface
from seyfert.utils import general_utils as gu
from seyfert import PROBE_LONG_TO_ALIAS

_fullname_to_alias_map = {key: value for key, value in PROBE_LONG_TO_ALIAS.items()}

_alias_to_fullname_map = {value: key for key, value in _fullname_to_alias_map.items()}


class FlexDict(GenericDictInterface[str, np.ndarray]):
    _base_dict: "Dict[str, np.ndarray]"

    def __init__(self, data_dict: "Dict[str, np.ndarray]" = None):
        super(FlexDict, self).__init__(data_dict=data_dict)

    def transform(self, X: "np.ndarray") -> "np.ndarray":
        return X

    @property
    def base_dict(self) -> "Dict[str, np.ndarray]":
        return self._base_dict

    @staticmethod
    def alias_to_fullname(p: "str") -> "str":
        return _alias_to_fullname_map[p]

    @staticmethod
    def fullname_to_alias(p: "str") -> "str":
        return _fullname_to_alias_map[p]

    @property
    def tuple_keys(self) -> "List[Tuple[str, str]]":
        return [self.str_key_to_tuple_key(key) for key in self.keys()]

    def __getitem__(self, key: "Union[str, Tuple[str, str]]") -> "np.ndarray":
        try:
            return self._base_dict[key]
        except KeyError:
            if isinstance(key, str):
                rev_key = gu.reverse_probes_comb_key(key)

                return self.transform(self._base_dict[rev_key])

            elif isinstance(key, tuple):
                p1, p2 = key
                p1 = self.alias_to_fullname(p1)
                p2 = self.alias_to_fullname(p2)
                try:
                    return self._base_dict[gu.get_probes_combination_key(p1, p2)]
                except KeyError:
                    return self.transform(self._base_dict[gu.get_probes_combination_key(p2, p1)])

            else:
                raise TypeError(f"Unrecognized key type {type(key)}")

    def tuple_key_to_str_key(self, tpl_key: "Tuple") -> "str":
        p1, p2 = tpl_key
        p1 = self.alias_to_fullname(p1)
        p2 = self.alias_to_fullname(p2)

        key = gu.get_probes_combination_key(p1, p2)

        if key in self:
            return key
        else:
            return gu.reverse_probes_comb_key(key)

    def str_key_to_tuple_key(self, str_key: "str") -> "Tuple":
        p1, p2 = gu.get_probes_from_comb_key(str_key)
        p1 = self.fullname_to_alias(p1)
        p2 = self.fullname_to_alias(p2)

        return p1, p2


class ClDict(FlexDict):
    def __init__(self, data_dict: "Dict[str, np.ndarray]" = None):
        super(ClDict, self).__init__(data_dict=data_dict)

    def transform(self, X: "np.ndarray") -> "np.ndarray":
        return np.transpose(X, axes=(0, 2, 1))


class TupleSmartDict(GenericDictInterface[Tuple[str, str], np.ndarray]):
    def transform(self, X: "np.ndarray") -> "np.ndarray":
        return X

    def __getitem__(self, item):
        try:
            return self._base_dict[item]
        except KeyError:
            p1, p2 = item

            return self.transform(self._base_dict[(p2, p1)])


class ClSmartDict(TupleSmartDict):
    def transform(self, X: "np.ndarray") -> "np.ndarray":
        return np.transpose(X, axes=(0, 2, 1))
