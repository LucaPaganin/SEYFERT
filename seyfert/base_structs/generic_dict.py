from abc import ABC
from typing import Dict, Iterator, TypeVar, Mapping, List

_KeyType = TypeVar('_KeyType')
_ValueType = TypeVar('_ValueType')


class GenericDictInterface(Mapping[_KeyType, _ValueType], ABC):
    _base_dict: "Dict[_KeyType, _ValueType]"

    def __init__(self, data_dict: "Dict[_KeyType, _ValueType]" = None):
        self._base_dict = {}
        if isinstance(data_dict, dict):
            self._base_dict = data_dict

    @property
    def base_dict(self) -> "Dict":
        return self._base_dict

    def keys(self) -> "List[_KeyType]":
        return list(super(GenericDictInterface, self).keys())

    def __setitem__(self, k: "_KeyType", v: "_ValueType") -> "None":
        self._base_dict[k] = v

    def __delitem__(self, k: "_KeyType") -> "None":
        del self._base_dict[k]

    def __getitem__(self, k: "_KeyType") -> "_ValueType":
        return self._base_dict[k]

    def __len__(self) -> "int":
        return len(self._base_dict)

    def __iter__(self) -> "Iterator[_KeyType]":
        return iter(self._base_dict)

    def __eq__(self, other: "GenericDictInterface") -> "bool":
        return self._base_dict == other._base_dict

    def __repr__(self):
        return self._base_dict.__repr__()

    def update(self, other: Dict) -> "None":
        self._base_dict.update(other)


class DictLike(dict):
    def __init__(self):
        super().__init__()

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

