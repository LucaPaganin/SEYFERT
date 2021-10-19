import numpy as np
import h5py
from functools import partial
import datetime
import logging
from abc import ABC, abstractmethod
import os
from pathlib import Path
from seyfert import VERSION
from seyfert.utils import general_utils as gu
from typing import Union, Dict, Callable


logger = logging.getLogger(__name__)


class H5FileModeError(Exception):
    pass


class H5KeyError(Exception):
    pass


class AbstractH5FileIO(ABC):
    HDF5_ARRAY_OPTS = {'dtype': 'f8', 'compression': 'gzip', 'compression_opts': 9}
    DATASET_DTYPES = (int, float, str, np.ndarray, np.bool_, np.int64, np.float64)
    _h5file: "h5py.File"
    _root_group: "h5py.Group"
    file_path: "Path"
    builder_func: "Callable"
    build_data: "Dict"
    root: "h5py.Group"

    def __init__(self, root: "str" = None):
        self._h5file = None
        self.file_path = None
        self.file_mode = None
        self._root_group = None
        self._root_path = root
        self.out_group_name = None
        self.builder_func = None
        self.build_data = {}
        self._attrs_excluded_from_equality = {'_h5file', '_root_group', '_root_path'}

    @property
    def root(self) -> "h5py.Group":
        return self._root_group

    @property
    def root_path(self) -> "str":
        return self.root.name

    @property
    def attrs(self) -> "h5py.AttributeManager":
        return self.root.attrs

    def __getitem__(self, item: "str") -> "Union[h5py.Dataset, h5py.Group]":
        if not isinstance(item, str):
            raise TypeError(f'expected str as key type, got {type(item)}')
        return self.root[item]

    def keys(self):
        return self.root.keys()

    def openFile(self, file: "Union[str, Path]", mode: "str", root: "str" = '/') -> None:
        if not isinstance(file, (str, Path)):
            raise TypeError(f'expected str or Path as file, got {type(file)}')
        if mode not in {'r', 'r+', 'w', 'w-', 'x', 'a'}:
            raise ValueError(f'invalid mode {mode}. Options: r, w, a')
        if not isinstance(root, str):
            raise TypeError(f'expected str as root, got {type(root)}')
        self.file_path = Path(file)
        self.file_mode = mode
        if self._root_path is not None and self._root_path != '/':
            logger.warning(f'root path already specified as {self._root_path}, overwriting with {root}')
        self._root_path = root
        base_metadata = {}
        tc_iso = datetime.datetime.now().isoformat()
        if self.file_path.exists():
            if mode == 'w':
                raise FileExistsError(f'file {file} already exists')
            elif mode != 'r' and root == '/':
                base_metadata['LastUpdate'] = tc_iso
        else:
            if root == '/':
                base_metadata.update({
                    'CreationDate': tc_iso,
                    'LastUpdate': tc_iso,
                    'Description': "",
                    'Author': os.getenv("USER", default=""),
                    'CodeVersion': VERSION
                })

        self._h5file = h5py.File(self.file_path, mode=mode)
        try:
            self._root_group = self._h5file[self._root_path]
        except KeyError:
            if mode != 'r':
                self._root_group = self._h5file.create_group(name=self._root_path)
            else:
                raise H5FileModeError(f'cannot create root {root} on readonly file')
        if mode != 'r':
            self.root.attrs.update(base_metadata)

    def closeFile(self) -> None:
        self._h5file.close()

    def readBuildingData(self) -> "None":
        pass

    def load(self, file: "Union[str, Path]", root: "str" = '/') -> "object":
        self.openFile(file, mode='r', root=root)
        self.readBuildingData()
        obj = self.builder_func(**self.build_data)
        self.writeToObject(obj)
        self.closeFile()
        return obj

    def save(self, obj: "object", file: "Union[str, Path]", root: "str" = '/'):
        mode = 'w' if not Path(file).exists() else 'a'
        self.openFile(file, mode=mode, root=root)
        self.writeObjectToFile(obj)
        self.closeFile()

    @abstractmethod
    def writeToObject(self, obj: "object") -> "None":
        if self.file_path is None:
            raise RuntimeError('source file is not open')
        if self.file_mode not in {'r', 'r+'}:
            raise RuntimeError('file mode must be r or r+ for writing to object')

    @abstractmethod
    def writeObjectToFile(self, obj: "object") -> "None":
        pass

    def populateObject(self, obj: "object", base_grp_path: "str" = None) -> "None":
        base = self.root[base_grp_path] if isinstance(base_grp_path, str) else self.root
        for key, item in base.items():
            if isinstance(item, h5py.Dataset):
                setattr(obj, key, item[()])

    def findDataset(self, name_wanted: "str") -> "h5py.Dataset":
        def visitor_func(name: "str", node: "object", wanted: "str"):
            if isinstance(node, h5py.Dataset) and name.endswith(wanted):
                return name, node
        return self._h5file.visititems(partial(visitor_func, wanted=name_wanted))

    def findGroup(self, name_wanted: "str") -> "h5py.Group":
        def visitor_func(name: "str", node: "object", wanted: "str"):
            if isinstance(node, h5py.Group) and (name == wanted or name.endswith(wanted)):
                return name, node
        return self._h5file.visititems(partial(visitor_func, wanted=name_wanted))

    def createGroup(self, name: "str", base_grp: "h5py.Group" = None, try_link_first: "bool" = False) -> "h5py.Group":
        base = base_grp if base_grp is not None else self.root
        # logger.info(f'Creating group {name} into group {base.name}')
        if try_link_first:
            found = self.findGroup(name)
            if found is not None:
                grp_name, grp = found
                entry = h5py.SoftLink(grp_name)
                base[name] = entry
            else:
                entry = self._create_group(name=name, base_grp=base_grp)
        else:
            entry = self._create_group(name=name, base_grp=base_grp)
        return entry

    def createDataset(self, name: "str", data, attrs: "Dict" = None,
                      base_grp: "h5py.Group" = None,
                      try_link_first: "bool" = False) -> "Union[h5py.Dataset, h5py.SoftLink]":
        base = base_grp if base_grp is not None else self.root

        if try_link_first:
            found = self.findDataset(name)
            if found is not None:
                found_name, _ = found
                base[name] = h5py.SoftLink(found_name)
                entry = base[name]
            else:
                entry = self._create_dataset(name=name, data=data, attrs=attrs, base_grp=base_grp)
        else:
            entry = self._create_dataset(name=name, data=data, attrs=attrs, base_grp=base_grp)
        return entry

    def _create_group(self, name: "str", base_grp: "h5py.Group" = None) -> "h5py.Group":
        base = base_grp if base_grp is not None else self.root
        return base.create_group(name)

    def _create_dataset(self, name: "str", data, attrs: "Dict" = None, base_grp: "h5py.Group" = None) -> "h5py.Dataset":
        base = base_grp if base_grp is not None else self.root
        if not isinstance(name, str):
            raise TypeError(f'name must be str, not {type(name)}')
        if not isinstance(data, self.DATASET_DTYPES):
            raise TypeError(f'name {name}: invalid data type {type(data)}')
        if isinstance(data, np.ndarray):
            dataset = base.create_dataset(name=name, data=data, **self.HDF5_ARRAY_OPTS)
        else:
            dataset = base.create_dataset(name=name, data=data)
        if attrs is not None:
            dataset.attrs.update(attrs)
        return dataset

    def createDatasets(self, data_dict: "Dict", base_grp: "h5py.Group" = None) -> None:
        base = base_grp if base_grp is not None else self.root
        for name, data in data_dict.items():
            self.createDataset(name=name, data=data, base_grp=base)

    def __getitem__(self, item) -> Union[h5py.Group, h5py.Dataset]:
        return self.root[item]

    def __eq__(self, other: "AbstractH5FileIO") -> bool:
        return gu.compare_objects(self, other, self._attrs_excluded_from_equality)
