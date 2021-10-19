from typing import Set, Union, Iterable, Dict, Tuple
from pathlib import Path
import pandas as pd
import re
import numpy as np
import h5py
import logging

from seyfert.fisher import fisher_utils as fu
from seyfert.utils import tex_utils as txu
from seyfert.utils import general_utils as gu
from seyfert.file_io.ext_input import ISTInputReader


logger = logging.getLogger(__name__)


class FisherMatrixError(Exception):
    pass


class FisherMatrix:
    matrix: "pd.DataFrame"
    _name: "str"
    inverse: "pd.DataFrame"
    correlation: "pd.DataFrame"
    _cosmological_parameters: "Set[str]"
    _nuisance_parameters: "Set[str]"
    _already_sliced: "bool"
    marginalize_nuisance: "bool"

    def __init__(self,
                 matrix_df: pd.DataFrame = None,
                 name: "str" = None,
                 cosmological_parameters: "Iterable[str]" = None,
                 nuisance_parameters: "Iterable[str]" = None,
                 marginalize_nuisance: "bool" = True):
        self.matrix = matrix_df
        self._cosmological_parameters = None
        self._nuisance_parameters = None
        self.marginalize_nuisance = marginalize_nuisance
        self._name = name
        self.inverse = None
        self.correlation = None
        self._translator = txu.TeXTranslator()
        if cosmological_parameters is not None:
            self._cosmological_parameters = set(list(cosmological_parameters))
        if nuisance_parameters is not None:
            self._nuisance_parameters = set(list(nuisance_parameters))

    @property
    def physical_parameters(self) -> "Set[str]":
        return self._nuisance_parameters | self._cosmological_parameters

    @property
    def cosmological_parameters(self) -> "Set[str]":
        return self._cosmological_parameters

    @property
    def nuisance_parameters(self) -> "Set[str]":
        return self._nuisance_parameters

    @property
    def name(self) -> "str":
        return self._name

    @property
    def index(self):
        return self.matrix.index

    @property
    def columns(self):
        return self.matrix.columns

    @property
    def has_params_sets(self) -> "bool":
        return self.cosmological_parameters is not None and self.nuisance_parameters is not None

    @name.setter
    def name(self, new_name: "str") -> "None":
        if isinstance(new_name, str):
            self._name = new_name
        else:
            raise TypeError(f'Invalid name {new_name}')

    @property
    def tex_name(self) -> "str":
        try:
            return self._translator.replaceFisherNamesToTeX(self.name, use_aliases=False)
        except KeyError:
            return self.name

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame", name: "str" = None,
                       cosmological_parameters: "Iterable[str]" = None,
                       nuisance_parameters: "Iterable[str]" = None) -> "FisherMatrix":
        return cls(matrix_df=df, name=name,
                   cosmological_parameters=cosmological_parameters, nuisance_parameters=nuisance_parameters)

    @classmethod
    def from_ISTfile(cls, file: "Union[str, Path]", name: "str" = None,
                     cosmological_parameters: "Iterable[str]" = None,
                     nuisance_parameters: "Iterable[str]" = None) -> "FisherMatrix":
        reader = ISTInputReader()
        if name is None:
            name, df = reader.readISTFisherMatrixFile(file)
        else:
            _, df = reader.readISTFisherMatrixFile(file)

        if cosmological_parameters is None and nuisance_parameters is None:
            logger.warning(f"no cosmological or nuisance parameters passed, defaulting to only cosmological and no "
                           f"nuisance parameters for fisher {name}")
            cosmological_parameters = set(df.index)
            nuisance_parameters = set()

        return cls(matrix_df=df, name=name,
                   cosmological_parameters=cosmological_parameters, nuisance_parameters=nuisance_parameters)

    def copy(self) -> "FisherMatrix":
        return FisherMatrix(name=self.name, matrix_df=self.matrix.copy(deep=True),
                            marginalize_nuisance=self.marginalize_nuisance,
                            cosmological_parameters=self.cosmological_parameters.copy(),
                            nuisance_parameters=self.nuisance_parameters.copy())

    @staticmethod
    def zeroPadMatrices(F1: "FisherMatrix", F2: "FisherMatrix") -> "Tuple[FisherMatrix, FisherMatrix]":
        sum_nuis_pars = F1.nuisance_parameters | F2.nuisance_parameters
        sum_cosmo_pars = F1.cosmological_parameters | F2.cosmological_parameters
        sum_phys_pars = sum_cosmo_pars | sum_nuis_pars
        missing_pars_1 = sum_phys_pars - F1.physical_parameters
        missing_pars_2 = sum_phys_pars - F2.physical_parameters
        for missing_par in missing_pars_1:
            F1.matrix[missing_par] = 0
            F1.matrix.loc[missing_par] = 0
        for missing_par in missing_pars_2:
            F2.matrix[missing_par] = 0
            F2.matrix.loc[missing_par] = 0

        return F1, F2

    def __add__(self, other: "FisherMatrix") -> "FisherMatrix":
        F1, F2 = self.zeroPadMatrices(self.copy(), other.copy())
        sum_cosmo_pars = F1.cosmological_parameters | F2.cosmological_parameters
        sum_nuis_pars = F1.nuisance_parameters | F2.nuisance_parameters
        sum_matrix_df = F1.matrix + F2.matrix
        fisher_sum = FisherMatrix(name=f'{F1.name} + {F2.name}',
                                  matrix_df=sum_matrix_df,
                                  cosmological_parameters=sum_cosmo_pars,
                                  nuisance_parameters=sum_nuis_pars)
        return fisher_sum

    def __sub__(self, other: "FisherMatrix") -> "FisherMatrix":
        F1, F2 = self.zeroPadMatrices(self.copy(), other.copy())
        sum_cosmo_pars = F1.cosmological_parameters | F2.cosmological_parameters
        sum_nuis_pars = F1.nuisance_parameters | F2.nuisance_parameters
        sum_matrix_df = F1.matrix - F2.matrix
        fisher_sum = FisherMatrix(name=f'{F1.name} - {F2.name}',
                                  matrix_df=sum_matrix_df,
                                  cosmological_parameters=sum_cosmo_pars,
                                  nuisance_parameters=sum_nuis_pars)
        return fisher_sum

    def __eq__(self, other: "FisherMatrix") -> "bool":
        return all([np.all(self.matrix.sort_index(axis=0).sort_index(axis=1) ==
                           other.matrix.sort_index(axis=0).sort_index(axis=1)),
                    self.name == other.name,
                    self.physical_parameters == other.physical_parameters])

    def __repr__(self) -> "str":
        return f"{self.name}\n {repr(self.matrix)}"

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def checkAttributes(self) -> "None":
        if not isinstance(self.matrix, pd.DataFrame):
            raise TypeError('fisher matrix must be a dataframe')
        if not isinstance(self._name, str):
            raise TypeError('name must be a string')
        fisher_rows = set(self.matrix.index)
        fisher_cols = set(self.matrix.columns)
        if fisher_rows != fisher_cols:
            raise Exception(f'Fisher matrix is not square: \n'
                            f'rows: {", ".join(fisher_rows)}\n'
                            f'cols: {", ".join(fisher_cols)}')
        if not isinstance(self._cosmological_parameters, set):
            raise ValueError('cosmological parameters attribute must be a set')
        if not isinstance(self._nuisance_parameters, set):
            raise ValueError('nuisance parameters attribute must be a set')
        if fisher_rows != self.physical_parameters:
            raise ValueError(f'fisher matrix rows are not equal to the union of cosmological and nuisance parameters.\n'
                             f'Rows: {", ".join(fisher_rows)}\n'
                             f'Physical parameters: {", ".join(self.physical_parameters)}')

    def selectRelevantFisherSubMatrix(self):
        relevant_pars = self.cosmological_parameters | self.nuisance_parameters
        all_phys_pars = set(self.matrix.index)
        if not all_phys_pars.issuperset(relevant_pars):
            raise FisherMatrixError(f'physical parameters are not subset of matrix indices for fisher {self.name}. '
                                    f'The diff is: {", ".join(relevant_pars - all_phys_pars)}')
        pars_to_exclude = set()
        if not self.marginalize_nuisance:
            pars_to_exclude |= self._nuisance_parameters
            self._nuisance_parameters = set()
        relevant_pars -= pars_to_exclude
        self.matrix = self.matrix.loc[relevant_pars, relevant_pars]
        self.matrix.sort_index(axis=0, inplace=True)
        self.matrix.sort_index(axis=1, inplace=True)

    def fixParameters(self, params: "Iterable[str]") -> "None":
        params = set(params)
        missing = params - self.physical_parameters
        if missing:
            raise Exception(f"parameters {missing} are not present in the matrix")
        self._cosmological_parameters = self.cosmological_parameters - params
        self._nuisance_parameters = self.nuisance_parameters - params
        self.matrix = self.matrix.drop(params, axis=0).drop(params, axis=1)

    def evaluateInverse(self):
        self.inverse = fu.invert_dataframe(self.matrix)

    def getFigureOfMerit(self, par_1: "str" = 'w0', par_2: "str" = 'wa') -> "float":
        pars_subset = {par_1, par_2}
        sub_inv_matrix = self.inverse.loc[pars_subset, pars_subset]

        return 1. / np.sqrt(np.linalg.det(sub_inv_matrix))

    def evaluateCorrelationMatrix(self) -> "None":
        sigmas = np.sqrt(np.diag(self.inverse))
        sigma_row_df = pd.DataFrame(data=np.repeat(sigmas[np.newaxis, :], self.matrix.shape[0], axis=0),
                                    index=self.matrix.index,
                                    columns=self.matrix.columns)
        sigma_col_df = pd.DataFrame(data=np.repeat(sigmas[:, np.newaxis], self.matrix.shape[1], axis=1),
                                    index=self.matrix.index,
                                    columns=self.matrix.columns)
        self.correlation = self.inverse / (sigma_row_df * sigma_col_df)
        self.correlation = self.correlation.loc[self._cosmological_parameters, self._cosmological_parameters]
        self.correlation.sort_index(axis=0, inplace=True)
        self.correlation.sort_index(axis=1, inplace=True)

    def marginalizeParameters(self, params: "Iterable[str]", ret_copy: "bool" = True) -> "Union[None, FisherMatrix]":
        params = set(params)
        all_params = set(self.index)
        missing = params - all_params
        if missing:
            raise FisherMatrixError(f"cannot marginalize on missing parameters {missing}")

        self.evaluateInverse()
        to_maintain = all_params - params
        sliced_inverse = self.inverse.loc[to_maintain, to_maintain]
        if ret_copy:
            fmat = self.copy()
            fmat._cosmological_parameters = fmat.cosmological_parameters - params
            fmat._nuisance_parameters = fmat.nuisance_parameters - params
            fmat.inverse = sliced_inverse
            marg_fisher = fu.invert_dataframe(sliced_inverse)
            fmat.matrix = marg_fisher.sort_index(axis=0).sort_index(axis=1)

            return fmat
        else:
            self._cosmological_parameters = self.cosmological_parameters - params
            self._nuisance_parameters = self.nuisance_parameters - params
            self.inverse = sliced_inverse
            marg_fisher = fu.invert_dataframe(sliced_inverse)
            self.matrix = marg_fisher.sort_index(axis=0).sort_index(axis=1)

            return None

    def marginalizeNuisance(self, ret_copy: "bool" = True) -> "Union[None, FisherMatrix]":
        if self.marginalize_nuisance:
            retval = self.marginalizeParameters(self.nuisance_parameters, ret_copy=ret_copy)
        else:
            logger.warning('marginalize nuisance flag is False but you explicitly '
                           'called the marginalize nuisance method. Not doing anything')
            retval = self.copy() if ret_copy else None

        return retval

    def getParameterSigma(self, name: "str") -> "float":
        if self.inverse is None:
            raise ValueError("cannot compute marginalized errors when inverse is None")
        else:
            return np.sqrt(self.inverse.loc[name, name])

    def getMarginalizedErrors(self, fom: "bool" = True, only_cosmo: "bool" = True) -> "pd.Series":
        if self.inverse is None:
            self.evaluateInverse()
        errs = pd.Series([self.getParameterSigma(name) for name in self.index], index=self.index)
        if only_cosmo:
            errs = errs.loc[self.cosmological_parameters]
        if fom and {'w0', 'wa'}.issubset(set(self.index)):
            errs.loc['FoM'] = self.getFigureOfMerit()

        errs.name = self.name

        return errs.sort_index()

    def getUnmarginalizedErrors(self, only_cosmo: "bool" = True) -> "pd.Series":
        errs = pd.Series(1 / np.sqrt(np.diag(self.matrix)), index=self.index)
        if only_cosmo:
            errs = errs.loc[self.cosmological_parameters]

        return errs

    def addPriorToParameters(self, fiducials_dict: "Dict[str, float]", frac_delta: "float"):
        prior_fisher = pd.DataFrame(np.zeros(self.matrix.shape), index=self.index, columns=self.columns)
        for name, value in fiducials_dict.items():
            sigma_prior = frac_delta * value
            prior_fisher.loc[name, name] = 1 / (sigma_prior**2)

        self.matrix = self.matrix.add(prior_fisher, fill_value=0)

    def getFilename(self) -> "str":
        sub_dict = {
            key: "_" for key in ["[", "]", "(", ")", " "]
        }
        sub_dict[','] = "-"
        suffix = gu.multiple_regex_replace(self.name, sub_dict=sub_dict).strip("_")
        suffix = re.sub(r"_+", "_", suffix)

        return f"fisher_{suffix}.hdf5"

    def writeToFile(self, outdir: "Union[str, Path]" = None, outfile: "Union[str, Path]" = None,
                    overwrite=False, skip_if_exists=False) -> "None":
        if outfile is None:
            if outdir is not None:
                outfile = Path(outdir) / self.getFilename()
            else:
                raise FisherMatrixError("At least one fo outdir and outfile must be provided")
        else:
            outfile = Path(outfile)

        if outfile.exists():
            if skip_if_exists:
                logger.warning(f"file {outfile} already exists, skipping without overwriting")
                return
            else:
                if overwrite:
                    outfile.unlink()
                else:
                    raise FileExistsError(f"{outfile}")

        hf = h5py.File(outfile, 'w')
        str_dtype = h5py.special_dtype(vlen=str)
        hf.create_dataset(name="name", data=self.name, dtype=str_dtype)
        hf.create_dataset(name="indices", data=np.array(self.matrix.index, dtype=str_dtype), dtype=str_dtype)
        hf.create_dataset(name="fisher_matrix", data=self.matrix.to_numpy(dtype=np.float64), dtype='f8')
        hf.create_dataset(name="cosmological_pars", data=np.array(list(self.cosmological_parameters), dtype=str_dtype),
                          dtype=str_dtype)
        if len(self.nuisance_parameters) > 0:
            hf.create_dataset(name="nuisance_pars", data=np.array(list(self.nuisance_parameters), dtype=str_dtype),
                              dtype=str_dtype)
        hf.close()

    def loadFromFile(self, file: "Union[Path, str]" = None, file_ext: "str" = 'hdf5') -> "None":
        if file_ext == 'hdf5':
            hf = h5py.File(file, 'r')
            self.name = hf["name"][()]
            try:
                self._nuisance_parameters = set(hf["nuisance_pars"][()])
            except KeyError:
                self._nuisance_parameters = set()
            self._cosmological_parameters = set(hf["cosmological_pars"][()])
            indices = hf["indices"][()]
            self.matrix = pd.DataFrame(data=hf["fisher_matrix"][()], index=indices, columns=indices)
            hf.close()

        elif file_ext == 'csv':
            matrix_df = pd.read_csv(file, index_col=0)
            self.matrix = matrix_df
            if self.cosmological_parameters is None:
                logger.warning(f'cosmological parameters not set')
            if self.nuisance_parameters is None:
                logger.warning(f'nuisance parameters not set')
            if self._name is None:
                logger.warning(f'fisher name not set')
        else:
            raise Exception(f'Unsupported file kind {file_ext}')

    @classmethod
    def fromHDF5(cls, file: "Union[Path, str]") -> "FisherMatrix":
        f = cls()
        f.loadFromFile(file)

        return f

    @classmethod
    def fromCSV(cls, file: "Union[Path, str]", cosmological_parameters: "Iterable[str]" = None,
                nuisance_parameters: "Iterable[str]" = None) -> "FisherMatrix":
        df = pd.read_csv(file, index_col=0)

        return cls.from_dataframe(df, name=Path(file).stem, cosmological_parameters=cosmological_parameters,
                                  nuisance_parameters=nuisance_parameters)


def fromHDF5(file: "Union[Path, str]") -> "FisherMatrix":
    return FisherMatrix.fromHDF5(file)
