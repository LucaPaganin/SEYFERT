"""
Module hosting classes for computing fisher matrix.
"""
import itertools
from typing import TYPE_CHECKING, Union, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import logging
import time

from seyfert.fisher.datavector import ClDataVector
from seyfert.utils.shortcuts import ClKey, SmartKeyDict, ClSmartKeyDict
from seyfert.utils import array_utils
from seyfert.fisher import fisher_utils as fu
from seyfert.utils import formatters
from seyfert.fisher.delta_cl import DeltaClCollection
from seyfert.fisher import linalg_tools

if TYPE_CHECKING:
    from seyfert.derivatives.cl_derivative import ClDerivativeCollector
    from seyfert.cosmology.parameter import PhysicalParametersCollection, PhysicalParameter

logger = logging.getLogger(__name__)

TClDictLike = Union[Dict[Any, np.ndarray], ClSmartKeyDict, SmartKeyDict]


class MultipoleError(Exception):
    pass


def mask_cl_blocks_to_common_ells(l_dict: "TClDictLike",
                                  cl_dict: "TClDictLike") -> "Tuple[np.ndarray, TClDictLike]":
    ells = array_utils.compute_arrays_intersection1d(list(l_dict.values()))
    masked_cl_dict = {}
    for key, block in cl_dict.items():
        mask = np.isin(l_dict[key], ells)
        masked_cl_dict[key] = block[mask]

    return ells, masked_cl_dict


class ClCovarianceMatrix:
    def __init__(self, data_vector: "ClDataVector", ells: "np.ndarray" = None):
        self.data_vector = data_vector
        self.common_ells = ells
        self.ell_blocks = {}
        self.cov_blocks = {}
        self.matrix = None
        self.inverse = None

    def evaluateInverse(self):
        self.inverse = np.linalg.inv(self.matrix)

    def buildMatrix(self):
        cov_rows = []
        for AB in self.data_vector:
            row_blocks = []
            for CD in self.data_vector:
                row_blocks.append(self.cov_blocks[(AB, CD)])

            cov_rows.append(row_blocks)

        self.matrix = np.block(cov_rows)

    def evaluateAllBlocks(self, delta_cls: "DeltaClCollection"):
        smart_ells = SmartKeyDict(delta_cls.ell_dict)
        smart_delta_cls = ClSmartKeyDict(delta_cls.delta_c_lij_array_dict)
        for row, AB in enumerate(self.data_vector):
            for col, CD in enumerate(self.data_vector):
                if row <= col:
                    ells, cov_block = self.computeCovBlock(AB, CD, smart_ells, smart_delta_cls)
                    self.ell_blocks[(AB, CD)] = ells
                    self.cov_blocks[(AB, CD)] = cov_block
                else:
                    self.ell_blocks[(AB, CD)] = self.ell_blocks[(CD, AB)]
                    self.cov_blocks[(AB, CD)] = np.transpose(self.cov_blocks[(CD, AB)], axes=(0, 2, 1))

    def maskCovBlocksToCommonElls(self):
        common_ells, self.cov_blocks = mask_cl_blocks_to_common_ells(self.ell_blocks, self.cov_blocks)
        self.common_ells = common_ells

    @staticmethod
    def computeCovBlock(AB: "ClKey", CD: "ClKey",
                        smart_ells: "SmartKeyDict", smart_delta_cls: "ClSmartKeyDict") -> "Tuple[np.ndarray, np.ndarray]":
        A, B = AB.toTupleRawKey()
        C, D = CD.toTupleRawKey()

        sel_block_keys = [(A, C), (B, D), (A, D), (B, C)]
        sel_ells = {}
        sel_delta_cls = {}
        for sel_key in sel_block_keys:
            sel_ells[sel_key] = smart_ells[sel_key]
            sel_delta_cls[sel_key] = smart_delta_cls[sel_key]

        ells, masked_delta_cls = mask_cl_blocks_to_common_ells(sel_ells, sel_delta_cls)

        enum_AB = linalg_tools.get_enumeration_map(smart_delta_cls[(A, B)].shape[1:], A == B)
        enum_CD = linalg_tools.get_enumeration_map(smart_delta_cls[(C, D)].shape[1:], C == D)
        n_ell = len(ells)
        n_rows = len(enum_AB)
        n_cols = len(enum_CD)

        cov_block = np.zeros((n_ell, n_rows, n_cols))

        for (i, j), row in enum_AB.items():
            for (k, m), col in enum_CD.items():
                cov_block[:, row, col] = masked_delta_cls[(A, C)][:, i, k] * masked_delta_cls[(B, D)][:, j, m] + \
                                         masked_delta_cls[(A, D)][:, i, m] * masked_delta_cls[(B, C)][:, j, k]

        cov_block *= np.expand_dims(1 / (2*ells + 1), (1, 2))

        return ells, cov_block


class FisherEvaluator:
    data_vector: "ClDataVector"
    covariance: "ClCovarianceMatrix"
    free_physical_parameters: "Dict[str, PhysicalParameter]"

    # noinspection PyTypeChecker
    def __init__(self, datavector_keys: "Union[List[str], ClDataVector]"):
        self.data_vector = ClDataVector(datavector_keys)
        self.free_physical_parameters = None
        self.free_nuisance_parameters = None
        self.free_cosmological_parameters = None
        self.covariance = None

    @property
    def fisher_name(self) -> "str":
        return self.data_vector.name

    @property
    def involved_probes(self):
        return self.data_vector.involved_probes

    @property
    def ells(self):
        return self.covariance.common_ells

    def computeFisherMatrix(self, phys_pars: "PhysicalParametersCollection",
                            dcoll_dvar_dict: "Dict[str, ClDerivativeCollector]",
                            delta_cls: "DeltaClCollection", return_f_ell=False,
                            auto_f_ells: "Dict[str, pd.DataFrame]" = None) -> "Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]":
        self.selectFreeParameters(phys_pars)
        logger.info(f"Building and inverting covariance matrix {self.fisher_name}...")
        t0 = time.time()
        self.buildAndInvertCovarianceMatrix(delta_cls)
        tf = time.time()
        logger.info(f"Elapsed time {formatters.string_time_format(tf-t0)}")

        logger.info(f"Building vectorized Cl derivatives...")
        t0 = time.time()
        vec_dcl_ells, vec_dcls_dict = self.buildVectorizedClDerivatives(dcoll_dvar_dict)
        tf = time.time()
        logger.info(f"Elapsed time {formatters.string_time_format(tf - t0)}")

        self.checkClDerivativeMultipolesConsistency(vec_dcl_ells)

        logger.info("Computing fisher elements...")
        t0 = time.time()
        f_ell_df = self.computeFisherOfMultipole(vec_dcls_dict=vec_dcls_dict)
        fisher_matrix_df = self.sumFisherOverMultipoles(f_ell_df=f_ell_df, ells=self.ells)
        tf = time.time()
        logger.info(f"Elapsed time {formatters.string_time_format(tf - t0)}")

        if not self.data_vector.is_single_autocorrelation:
            if auto_f_ells is None:
                raise ValueError("auto_f_ells parameter is needed for fisher computed from more than one probe")
            fisher_matrix_df = self.addInvolvedProbesContribution(f_mat_df=fisher_matrix_df, auto_f_ells=auto_f_ells)

        if return_f_ell:
            result = fisher_matrix_df, f_ell_df
        else:
            result = fisher_matrix_df

        return result

    def selectFreeParameters(self, phys_pars: "PhysicalParametersCollection"):
        self.free_cosmological_parameters = phys_pars.free_cosmological_parameters
        self.free_nuisance_parameters = {}
        for entry in self.data_vector:
            p1, p2 = entry.toLongKey().toParts()
            self.free_nuisance_parameters.update(phys_pars.getFreeNuisanceParametersForProbe(p1))
            self.free_nuisance_parameters.update(phys_pars.getFreeNuisanceParametersForProbe(p2))

        self.free_physical_parameters = {}
        self.free_physical_parameters.update(self.free_cosmological_parameters)
        self.free_physical_parameters.update(self.free_nuisance_parameters)

    def selectClDerivatives(self, l_dict: "Dict[str, np.ndarray]",
                            cl_dict: "Dict[str, np.ndarray]") -> "Tuple[SmartKeyDict, ClSmartKeyDict]":
        src_l_dict = SmartKeyDict(l_dict)
        src_cl_dict = ClSmartKeyDict(cl_dict)

        sel_l_dict = SmartKeyDict()
        sel_cl_dict = ClSmartKeyDict()
        for cl_key in self.data_vector:
            sel_l_dict[cl_key] = src_l_dict[cl_key]
            sel_cl_dict[cl_key] = src_cl_dict[cl_key]

        return sel_l_dict, sel_cl_dict

    def buildClDerivativeDataVector(self, dcoll: "ClDerivativeCollector") -> "Tuple[np.ndarray, np.ndarray]":
        sel_l_dict, sel_dcl_dict = self.selectClDerivatives(l_dict=dcoll.ell_dict, cl_dict=dcoll.dc_lij_array_dict)
        ells, sel_dcl_dict = mask_cl_blocks_to_common_ells(l_dict=sel_l_dict, cl_dict=sel_dcl_dict)

        vec_dcls = []
        for entry in self.data_vector:
            cl_block = sel_dcl_dict[entry]
            if entry.is_auto_correlation:
                vectorized_block = fu.vecpClArray(cl_block)
            else:
                vectorized_block = fu.vecClArray(cl_block)

            vec_dcls.append(vectorized_block)

        vec_dcls = np.concatenate(vec_dcls, axis=1)

        return ells, vec_dcls

    def buildVectorizedClDerivatives(self, dcoll_dvar_dict: "Dict[str, ClDerivativeCollector]") -> "Tuple[np.ndarray, Dict[str, np.ndarray]]":
        ells_list = []
        vec_dcl_dict = {}
        for dvar in self.free_physical_parameters:
            ells, vec_dcls_dvar = self.buildClDerivativeDataVector(dcoll_dvar_dict[dvar])
            vec_dcl_dict[dvar] = vec_dcls_dvar
            ells_list.append(ells)

        ells_list = np.stack(ells_list)
        unique_ells = np.unique(ells_list, axis=0).squeeze()

        return unique_ells, vec_dcl_dict

    def buildAndInvertCovarianceMatrix(self, delta_cls: "DeltaClCollection"):
        self.covariance = ClCovarianceMatrix(self.data_vector)
        self.covariance.evaluateAllBlocks(delta_cls)
        self.covariance.maskCovBlocksToCommonElls()
        self.covariance.buildMatrix()
        self.covariance.evaluateInverse()

    def checkClDerivativeMultipolesConsistency(self, vec_dcl_ells: "np.ndarray"):
        if not len(vec_dcl_ells.shape) == 1:
            raise MultipoleError(f"derivatives of c_ells have not a unique multipole domain: {vec_dcl_ells}")
        if not np.all(vec_dcl_ells == self.covariance.common_ells):
            raise MultipoleError(f"multipoles of cl derivatives and of covariance are not consistent."
                                 f"Cl derivatives have {vec_dcl_ells}\n "
                                 f"Covariance has {self.covariance.common_ells}")

    def computeFisherOfMultipole(self, vec_dcls_dict: "Dict[str, np.ndarray]") -> "pd.DataFrame":
        fisher_of_ell_elements = {}
        n_ells = self.ells.size
        for a, b in itertools.combinations_with_replacement(self.free_physical_parameters, 2):
            dCl_da = vec_dcls_dict[a]
            dCl_db = vec_dcls_dict[b]
            f_ell = np.array([
                dCl_da[l_idx] @ self.covariance.inverse[l_idx] @ dCl_db[l_idx]
                for l_idx in range(n_ells)
            ])

            fisher_of_ell_elements[(a, b)] = f_ell
            fisher_of_ell_elements[(b, a)] = f_ell

        fisher_of_ell_df = pd.DataFrame(fisher_of_ell_elements)
        fisher_of_ell_df.set_index(self.ells, inplace=True)
        fisher_of_ell_df.index.names = ['ell']

        return fisher_of_ell_df

    @staticmethod
    def sumFisherOverMultipoles(f_ell_df: "pd.DataFrame", ells: "np.ndarray") -> "pd.DataFrame":
        f_ell_sel = f_ell_df.loc[f_ell_df.index.isin(ells)]
        f_mat = f_ell_sel.sum().unstack()

        return f_mat

    def addInvolvedProbesContribution(self, f_mat_df: "pd.DataFrame", auto_f_ells: "Dict[str, pd.DataFrame]") -> "pd.DataFrame":
        logger.info("Adding single probes exclusive contributions")
        involved_probes = self.data_vector.getInvolvedAutoCorrelationProbes()
        for p in involved_probes:
            auto_f_df = auto_f_ells[p]

            auto_ells = auto_f_df.index
            exclusive_ells = auto_f_df.index[~auto_ells.isin(self.ells)]
            if exclusive_ells.size > 0:
                logger.info(f"Adding {p} exclusive contribution from {exclusive_ells[0]} to {exclusive_ells[-1]}")
                auto_f_contrib = self.sumFisherOverMultipoles(auto_f_df, exclusive_ells)
                f_mat_df = f_mat_df.add(auto_f_contrib, fill_value=0)

        return f_mat_df


def compute_fisher_matrix(data_vector, *args, **kwargs):
    f_ev = FisherEvaluator(data_vector)

    return f_ev.computeFisherMatrix(*args, **kwargs)
