from typing import Dict, List, Union
from pathlib import Path
import copy
import re
import logging

from seyfert.utils.workspace import WorkSpace
from seyfert.fisher import fisher_utils as fu
from seyfert.fisher.fisher_analysis import FisherAnalysis
from seyfert.fisher.fisher_matrix import FisherMatrix

logger = logging.getLogger(__name__)

LCDM_params = {"Omm", "Omb", "h", "ns", "sigma8"}
w0_wa_CDM_params = LCDM_params | {"w0", "wa"}
gamma_w0_wa_CDM_params = w0_wa_CDM_params | {"gamma"}


def get_free_cosmo_pars_for_cosmology(cosmology: "str"):
    return {
        "LCDM": LCDM_params,
        "w0_wa_CDM": w0_wa_CDM_params,
        "gamma_w0_wa_CDM": gamma_w0_wa_CDM_params,
    }[cosmology]


def create_final_results(rundir: "Union[str, Path]", outdir_name: "str", overwrite=False):
    ws = WorkSpace(rundir)
    params = ws.getParamsCollection()
    metadata = ws.getRunMetadata()
    fisher_matrices = {}
    for file in ws.base_fishers_dir.glob("fisher*.hdf5"):
        F = FisherMatrix.fromHDF5(file)
        fisher_matrices[F.name] = F

    main_outdir = ws.run_dir / outdir_name
    main_outdir.mkdir(exist_ok=True, parents=True)
    post_fishers_names = fu.load_a_posteriori_fisher_comb_names()
    logger.info(f"Evaluating the following a posteriori combinations:\n" + "\n".join(post_fishers_names))

    for cosmology in ["w0_wa_CDM", "LCDM"]:
        # marginalize before combining Fishers
        logger.info(f"{'#'*20} COSMOLOGY: {cosmology} {'#'*20}")
        logger.info("Creating results with nuisance parameters marginalized BEFORE doing a posteriori fisher sums")
        create_results_for_cosmology(fisher_matrices, cosmology, params, post_fishers_names, main_outdir,
                                     marg_nuis_before=True, overwrite=overwrite, metadata=metadata)
        # marginalize after combining Fishers
        logger.info("Creating results with nuisance parameters marginalized AFTER doing a posteriori fisher sums")
        create_results_for_cosmology(fisher_matrices, cosmology, params, post_fishers_names, main_outdir,
                                     marg_nuis_before=False, overwrite=overwrite, metadata=metadata)
        # fix nuisance parameters
        logger.info("Creating results with nuisance parameters FIXED before doing a posteriori fisher sums")
        create_results_for_cosmology(fisher_matrices, cosmology, params, post_fishers_names, main_outdir,
                                     fix_nuisance=True, overwrite=overwrite, metadata=metadata)
        logger.info("#"*60)


def create_results_for_cosmology(fisher_matrices, cosmology, params, post_fishers_names, main_outdir,
                                 marg_nuis_before=True, fix_nuisance=False, overwrite=False, metadata=None) -> "FisherAnalysis":

    # DeepCopy fishers in order to avoid overwriting the original ones when fixing or marginalizing parameters
    Fs = copy.deepcopy(fisher_matrices)
    free_params_names = get_free_cosmo_pars_for_cosmology(cosmology)
    free_cosmo_pars = {
        name: fiducial for name, fiducial in params.free_cosmo_pars_fiducials.items()
        if name in free_params_names
    }
    fixed_cosmo_pars = {
        name: fiducial for name, fiducial in params.free_cosmo_pars_fiducials.items()
        if name not in free_params_names
    }

    for fisher_name in Fs:
        Fs[fisher_name].fixParameters(set(fixed_cosmo_pars))

    if fix_nuisance:
        for fisher_name in Fs:
            Fs[fisher_name].fixParameters(Fs[fisher_name].nuisance_parameters)
    else:
        if marg_nuis_before:
            for fisher_name in Fs:
                Fs[fisher_name].marginalizeNuisance(ret_copy=False)

    evaluate_a_posteriori_fisher_combinations(Fs, post_fishers_names=post_fishers_names)

    an = FisherAnalysis(cosmo_pars_fiducial=free_cosmo_pars, fisher_matrices=Fs)
    an.evaluateMarginalizedErrors()
    an.evaluateRelativeMarginalizedErrors()
    if metadata is not None:
        an.metadata = metadata

    if fix_nuisance:
        dirname = "nuis_fixed"
    else:
        dirname = "marg_before" if marg_nuis_before else "marg_after"

    outdir = main_outdir / cosmology / dirname
    outdir.mkdir(parents=True, exist_ok=True)

    an.saveFishersToDisk(outdir, overwrite=overwrite)
    an.writeAbsoluteAndRelativeMarginalizedErrors(outdir=outdir, overwrite=overwrite)

    return an


def evaluate_a_posteriori_fisher_combinations(Fs: "Dict[str, FisherMatrix]",
                                              post_fishers_names: "List[str]" = None):
    if post_fishers_names is None:
        post_fishers_names = fu.load_a_posteriori_fisher_comb_names()
    for post_fish_name in post_fishers_names:
        result_fisher = _compute_a_posteriori_fisher_comb_from_name(post_fish_name, Fs)
        if result_fisher is not None:
            Fs[post_fish_name] = result_fisher
        else:
            pass


def _compute_a_posteriori_fisher_comb_from_name(comb_name: "str", Fs: "Dict[str, FisherMatrix]") -> "FisherMatrix":
    op_list = re.split(r" (\+|-) ", comb_name)
    if not op_list:
        raise Exception(f"Invalid name {comb_name}")
    operators_idxs = [idx for idx, el in enumerate(op_list) if el in {"+", "-"}]

    # Check that all fishers in op_list are in Fs, if not print a warning and skip
    for operand in op_list:
        if operand not in {"+", "-"} and operand not in Fs:
            logger.warning(f"Operand {operand} not present in available fisher matrices, skipping "
                           f"combination {comb_name}")
            return None

    result = Fs[op_list[0]]
    for idx in operators_idxs:
        operator = {"+": lambda x, y: x + y, "-": lambda x, y: x - y}[op_list[idx]]
        first_operand = result
        second_operand = Fs[op_list[idx + 1]]
        result = operator(first_operand, second_operand)

    return result
