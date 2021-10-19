import logging
from pathlib import Path
import pandas as pd
from typing import Dict, List

from seyfert.fisher.fisher_evaluator import FisherEvaluator
from seyfert.fisher.datavector import ClDataVector
from seyfert.fisher.fisher_matrix import FisherMatrix
from seyfert.utils.workspace import WorkSpace
from seyfert.cosmology.parameter import PhysicalParametersCollection
from seyfert.fisher.delta_cl import DeltaClCollection
from seyfert.derivatives.cl_derivative import ClDerivativeCollector

logger = logging.getLogger(__name__)


def prepare_inputs(ws: "WorkSpace", shot_noise_file=None):
    forecast_config = ws.getForecastConfiguration()
    if shot_noise_file is None:
        logger.info("No shot noise file passed, defaulting to forecast_config.shot_noise_file")
        shot_noise_file = forecast_config.shot_noise_file

    dcoll_dvar_dict = {
        dvar: ClDerivativeCollector.fromHDF5(ws.getClDerivativeFile(dvar)) for dvar in ws.collectDerivativeParams()
    }
    delta_cls = DeltaClCollection.fromHDF5(ws.delta_cls_file)
    if shot_noise_file is not None:
        logger.info(f"re-computing delta cls with shot noise loaded from file {shot_noise_file}")
        delta_cls.loadShotNoiseFromFile(shot_noise_file)
        delta_cls.evaluateSingleBlocks(f_sky=forecast_config.survey_f_sky)
    else:
        logger.info("No shot noise found, defaulting to already computed delta cls")

    return delta_cls, dcoll_dvar_dict


def compute_fisher_matrix(data_vector_like: "ClDataVector", ws: "WorkSpace",
                          phys_pars: "PhysicalParametersCollection", delta_cls: "DeltaClCollection",
                          dcoll_dvar_dict: "Dict[str, ClDerivativeCollector]") -> "FisherMatrix":
    fish_evaluator = FisherEvaluator(data_vector_like)
    data_vector = fish_evaluator.data_vector
    fish_name = data_vector.name

    if data_vector.is_single_autocorrelation:
        f_mat_df, f_ell_df = fish_evaluator.computeFisherMatrix(phys_pars=phys_pars, delta_cls=delta_cls,
                                                                dcoll_dvar_dict=dcoll_dvar_dict, return_f_ell=True)
        probe = list(data_vector.getInvolvedAutoCorrelationProbes())[0]
        with pd.HDFStore(ws.auto_f_ells_file, mode="a") as hdf:
            f_ell_df.to_hdf(hdf, key=probe)
    else:
        auto_f_ells = {}
        with pd.HDFStore(ws.auto_f_ells_file, mode="r") as hdf:
            for key in hdf:
                auto_f_ell_df: "pd.DataFrame" = hdf.get(key)
                auto_f_ells[Path(key).name] = auto_f_ell_df

        if len(auto_f_ells) < 3:
            raise Exception(f"Cannot compute fisher {fish_name} with only the following auto-correlation "
                            f"fisher matrices: {list(auto_f_ells)}")

        f_mat_df = fish_evaluator.computeFisherMatrix(phys_pars=phys_pars, delta_cls=delta_cls,
                                                      dcoll_dvar_dict=dcoll_dvar_dict, auto_f_ells=auto_f_ells)

    free_cosmo_pars = fish_evaluator.free_cosmological_parameters
    free_nuis_pars = fish_evaluator.free_nuisance_parameters

    fisher_matrix = FisherMatrix.from_dataframe(df=f_mat_df, name=fish_name,
                                                cosmological_parameters=free_cosmo_pars,
                                                nuisance_parameters=free_nuis_pars)
    return fisher_matrix


def compute_and_save_fishers(brief_str_data_vectors, outdir, ws: "WorkSpace",
                             phys_pars: "PhysicalParametersCollection", delta_cls: "DeltaClCollection",
                             dcoll_dvar_dict: "Dict[str, ClDerivativeCollector]") -> "List[FisherMatrix]":
    fishers = []
    n_fishers = len(brief_str_data_vectors)
    for idx_fisher, brief_data_vector in enumerate(brief_str_data_vectors):
        logger.info(f"{'#'*60}")
        logger.info("#")
        logger.info(f"# Computing fisher for {brief_data_vector}, {idx_fisher+1}/{n_fishers}")
        data_vector = ClDataVector.fromBriefString(brief_data_vector)

        output_file = outdir / f"fisher_{brief_data_vector}.hdf5"

        if output_file.exists():
            logger.warning(f"fisher {brief_data_vector} already exists saved at {output_file}, skipping")
        else:
            fisher_matrix = compute_fisher_matrix(data_vector, ws, phys_pars, delta_cls, dcoll_dvar_dict)
            fishers.append(fisher_matrix)
            logger.info(f"Writing output file {output_file.name}")
            fisher_matrix.writeToFile(outfile=output_file)

        logger.info("# Done")
        logger.info(f"{'#' * 60}")
        logger.info("")

    return fishers
