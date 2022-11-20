from typing import TYPE_CHECKING, Dict
import numpy as np
from sympy import comp

from seyfert.main.cl_core import compute_cls_variations
from seyfert.derivatives import cl_derivative, differentiator
from seyfert.utils import general_utils as gu

if TYPE_CHECKING:
    from seyfert.cosmology.cosmology import Cosmology
    from seyfert.cosmology.parameter import PhysicalParametersCollection
    from seyfert.cosmology.c_ells import AngularCoefficientsCollector
    from seyfert.utils.workspace import WorkSpace
    from seyfert.cosmology.redshift_density import RedshiftDensity
    from seyfert.config.forecast_config import ForecastConfig
    from seyfert.config.main_config import AngularConfig, PowerSpectrumConfig


def compute_cls_derivatives_wrt(dvar: "str", fid_cls: "AngularCoefficientsCollector", ws: "WorkSpace",
                                phys_pars: "PhysicalParametersCollection", densities: "Dict[str, RedshiftDensity]",
                                forecast_config: "ForecastConfig", 
                                angular_config: "AngularConfig",
                                pmm_cfg: "PowerSpectrumConfig" = None, 
                                compute_pmm=False,
                                fiducial_cosmology: "Cosmology" = None) -> "cl_derivative.ClDerivativeCollector":
    cls_dvar = compute_cls_variations(dvar, fid_cls, ws, phys_pars, densities, forecast_config, angular_config,
                                      fiducial_cosmology=fiducial_cosmology, 
                                      pmm_cfg=pmm_cfg, compute_pmm=compute_pmm)
    deriv_coll = cl_derivative.ClDerivativeCollector(dvar=dvar)
    deriv_coll.dcl_dict = {}

    for probe_key in cls_dvar[0].cl_dict:
        p1, p2 = gu.get_probes_from_comb_key(probe_key)

        diff = differentiator.SteMClDifferentiator(probe1=p1, probe2=p2, param=phys_pars[dvar], phys_pars=phys_pars,
                                                   vectorized=True)
        diff.c_dvar_lij = np.stack([cls_dvar[step][probe_key].c_lij for step in sorted(cls_dvar)])

        dcl = cl_derivative.ClDerivative(p1=p1, p2=p2)
        dcl.l_bin_centers = cls_dvar[0][probe_key].l_bin_centers
        dcl.dc_lij = diff.doComputeDerivative()

        deriv_coll.dcl_dict[probe_key] = dcl

    return deriv_coll
