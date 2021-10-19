from abc import ABC, abstractmethod
import numpy as np
from scipy import integrate
from typing import List
from pathlib import Path
from seyfert.cosmology import c_ells
from seyfert.cosmology.parameter import PhysicalParameter, PhysicalParametersCollection
from seyfert.utils.workspace import WorkSpace
import logging

logger = logging.getLogger(__name__)


class DifferentiationError(Exception):
    pass


class ClDifferentiator(ABC):
    param: "PhysicalParameter"
    workspace: "WorkSpace"
    fiducial_cl: "c_ells.AngularCoefficient"

    def __init__(self, probe1: "str" = None, probe2: "str" = None, param: "PhysicalParameter" = None,
                 workspace: "WorkSpace" = None):
        self.probe1 = probe1
        self.probe2 = probe2
        self.param = param
        self.workspace = workspace
        self.fiducial_cl = None

    def computeDerivative(self) -> "np.ndarray":
        if self.param.is_nuisance and self.param.probe not in {self.fiducial_cl.probe1, self.fiducial_cl.probe2}:
            logger.info(f"Cl {self.fiducial_cl.obs_key} does not depend on {self.param.name}, "
                        f"setting derivative to 0")
            dc_lij = np.zeros(self.fiducial_cl.c_lij.shape)
        else:
            dc_lij = self.doComputeDerivative()

        return dc_lij

    @abstractmethod
    def loadClData(self) -> "None":
        fid_cl_file = self.workspace.getClFile(dvar='central', step=0)
        self.fiducial_cl = c_ells.AngularCoefficient.fromHDF5(fid_cl_file, self.probe1, self.probe2, root='cls')

    @abstractmethod
    def doComputeDerivative(self) -> "np.ndarray":
        pass


class SteMClDifferentiator(ClDifferentiator):
    c_dvar_lij: "np.ndarray"

    def __init__(self, vectorized: "bool" = False, phys_pars: "PhysicalParametersCollection" = None, **kwargs):
        super(SteMClDifferentiator, self).__init__(**kwargs)
        self.c_dvar_lij = None
        self.vectorized = vectorized
        self.phys_pars = phys_pars

    def loadClData(self) -> "None":
        super(SteMClDifferentiator, self).loadClData()
        dvar_cl_files = self.collectClVariationsFiles()
        c_dvar_lij = []
        for f in dvar_cl_files:
            cl = c_ells.AngularCoefficient.fromHDF5(f, self.probe1, self.probe2, root='cls')
            c_dvar_lij.append(cl.c_lij)

        self.c_dvar_lij = np.stack(c_dvar_lij)

    def doComputeDerivative(self) -> "np.ndarray":
        dc_lij = np.zeros(self.c_dvar_lij.shape[1:])
        if self.phys_pars is None:
            self.phys_pars = self.workspace.getParamsCollection()
        dvar_stem_values = self.phys_pars.computePhysParSTEMValues(dvar=self.param.name)
        logger.info(f"computing derivative of cl {self.probe1}_{self.probe2}")
        if self.vectorized:
            dc_lij = self.vectorizedSteM(dvar_stem_values, self.c_dvar_lij)
        else:
            l_number, i_number, j_number = dc_lij.shape
            for lidx in range(l_number):
                for myi in range(i_number):
                    for myj in range(j_number):
                        dc_lij[lidx, myi, myj] = self.STEM_derivative(x=dvar_stem_values,
                                                                      y=self.c_dvar_lij[:, lidx, myi, myj])

        return dc_lij

    @staticmethod
    def STEM_derivative(x: "np.ndarray", y: "np.ndarray") -> "float":
        if np.min(y) == np.max(y):
            slope = 0
        else:
            linear = False
            slope = None
            while not linear:
                fit_pars = np.polyfit(x, y, 1)
                slope = fit_pars[0]
                intercept = fit_pars[1]
                y_fit = intercept + slope * x
                percentile_difference = np.abs((y - y_fit) / y)
                if all(percentile_difference < 0.01) or len(y) <= 3:
                    linear = True
                else:
                    x = x[1:-1]
                    y = y[1:-1]

        return slope

    @staticmethod
    def vectorizedSteM(dvar_vals: "np.ndarray", c_dvar_lij: "np.ndarray") -> "np.ndarray":
        x = dvar_vals
        n_dvar, n_ells, ni, nj = c_dvar_lij.shape
        y = c_dvar_lij.reshape((n_dvar, n_ells * ni * nj))
        slopes = np.zeros(y.shape[1])
        intercepts = np.zeros(y.shape[1])

        assert x.shape[0] == y.shape[0]

        if np.min(y) == np.max(y):
            pass
        else:
            not_linear_mask = np.ones(y.shape[1]).astype(bool)
            stop = False
            while not stop:
                fit_results = np.polyfit(x, y, 1)
                slopes[not_linear_mask] = fit_results[0, not_linear_mask]
                intercepts[not_linear_mask] = fit_results[1, not_linear_mask]

                yfit = np.expand_dims(x, 1) * slopes + np.expand_dims(intercepts, 0)
                y_abs_diff = np.abs(yfit - y)
                err = np.zeros(y_abs_diff.shape)
                mask = y != 0
                err[mask] = y_abs_diff[mask] / y[mask]
                not_linear_mask = np.any(err > 0.01, axis=0)

                x = x[1:-1]
                y = y[1:-1]

                less_than_three_points = y.shape[0] <= 3
                stop = np.count_nonzero(not_linear_mask) == 0 or less_than_three_points

        dc_lij = slopes.reshape((n_ells, ni, nj))

        return dc_lij

    def collectClVariationsFiles(self) -> "List[Path]":
        par_name = self.param.name
        dvar_cl_files = list(self.workspace.cl_dir.glob(f"dvar_{par_name}_*/cl*.h5"))
        dvar_cl_files.append(self.workspace.getClFile(dvar='central', step=0))

        def _sorting_func(file: "Path"):
            dvar, step = self.workspace.getDvarStepFromCosmoDirName(file.parent.name)
            return step

        dvar_cl_files.sort(key=_sorting_func)

        if len(dvar_cl_files) <= 1:
            raise Exception(f"Invalid number of cl variations: {len(dvar_cl_files)}")

        return dvar_cl_files


class AnalyticClDifferentiator(ClDifferentiator):
    def __init__(self, **kwargs):
        super(AnalyticClDifferentiator, self).__init__(**kwargs)

    def loadClData(self) -> "None":
        super(AnalyticClDifferentiator, self).loadClData()
        pmm_file = self.workspace.getPowerSpectrumFile(dvar='central', step=0)
        self.fiducial_cl.loadCosmology(pmm_file, load_power_spectrum=True)
        self.fiducial_cl.cosmology.evaluateOverRedshiftGrid()

    def doComputeDerivative(self) -> "np.ndarray":
        if not self.param.is_galaxy_bias_parameter:
            raise NotImplementedError(f"Analytic fiducial_cl derivative is not implemented "
                                      f"for parameter {self.param.name}")
        else:
            p1, p2 = self.fiducial_cl.probe1, self.fiducial_cl.probe2
            probe = self.param.probe
            if probe != p1 and probe != p2:
                raise DifferentiationError(f"What are you doing bro? fiducial_cl "
                                           f"{self.fiducial_cl.obs_key} cannot depend on bias {probe}")
            dc_lij = np.zeros(self.fiducial_cl.c_lij.shape)
            bias_model_name = self.fiducial_cl.getWeightFunction(probe).bias.model_name
            if bias_model_name == "constant" or bias_model_name == "piecewise":
                k = int(self.param.name[-1])
                bk = self.param.fiducial
                if bias_model_name == "constant":
                    _, i, j = np.indices(self.fiducial_cl.c_lij.shape)
                    if probe == p1:
                        delta_ik = (i == k).astype(np.int64)
                        dc_lij += (delta_ik / bk) * self.fiducial_cl.c_lij
                    if probe == p2:
                        delta_jk = (j == k).astype(np.int64)
                        dc_lij += (delta_jk / bk) * self.fiducial_cl.c_lij
                elif bias_model_name == "piecewise":
                    self.fiducial_cl.evaluateLimberApproximatedPowerSpectrum()
                    z_b_edg = self.fiducial_cl.getWeightFunction(probe).z_bin_edges
                    z_grid = self.fiducial_cl.z_array
                    bin_mask = (z_grid >= z_b_edg[k]) & (z_grid <= z_b_edg[k+1])

                    for myi in range(dc_lij.shape[1]):
                        for myj in range(dc_lij.shape[2]):
                            cl_integrand = self.fiducial_cl.getLimberRedshiftIntegrand(myi, myj)
                            if probe == p1:
                                dc_lij[:, myi, myj] += (1 / bk) * integrate.simps(y=cl_integrand[:,  bin_mask],
                                                                                  x=z_grid[bin_mask], axis=-1)
                            if probe == p2:
                                dc_lij[:, myi, myj] += (1 / bk) * integrate.simps(y=cl_integrand[:, bin_mask],
                                                                                  x=z_grid[bin_mask], axis=-1)
            else:
                raise NotImplementedError(f"analytic derivative is not implemented for bias model {bias_model_name}")

        return dc_lij
