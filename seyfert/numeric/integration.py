import numpy as np
from scipy import integrate
import logging
from typing import Union

logger = logging.getLogger(__name__)


class IntegrationError(Exception):
    pass


class CompositeNewtonCotesIntegrator:
    x: "np.ndarray"
    y: "np.ndarray"
    order: "int"

    def __init__(self, x: "np.ndarray", y: "np.ndarray", order: "int", axis: "int" = -1):
        self.x = x
        self.y = y
        self.order = order
        self.y_integ_axis = axis
        self.n_tot_points = None
        self.n_bins = None
        self.n_excess_points = None
        self._max_order_allowed = 6

        self.checkValid()

        self.n_tot_points = self.x.shape[0]
        self.n_bins = (self.n_tot_points - 1) // self.order
        self.n_excess_points = (self.n_tot_points - 1) % self.order

    def checkValid(self) -> "None":
        if not isinstance(self.x, np.ndarray):
            raise TypeError("x must be np.ndarray")
        if not isinstance(self.y, np.ndarray):
            raise TypeError("y must be np.ndarray")
        xshape = self.x.shape
        n_x_dims = len(xshape)
        if n_x_dims != 1:
            raise Exception(f"x array must be 1-D, not {n_x_dims}-D")
        if not isinstance(self.order, (int, np.int16, np.int32, np.int64)):
            logger.warning(f"{self.order} is not integer-like, try to cast as integer")
            try:
                self.order = int(self.order)
            except TypeError:
                raise TypeError(f"cannot cast order as integer, invalid type {type(self.order)}")
            except ValueError:
                raise ValueError(f"cannot cast order as integer, invalid value {self.order}")

        if self.order <= 0:
            raise ValueError(f"Invalid Newton Cotes order {self.order}, must be integer > 0")

        if self.order > self._max_order_allowed:
            raise IntegrationError(f"Newton Cotes order {self.order} too high, "
                                   f"maximum allowed: {self._max_order_allowed}")

        n_x_points = xshape[0]
        yshape = self.y.shape
        n_y_points = yshape[self.y_integ_axis]
        if n_y_points != n_x_points:
            raise Exception(f"invalid y axis dimension {n_y_points} for x dimension {n_x_points}")

        self.checkAlmostEqualSpacing(atol=1e-15, rtol=1e-15)

    def checkAlmostEqualSpacing(self, atol: "float", rtol: "float"):
        x_deltas = np.diff(self.x)
        x_delta0 = x_deltas[0]
        tolerance = atol + rtol * np.abs(x_delta0)
        err_diff_max = np.max(np.abs(x_deltas - x_delta0))
        if not err_diff_max <= tolerance:
            logger.warning(f"maximum deviation {err_diff_max} from constant spacing "
                           f"is bigger than tolerance {tolerance}")

    def computeIntegral(self) -> "Union[float, np.ndarray]":
        self.checkValid()
        result = self.doIntegrate()
        return result

    def doIntegrate(self) -> "Union[float, np.ndarray]":
        an, B = integrate.newton_cotes(self.order, 1)
        dx = self.x[1] - self.x[0]

        axis = self.y_integ_axis

        k = np.arange(self.n_bins)
        result = 0
        for delta in range(self.order + 1):
            idxs = self.order * k + delta
            y_slice = np.take(self.y, idxs, axis=self.y_integ_axis)
            p_sum = dx * np.sum(an[delta] * y_slice, axis=self.y_integ_axis)
            result += p_sum

        if self.n_excess_points != 0:
            aend, Bend = integrate.newton_cotes(self.n_excess_points, 1)
            slc = [slice(None)] * len(self.y.shape)
            slc[axis] = slice(-self.n_excess_points - 1, self.y.shape[axis])
            yend = self.y[tuple(slc)]
            last_bin_contrib = dx * np.sum(aend * yend, axis=axis)

            result += last_bin_contrib

        return result
