import numpy as np
import pandas as pd
from scipy import special
from typing import Callable, Union, Any


def smoothstep(x: "Union[float, np.ndarray]",
               x_min: "float" = 0, x_max: "float" = 1, N: "int" = 3) -> "Union[float, np.ndarray]":
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    result = 0
    for n in range(N + 1):
        result += special.comb(N + n, n) * special.comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    return result


def top_hat_filter(x: "Union[float, np.ndarray]", R: "float") -> "Union[float, np.ndarray]":
    return 1 - smoothstep(x, x_min=0.9*R, x_max=1.1*R, N=3)


def callable_stencil_derivative(f: "Callable", x: "Union[float, np.ndarray]",
                                step: "float", **kwargs) -> "Union[float, np.ndarray]":
    f_m2 = f(x - 2 * step, **kwargs)
    f_m1 = f(x - 1 * step, **kwargs)
    f_p1 = f(x + 1 * step, **kwargs)
    f_p2 = f(x + 2 * step, **kwargs)
    return (-f_p2 + 8*f_p1 - 8*f_m1 + f_m2)/(12*step)


def pad(x: "Any", y: "Any") -> "Any":
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        res = np.zeros(x.shape)
        mask = (x + y) != 0
        diff = np.abs(x - y)
        den = np.abs(x + y)
        res[mask] = 200 * diff[mask] / den[mask]
    elif isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        if set(x.index) != set(y.index):
            raise Exception(f"x and y have different index. "
                            f"index of x missing from y {set(x.index) - set(y.index)} \n"
                            f"index of y missing from x {set(y.index) - set(x.index)} \n")
        if set(x.columns) != set(y.columns):
            raise Exception(f"x and y have different columns. "
                            f"columns of x missing from y {set(x.columns) - set(y.columns)} \n"
                            f"columns of y missing from x {set(y.columns) - set(x.columns)} \n")

        den = np.where(x + y != 0, (x + y)/2, 1)
        num = np.abs(x - y)
        res = pd.DataFrame(100*(num / den), index=x.index, columns=x.columns)
    else:
        res = 200*np.abs((x-y)/(x+y))
    return res


def odm_weighted_pad(x: "Any", y: "Any", ref: "str" = "mean") -> "Any":
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        func_refs = {
            "mean": np.mean,
            "max": np.max
        }

        normalization = np.abs(x+y)/2
        odm = normalization / func_refs[ref](normalization)
        res = pad(x, y) * odm
    else:
        raise NotImplemented(f"function not implemented for {type(x)} yet")
    return res


def percentage_difference(minuend: "Any", ref: "Any"):
    return 100 * (minuend - ref) / ref
