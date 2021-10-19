from typing import Dict
from sympy import symbols, Matrix
from functools import partial
import re
import json
import numpy as np
from pathlib import Path
import copy
import pandas as pd
from seyfert.file_io import ext_input
from seyfert.utils.tex_utils import TeXTranslator

translator = TeXTranslator()


def read_ISTGCphFishers(direc, flat, scenario):
    if not isinstance(direc, (str, Path)):
        raise TypeError("direc argument must be str or Path!")
    if not isinstance(flat, bool):
        raise TypeError("flat argument must be bool!")
    if scenario not in {"optimistic", "pessimistic"}:
        raise ValueError(f"invalid scenario {scenario}")
    
    direc = Path(direc)
    curv_dirname = "Flat" if flat else "NonFlat"
    subdir = Path(direc) / curv_dirname / scenario

    if not subdir.exists():
        raise FileNotFoundError(str(subdir))
    
    with open(direc / "legend.json", mode="r") as jsf:
        legend = json.load(jsf)
    
    regex = re.compile(r"^fisher_matrices_results_baseline_GR_(?P<code_key>[A-Z]{2}).*\.txt$")
    files = {}
    for f in subdir.glob("*.txt"):
        match = regex.match(f.name)
        if not match:
            raise Exception(f"Unexpected file name {f.name} does not match {regex.pattern}")
        code_key = match.group("code_key")
        try:
            code_name = legend[code_key]
            files[code_name] = f
        except KeyError:
            raise KeyError(f"Invalid code key {code_key}. \n Viable options are: {', '.join(legend.keys())}")
    
    reader = ext_input.ISTInputReader()
    fishers = {
        code: reader.readISTFisherMatrixFile(file)[1]
        for code, file in files.items()
    }

    return fishers


def compute_errs(fishers: "Dict[str, pd.DataFrame]"):
    funcs = {
        "marg": lambda x: np.sqrt(np.diag(np.linalg.inv(x))),
        "unmarg": lambda x: 1/np.sqrt(np.diag(x))
    }
    
    errs = {}
    for kind in funcs:
        errs[kind] = pd.DataFrame({key: pd.Series(funcs[kind](fisher), index=fisher.index)
                                   for key, fisher in fishers.items()}).T
        errs[kind].index.names = ["code"]
        errs[kind].columns.names = ["cosmo_par"]
    
    return pd.concat(errs, names=["kind"])


# noinspection PyTypeChecker
def compute_deltas_errs(errs: "pd.DataFrame"):
    deltas = {}
    ref_funcs = {
        "mean": partial(np.mean, axis=0),
        "median": partial(np.median, axis=0)
    }
    for ref_name, ref_func in ref_funcs.items():
        deltas[ref_name] = {}
        for err_kind in ["marg", "unmarg"]:
            err_kind_data = errs.loc[err_kind]
            ref = pd.Series(ref_func(err_kind_data), index=err_kind_data.columns)
            deltas[ref_name][err_kind] = 100 * (err_kind_data - ref) / ref
    
    return pd.concat({
        ref: pd.concat(deltas[ref], names=["kind"]) for ref in deltas
    }, names=["ref"])


def get_params_bases(f, fiducials):
    theta = dict(zip(f.index, symbols([translator.PhysParNameToTeX(name) for name in f.index])))
    fid_theta = {theta[name]: fiducials[name] for name in theta}

    thetap = {}
    for name, sym in theta.items():
        if name.startswith("bg"):
            thetap[f"{name}_sigma8"] = sym*theta["sigma8"]
        else:
            thetap[name] = sym

    thetapp = copy.deepcopy(thetap)
    del thetapp["sigma8"]
    
    thetaf = {name: sym for name, sym in theta.items() if not name.startswith("bg")}
    
    return theta, thetap, thetapp, thetaf, fid_theta


def T1(f, theta, thetap, fid_theta):  
    J1 = Matrix([[tp.diff(t) for t in theta.values()] for tp in thetap.values()]).inv()
    numJ1 = np.array(J1.subs(fid_theta)).astype(np.float64)
    F1 = pd.DataFrame(numJ1.T @ f.to_numpy() @ numJ1, index=thetap.keys(), columns=thetap.keys())
    
    return F1


def T2(f, thetapp, thetaf, fid_theta):    
    J2 = Matrix([[tp.diff(tf) for tf in thetaf.values()] for tp in thetapp.values()])
    numJ2 = np.array(J2.subs(fid_theta)).astype(np.float64)
    F2 = pd.DataFrame(numJ2.T @ f.to_numpy() @ numJ2, index=thetaf.keys(), columns=thetaf.keys())
    
    return F2


def apply_T1_remove_sigma8_then_T2(f, fiducials):
    theta, thetap, thetapp, thetaf, fid_theta = get_params_bases(f, fiducials)
    
    F1 = T1(f, theta, thetap, fid_theta)    
    F1 = F1.drop("sigma8", axis=0).drop("sigma8", axis=1)
    F2 = T2(F1, thetapp, thetaf, fid_theta)
    
    return F2


def apply_T1_remove_sigma8(f, fiducials):
    theta, thetap, _, _, fid_theta = get_params_bases(f, fiducials)

    F1 = T1(f, theta, thetap, fid_theta)
    F1 = F1.drop("sigma8", axis=0).drop("sigma8", axis=1)

    return F1
