import numpy as np
from typing import Tuple, Dict, Callable


def get_enumeration_map(shape: "Tuple[int, int]", symm: "bool") -> "Dict[Tuple[int, int], int]":
    nA, nB = shape
    if symm:
        if nA != nB:
            raise Exception("Cannot perform symmetric enumeration for not square shape!")
        i, j = triu_idxs_by_col(nA)
    else:
        i, j = np.indices(shape)
        i = i.ravel(order='F')
        j = j.ravel(order='F')

    enum_map = {(myi, myj): idx for idx, (myi, myj) in enumerate(zip(i, j))}

    return enum_map


def _get_idxs_by(func: "Callable", n: "int", order='F') -> "Tuple[np.ndarray, np.ndarray]":
    i, j = np.indices((n, n))
    i = i.ravel(order=order)
    j = j.ravel(order=order)

    mask = func(i, j)

    return i[mask], j[mask]


def triu_idxs_by_col(n: "int", order='F') -> "Tuple[np.ndarray, np.ndarray]":
    return _get_idxs_by(lambda x, y: x <= y, n, order=order)


def tril_idxs_by_col(n: "int", order='F') -> "Tuple[np.ndarray, np.ndarray]":
    return _get_idxs_by(lambda x, y: x >= y, n, order=order)


def diag_idxs(n: "int", order='F') -> "Tuple[np.ndarray, np.ndarray]":
    return _get_idxs_by(lambda x, y: x == y, n, order=order)


def vecp(A: "np.ndarray") -> "np.ndarray":
    nrows, ncols = A.shape

    if nrows != ncols:
        raise Exception("Cannot vecp not square matrix")

    i, j = triu_idxs_by_col(A.shape[0])

    return A[i, j]


def unvecp(v: "np.ndarray") -> "np.ndarray":
    p = int(0.5 * (np.sqrt(1 + 8*len(v)) - 1))

    result = np.zeros((p, p))
    i, j = triu_idxs_by_col(p)
    result[i, j] = v

    result = result + result.T
    result[diag_idxs(p)] /= 2

    return result


def vec(A: "np.ndarray") -> "np.ndarray":
    return A.ravel(order='F')


def unvec(v: "np.ndarray") -> "np.ndarray":
    n = int(np.sqrt(len(v)))

    return v.reshape((n, n), order='F')


def duplication_matrix(n: "int") -> "np.ndarray":
    iden = np.eye(n * (n + 1) // 2)

    return np.array([unvecp(x).ravel() for x in iden]).T


def elimination_matrix(n: "int") -> "np.ndarray":
    vecp_indices = vec(np.triu(np.ones((n, n))))

    return np.eye(n * n)[vecp_indices != 0]


def transition_matrix(n: "int") -> "np.ndarray":
    return np.linalg.pinv(duplication_matrix(n).T)


def commutation_matrix(p: "int", q: "int") -> "np.ndarray":
    K = np.eye(p * q)
    indices = np.arange(p * q).reshape((p, q), order='F')

    return K.take(indices.ravel(), axis=0)
