import numpy as np
from typing import Union


def InchesToPts(l: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return l * 72


def PtsToInches(l: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return l / 72


def InchesToCm(l: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return l * 2.54


def CmToInches(l: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return l / 2.54


def SteradToDeg2(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x * (180 / np.pi)**2


def SteradToArcmin2(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x * (180 * 60 / np.pi) ** 2


def Deg2ToSterad(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x / ((180 / np.pi) ** 2)


def Arcmin2ToSterad(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x / ((180 * 60 / np.pi) ** 2)


def invDeg2ToInvSterad(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x * (180 / np.pi)**2


def invSteradToinvDeg2(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x / ((180 / np.pi)**2)


def invArcmin2ToInvSterad(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x * (180 * 60 / np.pi)**2


def invSteradToinvArcmin2(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x / ((180 * 60 / np.pi)**2)

