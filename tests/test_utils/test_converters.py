import pytest
import numpy as np
from seyfert.utils import converters as cnv


def test_inches_pts_convs():
    assert cnv.InchesToPts(1) == 72
    assert cnv.PtsToInches(1) == 1 / 72


def test_inches_cms_convs():
    assert cnv.InchesToCm(1) == 2.54
    assert cnv.CmToInches(1) == 1 / 2.54


def test_square_angle_convs():
    sterad_to_deg2 = (180 / np.pi)**2
    sterad_to_arcmin2 = (180 * 60 / np.pi) ** 2

    assert cnv.SteradToDeg2(1) == 1 * sterad_to_deg2
    assert cnv.SteradToArcmin2(1) == 1 * sterad_to_arcmin2
    assert cnv.invSteradToinvDeg2(1) == 1 / sterad_to_deg2
    assert cnv.invSteradToinvArcmin2(1) == 1 / sterad_to_arcmin2
    assert cnv.invDeg2ToInvSterad(1) == 1 * sterad_to_deg2
    assert cnv.invArcmin2ToInvSterad(1) == 1 * sterad_to_arcmin2


