#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Reference values for selection and classification functions."""

import math

import pytest

from prommis.comminution.functions.selection import (
    austin_selection,
    selection_value,
    vogel_peukert_selection,
    whiten_selection,
)


@pytest.mark.unit
def test_whiten_anchor():
    k1, k2, k3 = 0.200, 0.250, 2.3
    assert whiten_selection(0.200, k1, k2, k3) == 0.0
    assert whiten_selection(0.250, k1, k2, k3) == 1.0
    # C(0.225) = 1 - 0.5^2.3
    assert whiten_selection(0.225, k1, k2, k3) == pytest.approx(
        1.0 - 0.5**2.3, rel=1e-12
    )


@pytest.mark.unit
def test_king_anchor():
    # d_min = 12.5 mm, d_max = 30 mm
    king = {"CSS": 0.025, "alpha1": 0.5, "alpha2": 1.2, "n": 1.0}
    assert selection_value("king", 0.0125, king) == 0.0
    assert selection_value("king", 0.030, king) == 1.0
    # linear ramp (n=1): S(21.25 mm) = 0.5
    assert selection_value("king", 0.02125, king) == pytest.approx(0.5, rel=1e-12)
    # quadratic ramp (n=2): S(21.25 mm) = 1 - 0.5**2
    assert selection_value("king", 0.02125, dict(king, n=2.0)) == pytest.approx(
        0.75, rel=1e-12
    )


@pytest.mark.unit
def test_austin_anchor():
    s1, d1, a = 0.5, 10e-3, 0.5
    assert austin_selection(2.5e-3, s1, d1, a) == pytest.approx(0.25, rel=1e-12)
    assert austin_selection(10e-3, s1, d1, a) == pytest.approx(0.5, rel=1e-12)
    assert austin_selection(1000e-3, 0.9, d1, a) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.unit
def test_vogel_peukert_kernel_anchor():
    f_mat, xw_min, v = 0.5, 10.0, 10.0  # W_kin = 1/2 v^2 = 50 J/kg
    # threshold d = xw_min / W_kin = 0.2 m
    assert vogel_peukert_selection(0.30, f_mat, xw_min, v) == pytest.approx(
        1.0 - math.exp(-(0.30 * 50 - 10) * f_mat), rel=1e-12
    )
    # Check values below, at, and just above the selection threshold.
    assert vogel_peukert_selection(0.10, f_mat, xw_min, v) == 0.0
    assert vogel_peukert_selection(0.20, f_mat, xw_min, v) == 0.0
    just_above = vogel_peukert_selection(0.21, f_mat, xw_min, v)
    assert 0.0 < just_above < 0.5
