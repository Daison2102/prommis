#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Tests for the assumed-shape distribution forms and their discretization."""

import math

import pytest
from pyomo.environ import ConcreteModel, Var, value

from prommis.comminution.core.psd_math import cumulative_passing, size_at_passing
from prommis.comminution.core.size_mesh import geometric_series
from prommis.comminution.functions import distributions
from prommis.comminution.functions.distributions import (
    assert_shape_valid,
    distribution_fractions,
    shape_size_upper_bound,
)

HANDCALC_EDGES = [1e-3, 2e-3, 4e-3, 8e-3, 16e-3]


@pytest.mark.unit
def test_rr_happy_path_sum_and_p80():
    edges = list(geometric_series(top=40e-3, bottom=0.1e-3, ratio=400 ** (1.0 / 60)))
    d63, n = 10e-3, 1.5
    fracs = distribution_fractions("rosin_rammler", edges, d63, n)
    assert sum(fracs) == pytest.approx(1.0, rel=1e-12)
    cum = cumulative_passing(fracs)
    p80 = size_at_passing(edges, cum, 0.8)
    analytic = d63 * (-math.log(0.2)) ** (1.0 / n)
    assert p80 == pytest.approx(analytic, rel=2e-3)  # 60-bin interpolation
    # A Pyomo characteristic size must remain symbolic when its value changes.
    m = ConcreteModel()
    m.d63 = Var(initialize=d63)
    live = distribution_fractions("rosin_rammler", edges, m.d63, n)
    assert [value(f) for f in live] == pytest.approx(fracs, rel=1e-12)
    m.d63.set_value(12e-3)
    assert [value(f) for f in live] == pytest.approx(
        distribution_fractions("rosin_rammler", edges, 12e-3, n), rel=1e-12
    )
    with pytest.raises(ValueError, match="shape exponent n"):
        distribution_fractions("rosin_rammler", edges, m.d63, 0.0)


@pytest.mark.unit
def test_ggs_happy_path_sum_and_p80():
    edges = list(geometric_series(top=40e-3, bottom=0.1e-3, ratio=400 ** (1.0 / 60)))
    d_max, n = 20e-3, 2.0
    fracs = distribution_fractions("gates_gaudin_schuhmann", edges, d_max, n)
    assert sum(fracs) == pytest.approx(1.0, rel=1e-6)
    cum = cumulative_passing(fracs)
    p80 = size_at_passing(edges, cum, 0.8)
    analytic = d_max * 0.8 ** (1.0 / n)
    assert p80 == pytest.approx(analytic, rel=3e-3)


@pytest.mark.unit
def test_assert_shape_valid(monkeypatch):
    # Rosin-Rammler requires at least 0.95 passing at the top mesh edge.
    with pytest.raises(ValueError):
        assert_shape_valid("rosin_rammler", HANDCALC_EDGES, 10e-3, 1.5)
    assert_shape_valid("rosin_rammler", [0.1e-3, 1e-3, 5e-3, 40e-3], 5e-3, 1.5)
    # GGS requires d_max at or below the top mesh edge.
    with pytest.raises(ValueError):
        assert_shape_valid("gates_gaudin_schuhmann", HANDCALC_EDGES, 20e-3, 2.0)
    assert_shape_valid("gates_gaudin_schuhmann", HANDCALC_EDGES, 8e-3, 2.0)
    msg = r"top mesh edge \(d_max = 0\.02 > x_N = 0\.016\)\.$"
    with pytest.raises(ValueError, match=msg):
        distribution_fractions("gates_gaudin_schuhmann", HANDCALC_EDGES, 20e-3, 2.0)
    # Rounding makes the initial size limit fail 0.95 passing; one step lower passes.
    x_top, n = 0.01388, 2.85310
    ub = shape_size_upper_bound("rosin_rammler", x_top, n)
    assert_shape_valid("rosin_rammler", [x_top / 2, x_top], ub, n)
    with pytest.raises(ValueError):
        assert_shape_valid(
            "rosin_rammler", [x_top / 2, x_top], math.nextafter(ub, math.inf), n
        )
    # A CDF that never reaches the threshold exhausts the retry limit.
    monkeypatch.setattr(distributions, "rosin_rammler_cdf", lambda *args: 0.0)
    with pytest.raises(ValueError, match="rounding steps"):
        shape_size_upper_bound("rosin_rammler", x_top, n)
