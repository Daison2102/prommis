#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Tests for crusher distribution sizes, power, and invalid specifications."""

import math
from unittest.mock import patch

import pytest

from pyomo.environ import Block, Constraint, Var, units, value

from idaes.core.initialization import InitializationStatus
from idaes.core.util.exceptions import ConfigurationError

from prommis.comminution.functions import distributions
from prommis.comminution.functions.power_laws import numeric_specific_energy
from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models._crusher import initializer as crusher_initializer
from prommis.comminution.unit_models.tests.crusher_test_support import (
    DENSITY_TWO,
    EDGES_REF,
    F80_LIVE_REF,
    FEED_REF,
    GGS,
    RR,
    crusher,
    fix_feed,
    flowsheet,
    one_mineral_flowsheet,
    two_time_flowsheet,
)

# This mesh has a sub-micron lower edge and isolates the Rosin-Rammler
# size-bound check at exponent 0.0675.
EDGES_SHAPE_SIZE_BOUND = (1e-7, 1.225e-6, 1e-4, 1e-3, 16e-3)

# Duty for FEED_REF (10 kg/s), Wi = 14 kWh/t, and product P80 = 8 mm;
# this P80 is valid for both distribution shapes.
DATUM_POWER_REF = (
    3.6 * 14.0 * numeric_specific_energy("bond", 1.0, 8.0e-3, F80_LIVE_REF) * 10.0
)


@pytest.mark.unit
def test_ggs_shape_size_mapping():
    m = flowsheet()
    cr = crusher(
        m,
        psd_method="distribution_function",
        distribution_shape=GGS,
        shape_exponent=1.5,
        target_product_p80=5e-3,
    )
    expected = 5e-3 / (0.8 ** (1.0 / 1.5))
    cr.d_max[0].set_value(expected)
    assert math.isclose(value(cr.product_p80[0]), 5e-3, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.component
@pytest.mark.solver
def test_distribution_function_zero_power_preserves_p80_not_full_psd():
    m = one_mineral_flowsheet()
    cr = crusher(
        m,
        psd_method="distribution_function",
        distribution_shape=RR,
        shape_exponent=1.5,
        bond_work_index=12.0,
    )
    feed = [1.0, 1.0, 7.0, 1.0]  # F80 about 7.4 mm; the RR shape is mesh-valid
    fix_feed(cr.properties_in[0], {"OreA": feed})
    cr.power[0].fix(0.0)
    native.CrusherSolidPSDInitializer().initialize(cr)
    assert value(cr.product_p80[0]) == pytest.approx(
        value(cr.properties_in[0].percentile_size[0.8]), rel=1e-3
    )
    total = value(cr.properties_out[0].flow_mass_sized)
    prod = [
        value(cr.properties_out[0].flow_mass_sized_comp_size["OreA", k]) / total
        for k in range(4)
    ]
    feed_frac = [v / sum(feed) for v in feed]
    assert any(abs(p - f) > 0.02 for p, f in zip(prod, feed_frac))


@pytest.mark.component
@pytest.mark.solver
def test_distribution_function_rr_size_spec_bins_match_analytic():
    n, d80 = 1.5, 5e-3
    m = one_mineral_flowsheet()
    cr = crusher(
        m,
        psd_method="distribution_function",
        distribution_shape=RR,
        shape_exponent=n,
        target_product_p80=d80,
        bond_work_index=12.0,
    )
    fix_feed(cr.properties_in[0], {"OreA": [0.0, 0.0, 0.0, 10.0]})
    native.CrusherSolidPSDInitializer().initialize(cr)
    d63 = d80 / ((-math.log(0.2)) ** (1.0 / n))
    edges = list(EDGES_REF)
    cdf = [1.0 - math.exp(-((e / d63) ** n)) for e in edges]
    q_top = cdf[-1]
    expected = [cdf[1] / q_top] + [(cdf[k + 1] - cdf[k]) / q_top for k in range(1, 4)]
    total = value(cr.properties_out[0].flow_mass_sized)
    got = [
        value(cr.properties_out[0].flow_mass_sized_comp_size["OreA", k]) / total
        for k in range(4)
    ]
    for g, exp in zip(got, expected):
        assert g == pytest.approx(exp, abs=1e-6)


@pytest.mark.component
@pytest.mark.solver
def test_convention_split_distribution_size_spec():
    # Near 95% passing at the top edge, d_63 is about 7.699 mm.
    # Analytic and outlet P80 are about 10.574 and 10.878 mm.
    m = flowsheet()
    cr = crusher(
        m,
        psd_method="distribution_function",
        distribution_shape=RR,
        shape_exponent=1.5,
        target_product_p80=10.57387e-3,
        bond_work_index=12.0,
    )
    fix_feed(cr.properties_in[0], FEED_REF)  # m_sized = 10 kg/s, all in the top bin
    p80_power = value(units.convert(cr.target_product_p80, to_units=units.m))
    f80_live = value(cr.properties_in[0].percentile_size[0.8])
    assert math.isclose(f80_live, 0.01440, rel_tol=1e-6, abs_tol=0.0)
    native.CrusherSolidPSDInitializer().initialize(cr)
    p80_live_product = value(cr.properties_out[0].percentile_size[0.8])
    assert p80_live_product == pytest.approx(10.87783e-3, rel=1e-7)
    # Power uses the analytic P80, not the outlet state's P80.
    assert value(cr.power[0]) == pytest.approx(6.01135, rel=1e-6)
    c_power = numeric_specific_energy("bond", 1.0, p80_power, f80_live)
    c_live = numeric_specific_energy("bond", 1.0, p80_live_product, f80_live)
    assert value(cr.power[0]) != pytest.approx(3.6 * 12.0 * c_live * 10.0, rel=1e-3)
    # Compare the specific-energy difference in kWh/t, independent of feed rate.
    offset = 12.0 * (c_live - c_power)
    assert offset == pytest.approx(-0.01642, abs=1e-7)


def _entry_state(block):
    """Return Var values and fixed flags, plus Constraint and Block active flags."""
    return (
        {v.name: (v.fixed, v.value) for v in block.component_data_objects(Var)},
        {
            c.name: c.active
            for c in block.component_data_objects(Constraint, active=None)
        },
        {b.name: b.active for b in block.component_data_objects(Block, active=None)},
    )


def _free_shape_size_unit(
    power=None,
    edges=None,
    feed=(0.0, 0.0, 0.0, 10.0),
    shape=RR,
    exponent=1.5,
    target=None,
):
    """Build a one-mineral distribution crusher with Wi = 14 kWh/t.

    ``target`` selects a size specification; otherwise ``power`` can fix
    the duty. The characteristic size remains free. Return model and unit.
    """
    m = one_mineral_flowsheet(edges=edges)
    config = dict(
        psd_method="distribution_function",
        distribution_shape=shape,
        shape_exponent=exponent,
        bond_work_index=14.0,
    )
    if target is not None:
        config["target_product_p80"] = target
    cr = crusher(m, **config)
    fix_feed(cr.properties_in[0], {"OreA": list(feed)})
    if power is not None:
        cr.power[0].fix(power)
    return m, cr


def _fixed_shape_size_unit(shape):
    """Build a two-time case with a measured size fixed at t = 0.

    Both times use FEED_REF and DATUM_POWER_REF, with a free work index.
    The RR size exceeds its bound; the GGS size implies P80 below the mesh.
    """
    m = two_time_flowsheet(species=["OreA", "OreB"], density=DENSITY_TWO)
    cr = crusher(
        m,
        psd_method="distribution_function",
        distribution_shape=shape,
        shape_exponent=1.5,
        bond_work_index=14.0,
    )
    for t in (0.0, 1.0):
        fix_feed(cr.properties_in[t], FEED_REF)
        cr.power[t].fix(DATUM_POWER_REF)
    cr.bond_work_index.unfix()
    if shape == RR:
        cr.d_63[0.0].fix(1.01 * cr.d_63[0.0].ub)
    else:
        floor = EDGES_REF[0] + 0.8 * (EDGES_REF[1] - EDGES_REF[0])
        cr.d_max[0.0].fix(distributions.shape_size_from_p80(GGS, 0.99 * floor, 1.5))
    return m, cr


def _overflow_shape_size_unit():
    """Build a Rosin-Rammler case whose fixed size overflows shape validation.

    The exponent is 1000, d_63 is fixed at 3 mm, the duty is 50 kW, and
    the work index is free.
    """
    m = flowsheet()
    cr = crusher(
        m,
        psd_method="distribution_function",
        distribution_shape=RR,
        shape_exponent=1000.0,
    )
    fix_feed(cr.properties_in[0], FEED_REF)
    cr.power[0].fix(50.0)
    cr.d_63[0].fix(3e-3)
    cr.bond_work_index.unfix()
    return m, cr


BUILD_REJECTIONS = [
    (
        "rr_exponent_0.001",
        {"shape_exponent": 0.001},
        "rosin_rammler.*exponent.*characteristic size",
    ),
    (
        "rr_exponent_0.01",
        {"shape_exponent": 0.01},
        "rosin_rammler.*exponent.*characteristic size",
    ),
    (
        "rr_exponent_1000_with_target",
        {"shape_exponent": 1000.0, "target_product_p80": 0.003},
        "rosin_rammler.*exponent.*characteristic size",
    ),
]

# Each case gives a name, helper options, and an expected precheck message.
FREE_SHAPE_SIZE_REJECTIONS = [
    ("negative_power", {"power": -1.0}, "is negative"),
    ("implied_p80_below_mesh_floor", {"power": 1.0e4}, "attainable mesh band"),
    ("rr_shape_unattainable", {"power": 1.0e-6}, "distribution is invalid"),
    (
        "shape_size_below_lower_bound",
        {
            "power": 4771.37546,
            "exponent": 0.0675,
            "edges": EDGES_SHAPE_SIZE_BOUND,
        },
        "must lie within",
    ),
    (
        "ggs_shape_unattainable",
        {"power": 1.0, "shape": GGS, "exponent": 1e-4},
        "distribution is invalid",
    ),
    (
        "size_spec_coarsens_the_feed",
        {"target": 8.0e-3, "feed": (10.0, 0.0, 0.0, 0.0)},
        "no-coarsening",
    ),
]

# Fixed characteristic sizes are measurements checked during precheck;
# each case gives a name, helper options, and expected message.
FIXED_SHAPE_SIZE_REJECTIONS = [
    ("rr_shape_size_above_upper_bound", {"shape": RR}, "distribution is invalid"),
    ("ggs_p80_below_mesh_floor", {"shape": GGS}, "attainable mesh band"),
]


def _assert_precheck_rejects(m, cr, match):
    """Check the precheck error, unchanged state, and absence of a solver call."""
    # Give the outlet a distinct value so a premature write changes the snapshot.
    cr.properties_out[0.0].flow_mass_sized_comp_size["OreA", 1].set_value(0.75)
    entry = _entry_state(cr)
    init = native.CrusherSolidPSDInitializer()
    with patch.object(
        crusher_initializer,
        "get_solver",
        side_effect=AssertionError("unexpected solver call"),
    ) as solver_spy:
        with pytest.raises(ConfigurationError, match=match) as exc:
            init.initialize(cr)
    solver_spy.assert_not_called()
    assert f"t = {m.fs.time.first()}" in str(exc.value)
    assert init.summary[cr]["status"] == InitializationStatus.PrecheckFailed
    assert _entry_state(cr) == entry


@pytest.mark.component
def test_distribution_specification_rejections():
    failures = []
    for case_id, kwargs, match in BUILD_REJECTIONS:
        try:
            with pytest.raises(ConfigurationError, match=match):
                crusher(flowsheet(), psd_method="distribution_function", **kwargs)
        except (Exception, pytest.fail.Exception) as exc:
            failures.append(f"{case_id}: {exc}")
    for case_id, kwargs, match in FREE_SHAPE_SIZE_REJECTIONS:
        try:
            _assert_precheck_rejects(*_free_shape_size_unit(**kwargs), match)
        except (Exception, pytest.fail.Exception) as exc:
            failures.append(f"{case_id}: {exc}")
    for case_id, kwargs, match in FIXED_SHAPE_SIZE_REJECTIONS:
        try:
            _assert_precheck_rejects(*_fixed_shape_size_unit(**kwargs), match)
        except (Exception, pytest.fail.Exception) as exc:
            failures.append(f"{case_id}: {exc}")
    # Shape-check overflow is reported as an invalid distribution.
    try:
        _assert_precheck_rejects(
            *_overflow_shape_size_unit(), "distribution is invalid"
        )
    except (Exception, pytest.fail.Exception) as exc:
        failures.append(f"rr_exponent_1000_fixed_shape_size: {exc}")
    assert not failures, "\n".join(failures)
