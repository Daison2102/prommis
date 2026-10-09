#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Tests for crusher equipment presets, stage selection, and diagnostics."""

import math
import re

import pytest

from idaes.core.initialization import InitializationStatus
from idaes.core.util.exceptions import ConfigurationError, InitializationError

from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models.tests.crusher_test_support import (
    EDGES_X,
    FEED_REF,
    FEED_X,
    SINGLE_STRESS_REF,
    crusher,
    fix_feed,
    flowsheet,
    one_mineral_flowsheet,
    state_snapshot,
    two_time_flowsheet,
    x_common,
)


@pytest.mark.unit
def test_equipment_presets_and_stage_resolution(caplog):
    caplog.set_level("WARNING", logger=native._log.name)
    whiten = {"K1": 2e-3, "K2": 8e-3, "K3": 2.3}
    vogel = {"q": 0.05, "dprime": 0.005}
    cr = crusher(
        one_mineral_flowsheet(),
        crusher_equipment="jaw",
        recycle="single_pass",
        selection_params=whiten,
        breakage_params=vogel,
    )
    assert cr.config.selection_function == "whiten"
    assert cr.config.breakage_function == "vogel"
    assert cr.config.crusher_stage == "primary"
    # a matching explicit stage is accepted (default classification_recycle)
    cr = crusher(
        flowsheet(),
        crusher_equipment="jaw",
        crusher_stage="primary",
        selection_params={"K1": 1.0e-3, "K2": 16.0e-3, "K3": 2.3},
        breakage_params=vogel,
    )
    assert cr.config.crusher_stage == "primary"
    assert cr.config.breakage_function == "vogel"
    # an explicit function wins over the equipment preset
    cr = crusher(
        one_mineral_flowsheet(),
        crusher_equipment="jaw",
        recycle="single_pass",
        selection_function="user",
        selection_params=[0.0, 0.0, 0.5, 1.0],
        breakage_function="vogel",
        breakage_params=vogel,
    )
    assert cr.config.selection_function == "user"
    with pytest.raises(ConfigurationError, match="inconsistent"):
        crusher(
            one_mineral_flowsheet(),
            crusher_equipment="jaw",
            crusher_stage="tertiary",
            recycle="single_pass",
            selection_params=whiten,
            breakage_params=vogel,
        )
    # The cone preset selects King; missing King parameters must fail.
    with pytest.raises(ConfigurationError, match="king"):
        crusher(
            one_mineral_flowsheet(),
            crusher_equipment="cone",
            recycle="single_pass",
            breakage_params=vogel,
        )
    # The cone's King preset conflicts with the default Whiten recycle path.
    with pytest.raises(ConfigurationError, match="Whiten"):
        crusher(
            one_mineral_flowsheet(),
            crusher_equipment="cone",
            selection_params={"CSS": 0.025, "alpha1": 0.5, "alpha2": 1.2, "n": 1.0},
            breakage_params=vogel,
        )
    # Gyratory has no function preset, so the default Whiten selection still
    # needs K1 and K2.
    caplog.clear()
    with pytest.raises(ConfigurationError, match="whiten.*K1"):
        crusher(
            one_mineral_flowsheet(), crusher_equipment="gyratory", recycle="single_pass"
        )
    assert any("no equipment-specific" in r.message for r in caplog.records)
    cr = crusher(
        one_mineral_flowsheet(),
        crusher_equipment="gyratory",
        recycle="single_pass",
        selection_function="whiten",
        selection_params=whiten,
        breakage_function="vogel",
        breakage_params=vogel,
    )
    assert cr.config.crusher_stage == "primary"
    # For tabular and distribution paths, equipment sets only the stage.
    for equipment, method_kwargs in (
        ("jaw", {"psd_method": "tabular", "tabular_psd": [0.4, 0.3, 0.2, 0.1]}),
        (
            "gyratory",
            {"psd_method": "distribution_function", "shape_exponent": 1.5},
        ),
    ):
        caplog.clear()
        cr = crusher(
            one_mineral_flowsheet(), crusher_equipment=equipment, **method_kwargs
        )
        assert cr.config.crusher_stage == "primary"
        baseline = crusher(one_mineral_flowsheet(), **method_kwargs)
        assert cr.config.selection_function == baseline.config.selection_function
        assert cr.config.breakage_function == baseline.config.breakage_function
        assert not any("no equipment-specific" in r.message for r in caplog.records)


@pytest.mark.unit
def test_warn_only_helpers_never_raise_on_unavailable_values(caplog):
    m = flowsheet(edges=EDGES_X)
    cr = crusher(m, crusher_stage="primary", **x_common("single_stress"))
    fix_feed(cr.properties_in[0], FEED_X)
    cr.properties_in[0].flow_mass_sized_comp_size["OreA", 0].set_value(None)
    msg = cr.check_sized_feed_flow()
    assert isinstance(msg, str) and "has no value" in msg
    cr.properties_in[0].flow_mass_sized_comp_size["OreA", 0].set_value(0.0)
    cr.properties_out[0].flow_mass_sized_comp_size["OreA", 0].set_value(None)
    caplog.set_level("WARNING", logger=native._log.name)
    caplog.clear()
    before = state_snapshot(m)
    quiet = cr.check_applicability(emit_warning=False)
    assert isinstance(quiet, list) and len(quiet) == 1
    assert "has no value" in quiet[0]
    assert not [r for r in caplog.records if r.name == native._log.name]
    caplog.clear()
    loud = cr.check_applicability(emit_warning=True)
    assert loud == quiet
    records = [r for r in caplog.records if r.name == native._log.name]
    assert len(records) == 1
    assert quiet[0] in records[0].message
    assert "CrusherSolidPSD" in records[0].message
    assert records[0].levelname == "WARNING"
    assert state_snapshot(m) == before
    assert cr.check_solved_flows().count("has no value") == 1

    # A NaN in OreA does not suppress OreB's mass-balance warning.
    m = flowsheet()
    cr = crusher(m, **dict(SINGLE_STRESS_REF, n_stress_events=3))
    for state in (cr.properties_in[0], cr.properties_out[0]):
        for s in ("OreA", "OreB"):
            for k in range(4):
                state.flow_mass_sized_comp_size[s, k].set_value(0.0)
    for obj in cr.psd_stage.values():
        obj.set_value(0.0)
    cr.properties_out[0].flow_mass_sized_comp_size["OreB", 0].set_value(1.0)
    cr.properties_out[0].flow_mass_sized_comp_size["OreA", 0].set_value(
        math.nan, skip_validation=True
    )
    before = state_snapshot(m)
    with pytest.raises(InitializationError, match="not finite"):
        cr.validate_solved_flows()
    caplog.clear()
    message = cr.check_solved_flows()
    assert "not finite" in message and "OreA" in message
    assert (
        "mass balance failed at t = 0.0, component 'OreB': product flow 1.0 kg/s, "
        "feed flow 0.0 kg/s, residual 1.000e+00 kg/s, tolerance 1.000e-09 kg/s."
    ) in message
    assert len(caplog.records) == 1 and caplog.records[0].message == message
    assert state_snapshot(m) == before

    # A nonfinite inlet takes precedence over the minimum-flow warning.
    m = flowsheet()
    cr = crusher(m, **SINGLE_STRESS_REF)
    fix_feed(cr.properties_in[0], FEED_REF)
    cr.properties_in[0].flow_mass_sized_comp_size["OreA", 0].set_value(
        math.nan, skip_validation=True
    )
    before = state_snapshot(m)
    with pytest.raises(ConfigurationError, match="not finite"):
        cr.validate_sized_feed_flow()
    message = cr.check_sized_feed_flow()
    assert "not finite" in message and "at or below" not in message
    assert state_snapshot(m) == before

    # This 1e-7 relative overshoot is within both the 1e-5 relative and
    # 1e-8 m absolute allowances.
    m = one_mineral_flowsheet()
    dist = crusher(
        m,
        psd_method="distribution_function",
        distribution_shape="rosin_rammler",
        shape_exponent=1.5,
        target_product_p80=5.0e-3,
    )
    ub = dist.d_63[0].ub
    for shape_size, warns in (
        # d_63 implied by the target P80
        (5e-3 / ((-math.log(0.2)) ** (1.0 / 1.5)), False),
        (ub, False),  # largest valid d_63
        (ub * (1 + 1e-7), False),
        (ub * 1.001, True),
    ):
        dist.d_63[0].set_value(shape_size, skip_validation=True)
        before = state_snapshot(m)
        assert ("shape" in dist.check_distribution_shape()) is warns, shape_size
        assert state_snapshot(m) == before
    dist.d_63[0].set_value(None)
    msg = dist.check_distribution_shape()
    assert isinstance(msg, str) and "has no value" in msg
    # On this micron mesh, the 0.1% overshoot is only 4.8e-10 m.
    # It fails because the relative allowance is tighter.
    m = one_mineral_flowsheet(edges=[0.25e-6, 0.5e-6, 1.0e-6])
    micron = crusher(m, psd_method="distribution_function", shape_exponent=1.5)
    ub = micron.d_63[0].ub
    micron.d_63[0].set_value(ub * (1 + 1e-7), skip_validation=True)
    assert micron.check_distribution_shape() == ""
    micron.d_63[0].set_value(ub * 1.001, skip_validation=True)
    assert "shape validity" in micron.check_distribution_shape()

    m = flowsheet()
    cr = crusher(m, **SINGLE_STRESS_REF)
    assert cr.check_distribution_shape() == ""
    assert cr.config.crusher_stage is None
    assert cr.check_applicability() == []

    # The feed report collects unavailable values at every time point in one
    # message and one log record.
    m = two_time_flowsheet()
    cr = crusher(m, **SINGLE_STRESS_REF)
    for t in (0, 1):
        fix_feed(cr.properties_in[t], {"OreA": [0.0] * 4})
    cr.properties_in[0].flow_mass_sized_comp_size["OreA", 0].set_value(None)
    cr.properties_in[1].flow_mass_sized_comp_size["OreA", 0].set_value(
        math.nan, skip_validation=True
    )
    before = state_snapshot(m)
    caplog.clear()
    message = cr.check_sized_feed_flow()
    assert "t = 0.0 has no value" in message
    assert "t = 1.0 is not finite" in message
    assert len(caplog.records) == 1 and caplog.records[0].message == message
    assert state_snapshot(m) == before


def _tabular_with_stage(stage, q):
    edges = [0.005, 0.02, 0.05, 0.1, 0.15, 0.3]  # m, 0.5 to 30 cm
    m = one_mineral_flowsheet(edges=edges)
    cr = crusher(
        m,
        psd_method="tabular",
        crusher_stage=stage,
        tabular_psd=q,
        bond_work_index=12.0,
    )
    fix_feed(cr.properties_in[0], {"OreA": [0.0, 0.0, 0.0, 0.0, 10.0]})
    native.CrusherSolidPSDInitializer().initialize(cr)
    # Return the parent model alongside the unit to keep it alive.
    return m, cr


@pytest.mark.component
@pytest.mark.solver
def test_check_applicability_in_range_silent_and_out_warns():
    m_in, in_range = _tabular_with_stage("secondary", [0.0, 1.0, 0.0, 0.0, 0.0])
    assert in_range.check_applicability(emit_warning=False) == []
    m_out, out = _tabular_with_stage("secondary", [0.0, 0.0, 0.0, 0.0, 1.0])
    assert len(out.check_applicability(emit_warning=False)) == 1


@pytest.mark.component
@pytest.mark.solver
def test_power_law_soft_range_warning(caplog):
    # At 10 mm, product P80 exceeds Rittinger's range; initialization warns.
    m = one_mineral_flowsheet()
    cr = crusher(
        m,
        psd_method="distribution_function",
        shape_exponent=1.5,
        target_product_p80=10e-3,
        power_law="rittinger",
        bond_work_index=12.0,
    )
    fix_feed(cr.properties_in[0], {"OreA": [0.0, 0.0, 0.0, 10.0]})
    native.CrusherSolidPSDInitializer().initialize(cr)
    assert any("applicable range" in r.message for r in caplog.records)
    # The same 10 mm product is within Bond's applicable range.
    caplog.clear()
    m_bond = one_mineral_flowsheet()
    bond = crusher(
        m_bond,
        psd_method="distribution_function",
        shape_exponent=1.5,
        target_product_p80=10e-3,
        power_law="bond",
        bond_work_index=12.0,
    )
    fix_feed(bond.properties_in[0], {"OreA": [0.0, 0.0, 0.0, 10.0]})
    native.CrusherSolidPSDInitializer().initialize(bond)
    assert not any("applicable range" in r.message for r in caplog.records)


def _closed_repeated_stress_unit():
    """Build a three-event crusher with feed, stage, and outlet rows set to FEED_REF."""
    m = flowsheet()
    cr = crusher(m, **dict(SINGLE_STRESS_REF, n_stress_events=3))
    fix_feed(cr.properties_in[0], FEED_REF)
    for s, row in FEED_REF.items():
        for k, flow in enumerate(row):
            cr.properties_out[0].flow_mass_sized_comp_size[s, k].set_value(flow)
            for r in (1, 2):
                cr.psd_stage[0, r, s, k].set_value(flow)
    return m, cr


# Case ID, inserted values, and expected diagnostic after the unit name.
_UNAVAILABLE_SOLVED_FLOW_CASES = [
    # A nonfinite stage value takes precedence over an earlier product
    # mass-balance error.
    (
        "nan_stage",
        {("outlet", "OreA", 1): 1.0e-3, ("stage", 1, "OreB", 2): math.nan},
        "psd_stage[0.0, 1, 'OreB', 2] is not finite (nan).",
    ),
    # Report an infinite outlet value at its index before checking mass balance.
    (
        "inf_outlet",
        {("outlet", "OreA", 1): math.inf},
        "outlet[0.0]['OreA', 1] is not finite (inf).",
    ),
    (
        "unavailable_stage",
        {("stage", 2, "OreA", 0): None},
        "psd_stage[0.0, 2, 'OreA', 0] has no value.",
    ),
]


def _plant(cr, planted):
    """Write ``{location: value}`` into the outlet or stage rows of ``cr``."""
    for location, flow in planted.items():
        if location[0] == "outlet":
            target = cr.properties_out[0].flow_mass_sized_comp_size[location[1:]]
        else:
            target = cr.psd_stage[(0,) + location[1:]]
        target.set_value(flow, skip_validation=True)


@pytest.mark.unit
def test_solved_flow_backstops_report_sign_and_closure_defects():
    m, cr = _closed_repeated_stress_unit()
    assert cr.check_solved_flows() == ""
    assert cr.validate_solved_flows() is None
    out = cr.properties_out[0].flow_mass_sized_comp_size
    stage = cr.psd_stage

    # The row still sums to the feed. Only the negative entry beyond the
    # inlet-based tolerance is reported.
    for k, flow in enumerate([-1e-4, -1e-7, 0.0, 6.0 + 1e-4 + 1e-7]):
        out["OreA", k].set_value(flow, skip_validation=True)
    before = state_snapshot(m)
    message = cr.check_solved_flows()
    assert (
        "product flow at t = 0.0, component 'OreA', interval 0, is -0.0001 kg/s, "
        "below the permitted minimum -6.001e-06 kg/s"
    ) in message
    assert "interval 1" not in message
    assert "mass balance failed" not in message
    with pytest.raises(
        InitializationError,
        match="product flow at t = 0.0, component 'OreA', interval 0",
    ):
        cr.validate_solved_flows()
    assert state_snapshot(m) == before
    for k, flow in enumerate(FEED_REF["OreA"]):
        out["OreA", k].set_value(flow)

    # Stage signs use the inlet-flow tolerance even when stage totals are much larger.
    for k in range(4):
        stage[0, 1, "OreA", k].set_value(1e8 if k == 3 else 0.0)
        stage[0, 2, "OreA", k].set_value(
            1e8 if k == 3 else -1e-4 if k == 0 else 0.0, skip_validation=True
        )
    before = state_snapshot(m)
    message = cr.check_solved_flows()
    assert (
        "stage flow at t = 0.0, stage 2, component 'OreA', interval 0, is -0.0001 "
        "kg/s, below the permitted minimum -6.001e-06 kg/s"
    ) in message
    assert "stage mass balance failed at t = 0.0, stage 2" not in message
    with pytest.raises(InitializationError, match="stage flow"):
        cr.validate_solved_flows()
    assert state_snapshot(m) == before

    # An unavailable outlet does not stop later stage checks. Stage 2 uses
    # stage 1 for mass balance and the inlet for its sign tolerance.
    out["OreA", 0].set_value(None)
    for k in range(4):
        stage[0, 1, "OreA", k].set_value(-1.0 if k == 1 else 0.0, skip_validation=True)
        stage[0, 2, "OreA", k].set_value(10.0 if k == 0 else 0.0)
    before = state_snapshot(m)
    with pytest.raises(InitializationError, match="has no value"):
        cr.validate_solved_flows()
    message = cr.check_solved_flows()
    assert (
        "stage flow at t = 0.0, stage 1, component 'OreA', interval 1, is -1.0 kg/s, "
        "below the permitted minimum -6.001e-06 kg/s"
    ) in message
    assert "stage mass balance failed at t = 0.0, stage 2, component 'OreA'" in message
    assert "stage flow 10.0 kg/s, incoming flow -1.0 kg/s" in message
    assert "residual 1.100e+01 kg/s, tolerance 1.001e-06 kg/s" in message
    assert state_snapshot(m) == before
    for k, flow in enumerate(FEED_REF["OreA"]):
        out["OreA", k].set_value(flow)
        for r in (1, 2):
            stage[0, r, "OreA", k].set_value(flow)
    assert cr.check_solved_flows() == ""

    out["OreA", 0].set_value(99.0)
    before = state_snapshot(m)
    message = cr.check_solved_flows()
    out_total = math.fsum([99.0, 0.0, 0.0, 6.0])
    assert "mass balance failed at t = 0.0, component 'OreA'" in message
    assert f"product flow {out_total!r} kg/s, feed flow 6.0 kg/s" in message
    assert f"residual {out_total - 6.0:.3e} kg/s, tolerance 6.001e-06 kg/s" in message
    assert "component 'OreB'" not in message
    assert state_snapshot(m) == before

    # Report unavailable or nonfinite entries by index before mass-balance arithmetic.
    failures = []
    for case_id, planted, expected in _UNAVAILABLE_SOLVED_FLOW_CASES:
        try:
            m, cr = _closed_repeated_stress_unit()
            _plant(cr, planted)
            with pytest.raises(
                InitializationError, match=f"^{re.escape(f'{cr.name}: {expected}')}$"
            ):
                cr.validate_solved_flows()
        except (Exception, pytest.fail.Exception) as exc:
            failures.append(f"{case_id}: {exc}")
    assert not failures, "\n".join(failures)

    # After an unavailable stage 1 row, stage 3 still checks against stage 2.
    m = flowsheet()
    cr = crusher(m, **dict(SINGLE_STRESS_REF, n_stress_events=4))
    fix_feed(cr.properties_in[0], FEED_REF)
    for obj in cr.psd_stage.values():
        obj.set_value(0.0)
    cr.psd_stage[0, 1, "OreA", 0].set_value(None)
    cr.psd_stage[0, 2, "OreA", 1].set_value(-1.0, skip_validation=True)
    cr.psd_stage[0, 3, "OreA", 0].set_value(10.0)
    before = state_snapshot(m)
    with pytest.raises(InitializationError, match="has no value"):
        cr.validate_solved_flows()
    message = cr.check_solved_flows()
    assert (
        "stage flow at t = 0.0, stage 2, component 'OreA', interval 1, is -1.0 kg/s, "
        "below the permitted minimum -6.001e-06 kg/s"
    ) in message
    assert "stage mass balance failed at t = 0.0, stage 3, component 'OreA'" in message
    assert "stage flow 10.0 kg/s, incoming flow -1.0 kg/s" in message
    assert "residual 1.100e+01 kg/s, tolerance 1.001e-06 kg/s" in message
    assert state_snapshot(m) == before


def _optional_phase_unit(rows):
    """Build a two-mineral crusher with 100 kg/s each of unsized solid, liquid,
    and vapor. These flows do not count toward the sized-feed minimum.
    """
    m = flowsheet(
        unsized=["Inert"],
        vapors=["Air"],
        density={"OreA": 2800.0, "OreB": 4200.0, "Inert": 2000.0},
    )
    cr = crusher(m, **SINGLE_STRESS_REF)
    fix_feed(cr.properties_in[0], rows, liquid=100.0, unsized={"Inert": 100.0})
    cr.properties_in[0].flow_mass_vapor_comp["Air"].fix(100.0)
    return m, cr


@pytest.mark.unit
def test_sized_feed_floor_is_strict_to_validate_and_inclusive_to_report():
    # Test adjacent floating-point values around the 1e-6 kg/s sized-feed minimum.
    floor = 1e-6
    m, cr = _optional_phase_unit(FEED_REF)
    before = state_snapshot(m)
    assert cr.validate_sized_feed_flow() is None
    assert cr.check_sized_feed_flow() == ""
    assert state_snapshot(m) == before
    for factor in (math.nextafter(1.0, 0.0), 1.0, math.nextafter(1.0, math.inf)):
        m, cr = _optional_phase_unit(
            {"OreA": [0.0, 0.0, 0.0, factor * floor], "OreB": [0.0] * 4}
        )
        before = state_snapshot(m)
        if factor < 1:
            with pytest.raises(ConfigurationError, match="below the minimum"):
                cr.validate_sized_feed_flow()
            # The initializer precheck records the failure without changing state.
            init = native.CrusherSolidPSDInitializer()
            with pytest.raises(ConfigurationError, match="below the minimum"):
                init.precheck(cr)
            assert init.summary[cr]["status"] == InitializationStatus.PrecheckFailed
        else:
            assert cr.validate_sized_feed_flow() is None
        report = cr.check_sized_feed_flow()
        assert bool(report) is (factor <= 1)
        if factor <= 1:
            assert "at or below the minimum" in report
        assert state_snapshot(m) == before
