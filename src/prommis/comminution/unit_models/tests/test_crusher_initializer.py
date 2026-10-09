#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Test crusher initialization, state restoration, and parameter estimation."""

import logging
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from pyomo.common.collections import ComponentMap
from pyomo.core.base.var import VarData
from pyomo.environ import (
    Block,
    Constraint,
    Var,
    value,
)
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

from idaes.core.initialization import InitializationStatus
from idaes.core.util.exceptions import ConfigurationError, InitializationError
from idaes.core.util.model_statistics import degrees_of_freedom

from prommis.comminution.functions import distributions
from prommis.comminution.functions.power_laws import numeric_specific_energy
from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models._crusher import initializer as crusher_initializer
from prommis.comminution.unit_models.tests.crusher_test_support import (
    DENSITY_TWO,
    F80_LIVE_REF,
    FEED_REF,
    PATH_CONFIGS,
    RR,
    SINGLE_STRESS_REF,
    crusher,
    fix_feed,
    flowsheet,
    one_mineral_flowsheet,
    starting_values,
    two_time_flowsheet,
)

# Use an 8 mm target whose characteristic size lies inside the mesh.
INTERIOR_P80_REF = 8.0e-3


@contextmanager
def monkeypatched(obj, name, value_):
    with patch.object(obj, name, value_):
        yield


def _raiser(exc):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


class _NonOptimalSolver:
    """A solver stub whose result reports an infeasible termination."""

    def __init__(self, *args, **kwargs):
        self.options = {}

    def solve(self, model, **kwargs):
        res = SolverResults()
        res.solver.status = SolverStatus.warning
        res.solver.termination_condition = TerminationCondition.infeasible
        return res


class _RaisingSolver(_NonOptimalSolver):
    """A solver stub whose solve call itself raises."""

    def solve(self, model, **kwargs):
        raise RuntimeError("planted")


def _snapshot(block):
    """Return Var fixed flags and values, plus Constraint active flags."""
    var_state = {v.name: (v.fixed, v.value) for v in block.component_data_objects(Var)}
    con_state = {
        c.name: c.active for c in block.component_data_objects(Constraint, active=None)
    }
    return var_state, con_state


def _flags(block):
    """Return Var fixed flags and Constraint active flags."""
    fixed = {v.name: v.fixed for v in block.component_data_objects(Var)}
    active = {
        c.name: c.active for c in block.component_data_objects(Constraint, active=None)
    }
    return fixed, active


def _without(snapshot, name):
    """Return a ``_snapshot`` result with the Var ``name`` left out."""
    var_state, con_state = snapshot
    return {k: v for k, v in var_state.items() if k != name}, con_state


@pytest.mark.component
@pytest.mark.solver
def test_success_summary_and_state_restoration():
    m = flowsheet()
    cr = crusher(m, **PATH_CONFIGS["single_stress"])
    fix_feed(cr.properties_in[0], FEED_REF)
    # Start with a free inlet to check that initialization fixes it before solving.
    cr.properties_in[0].temperature.unfix()
    entry_flags = _flags(cr)
    seen = {}

    class _Observe(native.CrusherSolidPSDInitializer):
        """Record inlet fixed status and model degrees of freedom before the solve."""

        def initialization_routine(self, model):
            seen["inlet_fixed"] = all(
                var.fixed
                for t in model.flowsheet().time
                for comp in model.properties_in[t].define_state_vars().values()
                for var in comp.values()
            )
            seen["dof"] = degrees_of_freedom(model)
            return super().initialization_routine(model)

    init = _Observe()
    init.initialize(cr)
    assert init.summary[cr]["DoF"] == 0
    assert init.summary[cr]["solver_status"] is True
    assert init.summary[cr]["status"] == InitializationStatus.Ok
    assert seen["inlet_fixed"] is True
    assert seen["dof"] == 0
    assert not cr.properties_in[0].temperature.fixed
    assert _flags(cr) == entry_flags


GUESS_KEY = "properties_out[0.0].flow_mass_sized_comp_size[OreA,1]"


@pytest.mark.component
@pytest.mark.parametrize("branch", ["custom", "dof"])
def test_precheck_failure_restores_entry_state(branch):
    if branch == "custom":
        # The later feed is already in the finest bin. The table would make it
        # coarser, so initialization must leave both time points unseeded.
        m = two_time_flowsheet(species=["OreA", "OreB"], density=DENSITY_TWO)
        cr = crusher(m, bond_work_index=14.0, **PATH_CONFIGS["tabular"])
        fix_feed(cr.properties_in[0.0], FEED_REF)
        fix_feed(
            cr.properties_in[1.0],
            {"OreA": [6.0, 0.0, 0.0, 0.0], "OreB": [4.0, 0.0, 0.0, 0.0]},
        )
    else:
        m, cr = _ok_unit()
        cr.bond_work_index.unfix()
    # A failed precheck must restore this inlet variable's unfixed status.
    cr.properties_in[0].temperature.unfix()
    row = cr.find_component(GUESS_KEY)
    assert not row.fixed and row.value != 0.75
    entry = _snapshot(cr)
    init = native.CrusherSolidPSDInitializer()
    # Pass a temporary log level to check that failure resets it.
    if branch == "custom":
        with pytest.raises(ConfigurationError, match="no-coarsening") as exc:
            init.initialize(
                cr, initial_guesses={GUESS_KEY: 0.75}, output_level=logging.DEBUG
            )
        assert "t = 1.0" in str(exc.value)
        assert "change the table or the feed" in str(exc.value)
        # The calculation fails after the DoF check, so the summary keeps DoF = 0.
        expected_summary = {"status": InitializationStatus.PrecheckFailed, "DoF": 0}
    else:
        with pytest.raises(InitializationError, match="Degrees of freedom"):
            init.initialize(
                cr, initial_guesses={GUESS_KEY: 0.75}, output_level=logging.DEBUG
            )
        expected_summary = {"status": InitializationStatus.DoF, "DoF": 1}
    assert row.value == 0.75
    assert _without(_snapshot(cr), row.name) == _without(entry, row.name)
    assert expected_summary.items() <= init.summary[cr].items()
    assert "solver_status" not in init.summary[cr]
    assert init.get_output_level() == init.config.output_level
    if branch == "custom":
        return
    # Reusing the initializer must clear the previous attempt's DoF result.
    cr.bond_work_index.fix(200.0)
    entry = _snapshot(cr)
    with pytest.raises(
        ConfigurationError,
        match=r"bond_work_index value 200\.0 kWh/t must be finite and within",
    ):
        init.initialize(cr)
    assert init.summary[cr]["status"] == InitializationStatus.PrecheckFailed
    assert "solver_status" not in init.summary[cr]
    assert "DoF" not in init.summary[cr]
    assert _snapshot(cr) == entry


@pytest.mark.unit
def test_legacy_initialize_is_refused_before_any_side_effect():
    m, cr = _ok_unit()
    # Register a block to check that the rejected legacy call leaves it active.
    cr.registered = Block()
    cr._initialization_order.append(cr.registered)
    with pytest.raises(NotImplementedError, match="CrusherSolidPSDInitializer"):
        cr.initialize()
    with pytest.raises(NotImplementedError, match="CrusherSolidPSDInitializer"):
        cr.initialize(None, outlvl=logging.DEBUG)
    assert cr.registered.active


# Required summary fields for a successful initialization.
OK_SUMMARY = {"status": InitializationStatus.Ok, "DoF": 0, "solver_status": True}


def _ok_unit():
    """Return a single-stress crusher with the reference feed fixed."""
    m = flowsheet()
    cr = crusher(m, bond_work_index=14.0, **SINGLE_STRESS_REF)
    fix_feed(cr.properties_in[0], FEED_REF)
    return m, cr


@pytest.mark.component
@pytest.mark.solver
def test_solved_flow_failure_reports_error():
    _, cr = _ok_unit()
    init = native.CrusherSolidPSDInitializer()
    init.initialize(cr)
    # Patch the class method so flow validation fails after the solve.
    with monkeypatched(
        type(cr), "validate_solved_flows", _raiser(ConfigurationError("planted"))
    ):
        with pytest.raises(ConfigurationError, match="planted"):
            init.initialize(cr)
    assert {"status": InitializationStatus.Error, "DoF": 0}.items() <= init.summary[
        cr
    ].items()
    assert "solver_status" not in init.summary[cr]
    # An inconsistent fixed outlet row makes the real solve fail before
    # flow validation.
    cr.bond_work_index.unfix()
    row = cr.properties_out[0].flow_mass_sized_comp_size["OreA", 0]
    row.fix(99.0)
    assert degrees_of_freedom(cr) == 0
    with pytest.raises(InitializationError, match="did not reach optimal termination"):
        init.initialize(cr)
    assert {
        "status": InitializationStatus.Failed,
        "DoF": 0,
        "solver_status": False,
    }.items() <= init.summary[cr].items()
    assert row.fixed and value(row) == 99.0


@pytest.mark.unit
def test_initializer_raises_on_nonoptimal_solve(monkeypatch):
    m = one_mineral_flowsheet()
    cr = crusher(m, psd_method="tabular", tabular_psd=[0.25, 0.25, 0.25, 0.25])
    fix_feed(cr.properties_in[0], {"OreA": [0.0, 0.0, 0.0, 10.0]})
    # A nonoptimal solve must restore this inlet variable's unfixed status.
    cr.properties_in[0].temperature.unfix()
    entry_flags = _flags(cr)
    entry_fixed = {v.name: v.value for v in cr.component_data_objects(Var) if v.fixed}
    monkeypatch.setattr(crusher_initializer, "get_solver", _NonOptimalSolver)
    init = native.CrusherSolidPSDInitializer()
    assert init.summary == {}
    assert {
        "tol": 1e-8,
        "bound_push": 1e-8,
    }.items() <= init.config.solver_options.value().items()
    with pytest.raises(InitializationError, match="did not reach optimal termination"):
        init.initialize(cr)
    assert {
        "status": InitializationStatus.Failed,
        "DoF": 0,
        "solver_status": False,
    }.items() <= init.summary[cr].items()
    assert {
        v.name: v.value for v in cr.component_data_objects(Var) if v.fixed
    } == entry_fixed
    assert _flags(cr) == entry_flags
    monkeypatch.setattr(crusher_initializer, "get_solver", _RaisingSolver)
    with pytest.raises(RuntimeError, match="planted"):
        init.initialize(cr)
    assert {"status": InitializationStatus.Error, "DoF": 0}.items() <= init.summary[
        cr
    ].items()
    assert "solver_status" not in init.summary[cr]


@pytest.mark.component
@pytest.mark.solver
def test_no_entry_fixed_var_changes_value_during_initialization():
    m = flowsheet(
        unsized=["Inert"], density=dict(DENSITY_TWO, Inert=2000.0), vapors=["Air"]
    )
    cr = crusher(m, bond_work_index=14.0, **PATH_CONFIGS["classification_recycle"])
    feed = cr.properties_in[0]
    fix_feed(feed, FEED_REF, liquid=2.0, unsized={"Inert": 0.5})
    feed.flow_mass_vapor_comp["Air"].fix(0.25)
    # Fix measured power at its calculated value; initialization must leave it unchanged.
    cr.bond_work_index.unfix()
    cr.bond_work_index.set_value(20.0)
    datum = starting_values(cr)[cr.power[0].name]
    cr.power[0].fix(datum)
    entry_fixed = ComponentMap(
        (v, v.value) for v in cr.component_data_objects(Var) if v.fixed
    )
    assert cr.power[0] in entry_fixed
    real_set_value = VarData.set_value

    def _guarded(var, val, skip_validation=False):
        if var in entry_fixed and val != entry_fixed[var]:
            raise AssertionError(
                f"attempted to set entry-fixed Var {var.name} "
                f"from {entry_fixed[var]!r} to {val!r}"
            )
        return real_set_value(var, val, skip_validation=skip_validation)

    init = native.CrusherSolidPSDInitializer()
    with patch.object(VarData, "set_value", _guarded):
        init.initialize(cr)
    assert init.summary[cr]["status"] == InitializationStatus.Ok
    for var, entry_value in entry_fixed.items():
        assert var.fixed and var.value == entry_value
    assert cr.power[0].fixed and value(cr.power[0]) == datum
    assert value(cr.bond_work_index) == pytest.approx(20.0, rel=1e-6)
    out = cr.properties_out[0]
    assert value(out.flow_mass_unsized_comp["Inert"]) == pytest.approx(0.5, rel=1e-10)
    assert value(out.flow_mass_liquid_comp["H2O"]) == pytest.approx(2.0, rel=1e-10)
    assert value(out.flow_mass_vapor_comp["Air"]) == pytest.approx(0.25, rel=1e-10)
    # Writing the calculated power value would evade the guard. The second
    # unit fixes outlet temperature away from the inlet to expose a write,
    # then deactivates its pass-through equation to keep zero DoF.
    m2 = flowsheet(
        unsized=["Inert"], density=dict(DENSITY_TWO, Inert=2000.0), vapors=["Air"]
    )
    cr2 = crusher(m2, bond_work_index=14.0, **PATH_CONFIGS["classification_recycle"])
    feed2 = cr2.properties_in[0]
    fix_feed(feed2, FEED_REF, liquid=2.0, unsized={"Inert": 0.5})
    feed2.flow_mass_vapor_comp["Air"].fix(0.25)
    outlet_temperature = cr2.properties_out[0].temperature
    outlet_temperature.fix(310.0)
    cr2.temperature_eqn[0].deactivate()
    entry_fixed.update((v, v.value) for v in cr2.component_data_objects(Var) if v.fixed)
    assert outlet_temperature in entry_fixed
    init2 = native.CrusherSolidPSDInitializer()
    with patch.object(VarData, "set_value", _guarded):
        init2.initialize(cr2)
    assert init2.summary[cr2]["status"] == InitializationStatus.Ok
    assert outlet_temperature.fixed and value(outlet_temperature) == 310.0


# Test recovery of a work index near its lower bound from two initial guesses.
_WI_LO, _WI_HI = native.WORK_INDEX_BOUNDS
TWIN_CASES = [
    pytest.param(path, shape, implied, id=f"{path}-{shape}-{implied}")
    for path, shape, implied in [("distribution_power_spec", RR, _WI_LO + 0.001)]
]


def _estimation_twin(path, shape, implied, wi_entry):
    """Return a model, crusher, and fixed size measurement for two time points.

    ``implied`` sets measured power; ``wi_entry`` sets the initial work index.
    """
    m = two_time_flowsheet(species=["OreA", "OreB"], density=DENSITY_TWO)
    cfg = dict(PATH_CONFIGS[path], distribution_shape=shape)
    cr = crusher(m, bond_work_index=14.0, **cfg)
    for t in (0.0, 1.0):
        fix_feed(cr.properties_in[t], FEED_REF)
    # Compute the measured duty directly; power has not been fixed yet.
    measurement = (cr.d_63 if shape == RR else cr.d_max)[0.0]
    measurement.fix(distributions.shape_size_from_p80(shape, INTERIOR_P80_REF, 1.5))
    duty = (
        3.6
        * 10.0
        * numeric_specific_energy("bond", implied, INTERIOR_P80_REF, F80_LIVE_REF)
    )
    for t in (0.0, 1.0):
        cr.power[t].fix(duty)
    cr.bond_work_index.unfix()
    cr.bond_work_index.set_value(wi_entry)
    return m, cr, measurement


@pytest.mark.component
@pytest.mark.solver
@pytest.mark.parametrize("path,shape,implied", TWIN_CASES)
def test_two_time_studies_recover_the_planted_work_index(path, shape, implied):
    twins = {}
    for wi_entry in (14.0, 60.0):
        m, cr, measurement = _estimation_twin(path, shape, implied, wi_entry)
        shape_size = cr.d_63 if shape == RR else cr.d_max
        assert measurement.fixed and not cr.bond_work_index.fixed
        assert cr.power[0.0].fixed and cr.power[1.0].fixed
        assert not shape_size[1.0].fixed
        assert value(cr.bond_work_index) == wi_entry
        assert degrees_of_freedom(cr) == 0
        entry_fixed = {
            v.name: v.value for v in cr.component_data_objects(Var) if v.fixed
        }
        measured_value = value(measurement)
        init = native.CrusherSolidPSDInitializer()
        init.initialize(cr)
        twins[wi_entry] = {
            "m": m,
            "cr": cr,
            "measurement": measurement,
            "measured_value": measured_value,
            "entry_fixed": entry_fixed,
            "init": init,
        }
    for wi_entry, twin in twins.items():
        cr, init = twin["cr"], twin["init"]
        assert twin["measurement"].fixed
        assert value(twin["measurement"]) == twin["measured_value"]
        assert {
            v.name: v.value for v in cr.component_data_objects(Var) if v.fixed
        } == twin["entry_fixed"]
        assert OK_SUMMARY.items() <= init.summary[cr].items()
        assert value(cr.bond_work_index) == pytest.approx(implied, rel=1e-6)
        assert abs(value(cr.bond_work_index) - wi_entry) >= 1e-3
    cr_a, cr_b = twins[14.0]["cr"], twins[60.0]["cr"]
    got = value((cr_b.d_63 if shape == RR else cr_b.d_max)[1.0])
    exp = value((cr_a.d_63 if shape == RR else cr_a.d_max)[1.0])
    assert got == pytest.approx(exp, rel=1e-6)
    pp = cr_a.config.property_package
    for s in pp.sized_solid_list:
        m_s = value(cr_a.properties_in[1.0].flow_mass_sized_comp[s])
        for k in pp.size_interval_set:
            row_a = value(cr_a.properties_out[1.0].flow_mass_sized_comp_size[s, k])
            row_b = value(cr_b.properties_out[1.0].flow_mass_sized_comp_size[s, k])
            # Allow absolute and feed-relative differences between separate solves.
            assert abs(row_b - row_a) <= 1e-9 + 1e-6 * abs(m_s)
    # Reinitialize after doubling the feed and power, then check the outlet bin total.
    out = cr_a.properties_out[1.0]
    before = value(out.flow_mass_size[1])
    for component in cr_a.properties_in[1.0].define_state_vars().values():
        if component.local_name.startswith("flow_mass"):
            for var in component.values():
                var.fix(2.0 * value(var))
    cr_a.power[1.0].fix(2.0 * value(cr_a.power[1.0]))
    assert degrees_of_freedom(cr_a) == 0
    twins[14.0]["init"].initialize(cr_a)
    assert value(out.flow_mass_size[1]) == pytest.approx(2.0 * before, rel=1e-6)
    assert value(out.flow_mass_size[1]) == pytest.approx(
        sum(value(out.flow_mass_sized_comp_size[s, 1]) for s in pp.sized_solid_list),
        rel=1e-12,
    )


@pytest.mark.component
@pytest.mark.solver
def test_weak_positive_sensitivity_has_no_power_floor():
    m = one_mineral_flowsheet()
    cr = crusher(
        m,
        bond_work_index=14.0,
        psd_method="tabular",
        tabular_psd=[1.0e-3, 0.0, 0.0, 1.0 - 1.0e-3],
    )
    fix_feed(cr.properties_in[0], {"OreA": [0.0, 0.0, 0.0, 10.0]})
    assert degrees_of_freedom(m) == 0
    native.CrusherSolidPSDInitializer().initialize(cr)
    measured = value(cr.power[0])
    assert 0.0 < measured < 1.0e-2
    cr.bond_work_index.unfix()
    cr.bond_work_index.set_value(7.0)
    cr.power[0].fix(measured)
    assert degrees_of_freedom(m) == 0
    native.CrusherSolidPSDInitializer().initialize(cr)
    # Power changes little with work index here, so allow a 1e-5 relative error.
    assert value(cr.bond_work_index) == pytest.approx(14.0, rel=1e-5)
    assert value(cr.power[0]) == measured


@pytest.mark.component
@pytest.mark.solver
def test_target_product_p80_changed_after_build_is_used_and_rechecked():
    m = flowsheet()
    cr = crusher(m, **PATH_CONFIGS["distribution_size_spec"])
    fix_feed(cr.properties_in[0], FEED_REF)
    cr.target_product_p80 = 4e-3
    native.CrusherSolidPSDInitializer().initialize(cr)
    assert value(cr.product_p80[0]) == pytest.approx(4e-3, rel=1e-9)
    cr.target_product_p80 = 20e-3
    init = native.CrusherSolidPSDInitializer()
    with pytest.raises(ConfigurationError, match="attainable mesh band"):
        init.initialize(cr)
    assert init.summary[cr]["status"] == InitializationStatus.PrecheckFailed
