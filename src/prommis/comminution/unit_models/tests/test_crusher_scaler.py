#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Test CrusherSolidPSDScaler factors and solved-model conditioning."""

import math
import re

import pytest

from pyomo.common.collections import ComponentMap
from pyomo.environ import Constraint, Var, assert_optimal_termination, value

from idaes.core.scaling.util import (
    del_scaling_factor,
    get_scaling_factor,
    jacobian_cond,
    list_unscaled_constraints,
    list_unscaled_variables,
)
from idaes.core.solvers import get_solver
from idaes.core.util import DiagnosticsToolbox
from idaes.core.util.exceptions import ConfigurationError

from prommis.comminution.properties.solid_psd_properties import SolidPSDScaler
from prommis.comminution.unit_models.crusher import CrusherSolidPSDScaler
from prommis.comminution.unit_models.tests.crusher_test_support import (
    DENSITY_TWO,
    FEED_REF,
    GGS,
    PATH_CONFIGS,
    PENDULUM_REF,
    crusher,
    feed_percentile,
    fix_feed,
    flowsheet,
    two_time_flowsheet,
)

# Test each different set of model components; single_stress matches another case.
SCALING_CONFIGS = {k: v for k, v in PATH_CONFIGS.items() if k != "single_stress"}
SCALING_CONFIGS["pendulum"] = PENDULUM_REF
GGS_SIZE_SPEC = dict(PATH_CONFIGS["distribution_size_spec"], distribution_shape=GGS)
FEED_POS = {"OreA": [0.02, 1.0, 1.98, 3.0], "OreB": [0.4, 0.6, 1.0, 2.0]}
# t10 = 20 is a tabulated Ecs row; interpolation over size still applies.
PENDULUM_T10_20 = dict(
    PENDULUM_REF, breakage_params=dict(PENDULUM_REF["breakage_params"], t10=20.0)
)
# The size-spec path has a separate conditioning test.
SOLVE_TEST_CONFIGS = {
    k: v for k, v in SCALING_CONFIGS.items() if k != "distribution_size_spec"
}
# Keep tabular outlet flows off the zero bound during diagnostics.
SOLVE_TEST_CONFIGS["tabular"] = dict(
    SCALING_CONFIGS["tabular"], tabular_psd=[0.2, 0.4, 0.3, 0.1]
)
# The power-spec configuration needs a fixed power to solve.
FIXED_POWER_KW = {"distribution_power_spec": 6.0}


def build_crusher(cfg, feed=None, liquid=1.0, all_phases=False):
    """Return the model and crusher with ``feed`` (or ``FEED_REF``) fixed."""
    if all_phases:
        m = flowsheet(
            unsized=["Inert"],
            vapors=["Air"],
            density=dict(DENSITY_TWO, Inert=2000.0),
        )
    else:
        m = flowsheet()
    cfg = dict(cfg)
    if cfg.get("power_law") != "pendulum":
        cfg.setdefault("bond_work_index", 12.0)
    cr = crusher(m, **cfg)
    fix_feed(
        cr.properties_in[0],
        FEED_REF if feed is None else feed,
        liquid=liquid,
        unsized={"Inert": 0.5} if all_phases else None,
    )
    if all_phases:
        cr.properties_in[0].flow_mass_vapor_comp["Air"].fix(0.05)
    return m, cr


def scale_initialize_solve(cfg, power_kw=None, all_phases=False):
    """Scale, initialize and solve a crusher on ``FEED_POS``; return both."""
    # Positive feed flows keep fixed inlet variables away from zero bounds.
    m, cr = build_crusher(cfg, feed=FEED_POS, all_phases=all_phases)
    if power_kw is not None:
        cr.power[0].fix(power_kw)
    cr.default_scaler().scale_model(cr)
    cr.default_initializer().initialize(cr)
    assert_optimal_termination(get_solver("ipopt_v2").solve(m))
    return m, cr


class _FallbackMarkerScaler(CrusherSolidPSDScaler):
    """Use 0.125 when the input-based power estimate is unavailable."""

    INPUT_BASED_DEFAULT_POWER_FACTOR = 0.125


class _MarkedStateScaler(SolidPSDScaler):
    """Set OreA bin 3 to a 7 kg/s nominal for override tests."""

    def variable_scaling_routine(self, model, overwrite=False, submodel_scalers=None):
        super().variable_scaling_routine(model, overwrite=overwrite)
        self.set_variable_scaling_factor(
            model.flow_mass_sized_comp_size["OreA", 3], 1 / 7.0, overwrite=True
        )


def _f80_um(cr, t=0):
    """Feed 80%-passing size in um"""
    return feed_percentile(cr, t) * 1e6


def _bond_power_kw(p80_um, f80_um, sized_kg_s=10.0, work_index=12.0):
    """Return Bond power (kW) from P80/F80 (um), flow (kg/s), and Wi (kWh/t)."""
    return (
        3.6
        * sized_kg_s
        * 10.0
        * work_index
        * (1 / math.sqrt(p80_um) - 1 / math.sqrt(f80_um))
    )


def _factors(unit, ctype):
    return {
        c.name: get_scaling_factor(c)
        for c in unit.component_data_objects(ctype, descend_into=True)
    }


@pytest.mark.unit
@pytest.mark.parametrize("config_name", sorted(SCALING_CONFIGS))
def test_scale_model_covers_unfixed_vars_and_active_constraints(config_name):
    m, cr = build_crusher(SCALING_CONFIGS[config_name], all_phases=True)
    cr.default_scaler().scale_model(cr)
    assert list_unscaled_variables(cr) == []
    assert list_unscaled_constraints(cr) == []
    # Fixed inlet flows also receive factors; only fixed model parameters do not.
    allowed = {"bond_work_index"}
    if config_name == "pendulum":
        allowed |= {"pendulum_power_factor", "no_load_power"}
    unscaled_fixed = {
        v.parent_component().local_name
        for v in list_unscaled_variables(cr, include_fixed=True)
    }
    assert unscaled_fixed == allowed


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, working, balance",
    [
        ("classification_recycle", "recycle_load", "recycle_load_eqn"),
        ("repeated_stress", "psd_stage", "psd_stage_eqn"),
        ("tabular", None, "no_coarsening_comp_eqn"),
    ],
)
def test_flow_factors_use_component_totals(path, working, balance):
    # Every inlet bin exceeds the state scaler's floor, so its nominal equals its flow.
    feed = {"OreA": [0.4, 0.5, 29.1, 70.0], "OreB": [2e-5, 3e-5, 5e-5, 4e-5]}
    m, cr = build_crusher(PATH_CONFIGS[path], feed=feed)
    cr.default_scaler().scale_model(cr)
    # Crushing can move mass between bins, so scale each outlet bin from its
    # component's total inlet flow rather than the matching inlet bin.
    component_total = {"OreA": 100.0, "OreB": 1.4e-4}
    out = cr.properties_out[0].flow_mass_sized_comp_size
    for (s, k), var in out.items():
        assert get_scaling_factor(var) == pytest.approx(
            1 / component_total[s], rel=1e-12, abs=0
        ), (s, k)
    total = 100.00014  # sized feed, kg/s
    if working is not None:
        # OreB's total exceeds the working-flow floor, so no component is floored.
        for idx, var in getattr(cr, working).items():
            assert get_scaling_factor(var) == pytest.approx(
                1 / component_total[idx[-2]], rel=1e-12, abs=0
            ), idx
    for con in [*cr.product_psd_eqn.values(), *getattr(cr, balance).values()]:
        assert get_scaling_factor(con) == pytest.approx(
            1 / total, rel=1e-12, abs=0
        ), con.name


def _power_case(case):
    """Return (model, crusher, exact factor, hand-checked factor)."""
    recycle = PATH_CONFIGS["classification_recycle"]
    if case == "matrix_path_reduction_ratio_4":
        m, cr = build_crusher(recycle)
        f80 = _f80_um(cr)
        # By hand: all feed in the 8-16 mm bin gives F80 = 14.4 mm, and
        # P80 = F80 / 4, so E = 120 / sqrt(14400) = 1 kWh/t and P = 36 kW.
        return m, cr, 1 / _bond_power_kw(f80 / 4, f80), 1 / 36.0
    if case == "fixed_work_index":
        m, cr = build_crusher(recycle)
        cr.bond_work_index.fix(24.0)  # the fixed value wins over the configured 12
        f80 = _f80_um(cr)
        # By hand: doubling Wi doubles E to 2 kWh/t, so P = 72 kW.
        return m, cr, 1 / _bond_power_kw(f80 / 4, f80, work_index=24.0), 1 / 72.0
    if case == "kick_power_law":
        m, cr = build_crusher(dict(recycle, power_law="kick"))
        # Kick: E = C_K log10(F80/P80) with C_K = 1.151 x 10 Wi / sqrt(50000),
        # and F80/P80 = 4 exactly. By hand: E = 0.6177 x 0.60206 = 0.37189 kWh/t,
        # P = 36 x 0.37189 = 13.388 kW.
        c_k = 1.151 * 10.0 * 12.0 / math.sqrt(50000.0)
        return m, cr, 1 / (36.0 * c_k * math.log10(4.0)), 1 / 13.388
    if case == "finest_attainable_p80":
        fine = {"OreA": [5.0, 5.0, 0.0, 0.0], "OreB": [0.0] * 4}
        m, cr = build_crusher(recycle, feed=fine)
        # By hand: F80 = 2 + 0.3/0.5 x 2 = 3.2 mm, and F80/4 = 0.8 mm is below
        # the finest attainable P80, 1 + 0.8 x 1 = 1.8 mm. So
        # E = 120 (1/sqrt(1800) - 1/sqrt(3200)) = 0.70711 kWh/t, P = 25.456 kW.
        return m, cr, 1 / _bond_power_kw(1800.0, _f80_um(cr)), 1 / 25.456
    if case == "size_spec_target_p80":
        m, cr = build_crusher(GGS_SIZE_SPEC)
        # By hand: E = 120 (1/sqrt(5000) - 1/120) = 0.69706 kWh/t, P = 25.094 kW.
        return m, cr, 1 / _bond_power_kw(5000.0, _f80_um(cr)), 1 / 25.094
    if case == "tabular_table_per_component":
        cfg = dict(
            psd_method="tabular",
            tabular_psd_by_comp={
                "OreA": [0.25, 0.5, 0.25, 0.0],
                "OreB": [0.0, 0.0, 0.25, 0.75],
            },
        )
        m, cr = build_crusher(cfg)
        pp = cr.config.property_package
        # By hand: OreA 6 kg/s and OreB 4 kg/s give product bins
        # [1.5, 3.0, 2.5, 3.0] kg/s, passing 70% at 8 mm and 100% at 16 mm, so
        # P80 = 8 + 0.1/0.3 x 8 = 10.667 mm, E = 0.16190 kWh/t, P = 5.8285 kW.
        p80 = pp.size_at_passing_from_flows({None: [1.5, 3.0, 2.5, 3.0]})
        assert p80 == pytest.approx(10.667e-3, rel=1e-4)
        return m, cr, 1 / _bond_power_kw(p80 * 1e6, _f80_um(cr)), 1 / 5.8285
    if case == "coarsening_table_zero_energy":
        coarse_table = dict(psd_method="tabular", tabular_psd=[0.0, 0.0, 0.0, 1.0])
        fine = {"OreA": [10.0, 0.0, 0.0, 0.0], "OreB": [0.0] * 4}
        m, cr = build_crusher(coarse_table, feed=fine)
        # A coarser product gives negative specific energy, which is clamped to zero.
        # The power factor then uses the 1 kW scaling floor.
        return m, cr, 1.0, 1.0
    if case == "pendulum_max_ecs":
        m, cr = build_crusher(PENDULUM_T10_20)
        e_max = max(value(e) for e in cr.ecs.values())
        # By hand: at t10 = 20, Ecs is 0.8 kWh/t at 2 mm and 0.6 at 8 mm, a
        # drop of 0.1 per size doubling. The finest bin (1-2 mm, geometric mean
        # sqrt(2) mm) is half a doubling below 2 mm, so Ecs_max = 0.85 and
        # P = P0 + 3.6 A m Ecs_max = 20 + 3.6 x 1.3 x 10 x 0.85 = 59.78 kW.
        return m, cr, 1 / (20.0 + 3.6 * 1.3 * 10.0 * e_max), 1 / 59.78
    if case == "fixed_power_precedence":
        m, cr = build_crusher(PATH_CONFIGS["single_stress"])
        cr.power[0].fix(250.0)
        return m, cr, 1 / 250.0, 1 / 250.0
    if case == "fixed_negative_power":
        m, cr = build_crusher(PATH_CONFIGS["distribution_power_spec"])
        cr.power[0].fix(-5.0, skip_validation=True)
        # The magnitude sets the factor: 1 / |-5 kW|.
        return m, cr, 0.2, 0.2
    if case == "estimate_below_1_kw":
        small = {s: [0.01 * v for v in row] for s, row in FEED_REF.items()}
        m, cr = build_crusher(recycle, feed=small)
        # The 0.1 kg/s feed gives a 0.36 kW power estimate.
        # Its scaling factor uses the 1 kW floor.
        return m, cr, 1.0, 1.0
    if case == "feed_at_minimum":
        at_min = {"OreA": [0.0, 0.0, 0.0, 1e-6], "OreB": [0.0] * 4}
        m, cr = build_crusher(recycle, feed=at_min)
        # The 1e-6 kg/s feed still gives a power estimate.
        # Its scaling factor uses the 1 kW floor.
        return m, cr, 1.0, 1.0
    raise ValueError(case)


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "matrix_path_reduction_ratio_4",
        "fixed_work_index",
        "kick_power_law",
        "finest_attainable_p80",
        "size_spec_target_p80",
        "tabular_table_per_component",
        "coarsening_table_zero_energy",
        "pendulum_max_ecs",
        "fixed_power_precedence",
        "fixed_negative_power",
        "estimate_below_1_kw",
        "feed_at_minimum",
    ],
)
def test_power_factor_follows_input_estimate(case):
    m, cr, exact, hand = _power_case(case)
    # A 0.125 fallback reveals a missing estimate that the 1 kW floor could hide.
    _FallbackMarkerScaler().scale_model(cr)
    factor = get_scaling_factor(cr.power[0])
    assert factor == pytest.approx(exact, rel=1e-12, abs=0)
    assert factor == pytest.approx(hand, rel=1e-4)
    assert get_scaling_factor(cr.power_eqn[0]) == factor


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "power_spec_unfixed_power",
        "pendulum_factor_unfixed",
        "pendulum_no_load_unfixed",
        "feed_below_minimum",
        "feed_bin_unfixed",
    ],
)
def test_power_factor_falls_back_without_estimate(case):
    if case == "power_spec_unfixed_power":
        m, cr = build_crusher(PATH_CONFIGS["distribution_power_spec"])
    elif case == "pendulum_factor_unfixed":
        m, cr = build_crusher(PENDULUM_REF)
        cr.pendulum_power_factor.unfix()
    elif case == "pendulum_no_load_unfixed":
        m, cr = build_crusher(PENDULUM_REF)
        cr.no_load_power.unfix()
    elif case == "feed_below_minimum":
        below = {"OreA": [0.0, 0.0, 0.0, 5e-7], "OreB": [0.0] * 4}
        m, cr = build_crusher(PATH_CONFIGS["classification_recycle"], feed=below)
    else:
        m, cr = build_crusher(PATH_CONFIGS["tabular"])
        cr.properties_in[0].flow_mass_sized_comp_size["OreA", 3].unfix()
    _FallbackMarkerScaler().scale_model(cr)
    assert get_scaling_factor(cr.power[0]) == 0.125
    CrusherSolidPSDScaler(overwrite=True).scale_model(cr)
    assert get_scaling_factor(cr.power[0]) == 1.0


def _shape_size_case(case):
    """Return (model, crusher, shape-size Var, exact factor, hand-checked factor)."""
    if case == "size_spec_target":
        m, cr = build_crusher(GGS_SIZE_SPEC)
        # By hand: GGS P80 = d_max x 0.8**(1/n), so
        # d_max = 5e-3 / 0.8**(1/1.5) = 5.8019e-3 m.
        return m, cr, cr.d_max[0], 0.8 ** (1 / 1.5) / 5e-3, 172.355
    if case == "no_estimate_half_upper_bound":
        cfg = dict(PATH_CONFIGS["distribution_power_spec"], distribution_shape=GGS)
        m, cr = build_crusher(cfg)
        # Negative power cannot give a valid product P80.
        cr.power[0].fix(-5.0, skip_validation=True)
        # The GGS upper bound is the 16 mm top edge; half of it is 8 mm.
        return m, cr, cr.d_max[0], 1 / 8e-3, 125.0
    m, cr = build_crusher(PATH_CONFIGS["distribution_power_spec"])
    cr.power[0].fix(36.0)
    cr.bond_work_index.unfix()
    cr.bond_work_index.set_value(97.0)  # input-based mode uses the configured 12
    if case == "fixed_size_floor":
        cr.d_63[0].fix(1e-7)
        # The fixed 1e-7 m size is unchanged; scaling uses a 1e-6 m floor.
        return m, cr, cr.d_63[0], 1e6, 1e6
    # By hand: E = 36 / (3.6 x 10) = 1 kWh/t, so 1/sqrt(P80) = 1/120 + 1/sqrt(F80)
    # and P80 = 3600 um for F80 = 14400 um; RR: d_63 = P80 / ln(5)**(1/1.5).
    p80_um = 1 / (1 / 120 + 1 / math.sqrt(_f80_um(cr))) ** 2
    d_63 = p80_um * 1e-6 / math.log(5) ** (1 / 1.5)
    hand = math.log(5) ** (1 / 1.5) / 3600e-6
    return m, cr, cr.d_63[0], 1 / d_63, hand


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "size_spec_target",
        "power_spec_from_fixed_power",
        "fixed_size_floor",
        "no_estimate_half_upper_bound",
    ],
)
def test_shape_size_factor_follows_inputs(case):
    m, cr, shape_size, exact, hand = _shape_size_case(case)
    cr.default_scaler().scale_model(cr)
    factor = get_scaling_factor(shape_size)
    assert factor == pytest.approx(exact, rel=1e-12, abs=0)
    assert factor == pytest.approx(hand, rel=1e-5)


@pytest.mark.unit
def test_size_spec_rows_use_target_p80_and_unit_factor():
    m, cr = build_crusher(GGS_SIZE_SPEC)
    cr.default_scaler().scale_model(cr)
    # The 5 mm target P80 is 0.005 m, giving a numeric factor of 200.
    assert get_scaling_factor(cr.p80_shape_size_eqn[0]) == pytest.approx(
        200.0, rel=1e-12, abs=0
    )
    assert [get_scaling_factor(c) for c in cr.no_coarsening_eqn.values()] == [1.0] * 3


@pytest.mark.unit
@pytest.mark.parametrize("power_law", ["bond", "pendulum"])
def test_unfixed_parameters_use_configured_values(power_law):
    if power_law == "bond":
        m, cr = build_crusher(PATH_CONFIGS["classification_recycle"])
        cr.bond_work_index.unfix()
        cr.bond_work_index.set_value(97.0)
        cr.default_scaler().scale_model(cr)
        assert get_scaling_factor(cr.bond_work_index) == pytest.approx(
            1 / 12.0, rel=1e-12, abs=0
        )
        # The power estimate uses the configured 12, not the current value 97.
        f80 = _f80_um(cr)
        assert get_scaling_factor(cr.power[0]) == pytest.approx(
            1 / _bond_power_kw(f80 / 4, f80), rel=1e-12, abs=0
        )
        return
    cfg = dict(
        PENDULUM_REF, pendulum_power_params={"power_factor": 0.6, "no_load_power": 20.0}
    )
    m, cr = build_crusher(cfg)
    cr.pendulum_power_factor.unfix()
    cr.pendulum_power_factor.set_value(2.5)
    cr.no_load_power.unfix()
    cr.no_load_power.set_value(40.0)
    cr.default_scaler().scale_model(cr)
    # The scaling nominals are 1 for A (configured as 0.6) and 20 kW for P0.
    assert get_scaling_factor(cr.pendulum_power_factor) == 1.0
    assert get_scaling_factor(cr.no_load_power) == pytest.approx(
        1 / 20.0, rel=1e-12, abs=0
    )


@pytest.mark.unit
def test_factors_follow_each_time_point():
    m = two_time_flowsheet()
    cr = crusher(m, bond_work_index=12.0, **PATH_CONFIGS["classification_recycle"])
    sized = {0.0: 6.0, 1.0: 3.0}
    for t, m_sized in sized.items():
        fix_feed(cr.properties_in[t], {"OreA": [0.0, 0.0, 0.0, m_sized]})
    cr.default_scaler().scale_model(cr)
    for t, m_sized in sized.items():
        f80 = _f80_um(cr, t)
        power = get_scaling_factor(cr.power[t])
        assert power == pytest.approx(
            1 / _bond_power_kw(f80 / 4, f80, sized_kg_s=m_sized), rel=1e-12, abs=0
        )
        # By hand: the 1 kWh/t estimate gives 21.6 kW at t=0 and 10.8 kW at t=1.
        assert power == pytest.approx(1 / (3.6 * m_sized), rel=1e-6)
        assert get_scaling_factor(cr.power_eqn[t]) == power
        inlet = cr.properties_in[t].flow_mass_sized_comp_size
        sized_nominal = sum(1 / get_scaling_factor(v) for v in inlet.values())
        for idx, con in cr.product_psd_eqn.items():
            if idx[0] == t:
                assert get_scaling_factor(con) == pytest.approx(
                    1 / sized_nominal, rel=1e-12, abs=0
                ), con.name
        out = cr.properties_out[t].flow_mass_sized_comp_size
        # Outlet and recycle factors use each time point's OreA total, even when
        # the corresponding inlet bin is empty.
        assert get_scaling_factor(out["OreA", 0]) == pytest.approx(
            1 / sized_nominal, rel=1e-12, abs=0
        )
        assert get_scaling_factor(cr.recycle_load[t, "OreA", 3]) == pytest.approx(
            1 / sized_nominal, rel=1e-12, abs=0
        )


@pytest.mark.unit
def test_current_values_source_reads_current_values():
    m, cr = build_crusher(PATH_CONFIGS["classification_recycle"])
    cr.bond_work_index.unfix()
    cr.bond_work_index.set_value(20.0)
    cr.power[0].set_value(250.0)
    loads = {"OreA": [3.0, 0.5, 2e-6, 6.0], "OreB": [1.5, 0.25, 0.75, 4.0]}
    for s, row in loads.items():
        for k, v in enumerate(row):
            cr.recycle_load[0, s, k].set_value(v)
    out = cr.properties_out[0].flow_mass_sized_comp_size
    out["OreA", 3].set_value(0.4)
    out["OreA", 0].set_value(5e-8)
    CrusherSolidPSDScaler(factor_source="current_values").scale_model(cr)
    assert list_unscaled_variables(cr) == []
    assert list_unscaled_constraints(cr) == []
    assert get_scaling_factor(cr.bond_work_index) == pytest.approx(
        1 / 20.0, rel=1e-12, abs=0
    )
    assert get_scaling_factor(cr.power[0]) == pytest.approx(1 / 250.0, rel=1e-12, abs=0)
    # The 1% component floors (0.06 and 0.04 kg/s) exceed the
    # 1e-5 kg/s sized-feed floor used for load scaling.
    floor = {"OreA": 0.06, "OreB": 0.04}
    for s, row in loads.items():
        for k, v in enumerate(row):
            assert get_scaling_factor(cr.recycle_load[0, s, k]) == pytest.approx(
                1 / max(v, floor[s]), rel=1e-12, abs=0
            ), (s, k)
    # The 0.4 kg/s outlet bin uses its own value; the nearly empty bin
    # uses OreA's 0.06 kg/s scaling floor.
    assert get_scaling_factor(out["OreA", 3]) == pytest.approx(
        1 / 0.4, rel=1e-12, abs=0
    )
    assert get_scaling_factor(out["OreA", 0]) == pytest.approx(
        1 / 0.06, rel=1e-12, abs=0
    )

    # On a size-spec crusher the shape size and power also come from values.
    m_ggs, ggs = build_crusher(GGS_SIZE_SPEC)
    ggs.d_max[0].set_value(2.5e-3)
    ggs.power[0].set_value(0.5)
    CrusherSolidPSDScaler(factor_source="current_values").scale_model(ggs)
    assert get_scaling_factor(ggs.d_max[0]) == pytest.approx(400.0, rel=1e-12, abs=0)
    assert get_scaling_factor(ggs.power[0]) == 1.0  # 1 kW scaling floor


def _independence_case(case):
    if case == "unfixed_inlet":
        # Leave the inlet unfixed to test input-based fallback factors.
        m, cr = build_crusher(PATH_CONFIGS["tabular"])
        for v in cr.properties_in[0].component_data_objects(Var):
            v.unfix()
        return m, cr
    if case == "power_spec_unfixed_power":
        return build_crusher(PATH_CONFIGS["distribution_power_spec"])
    if case == "power_spec_fixed_power":
        m, cr = build_crusher(PATH_CONFIGS["distribution_power_spec"])
        cr.power[0].fix(36.0)
        cr.bond_work_index.unfix()
        return m, cr
    if case == "pendulum_parameters_unfixed":
        m, cr = build_crusher(PENDULUM_REF)
        cr.pendulum_power_factor.unfix()
        cr.no_load_power.unfix()
        return m, cr
    raise ValueError(case)


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "unfixed_inlet",
        "power_spec_unfixed_power",
        "power_spec_fixed_power",
        "pendulum_parameters_unfixed",
    ],
)
def test_input_based_factors_ignore_unfixed_guesses_and_write_no_values(case):
    m_plain, plain = _independence_case(case)
    m_guessed, guessed = _independence_case(case)
    for v in guessed.component_data_objects(Var, descend_into=True):
        if not v.fixed:
            v.set_value(4321.0, skip_validation=True)
    values = {
        v.name: v.value for v in guessed.component_data_objects(Var, descend_into=True)
    }
    plain.default_scaler().scale_model(plain)
    guessed.default_scaler().scale_model(guessed)
    assert {
        v.name: v.value for v in guessed.component_data_objects(Var, descend_into=True)
    } == values
    assert list_unscaled_variables(plain) == []
    assert list_unscaled_constraints(plain) == []
    for ctype in (Var, Constraint):
        assert _factors(guessed, ctype) == _factors(plain, ctype)


def _overwrite_case(case):
    """Return (model, crusher, targets): one target per setter path."""
    if case == "recycle":
        m, cr = build_crusher(PATH_CONFIGS["classification_recycle"], feed=FEED_POS)
        return (
            m,
            cr,
            [
                cr.properties_in[0].flow_mass_sized_comp_size["OreA", 3],
                # input-based: copied from inlet; current-values: outlet state scaler
                cr.properties_out[0].flow_mass_liquid_comp["H2O"],
                # input-based: component inlet total;
                # current-values: outlet value and floor
                cr.properties_out[0].flow_mass_sized_comp_size["OreA", 0],
                cr.power[0],
                cr.recycle_load[0, "OreA", 1],
                cr.liquid_passthrough_eqn[0, "H2O"],
                cr.product_psd_eqn[0, "OreA", 1],
                cr.power_eqn[0],
            ],
        )
    m, cr = build_crusher(GGS_SIZE_SPEC, feed=FEED_POS)
    cr.bond_work_index.unfix()
    return (
        m,
        cr,
        [
            cr.bond_work_index,
            cr.d_max[0],
            cr.p80_shape_size_eqn[0],
            next(iter(cr.no_coarsening_eqn.values())),  # factor of 1
        ],
    )


@pytest.mark.unit
@pytest.mark.parametrize("case", ["recycle", "size_spec"])
@pytest.mark.parametrize("factor_source", ["input_based", "current_values"])
def test_overwrite_keeps_or_replaces_preset_factors(case, factor_source):
    m, cr, targets = _overwrite_case(case)
    CrusherSolidPSDScaler(factor_source=factor_source).scale_model(cr)
    expected = [get_scaling_factor(c) for c in targets]
    sentinel = 37.0
    assert sentinel not in expected
    setter = CrusherSolidPSDScaler()
    for c in targets:
        if c.ctype is Var:
            setter.set_variable_scaling_factor(c, sentinel, overwrite=True)
        else:
            setter.set_constraint_scaling_factor(c, sentinel, overwrite=True)

    CrusherSolidPSDScaler(factor_source=factor_source).scale_model(cr)
    assert [get_scaling_factor(c) for c in targets] == [sentinel] * len(targets)

    CrusherSolidPSDScaler(factor_source=factor_source, overwrite=True).scale_model(cr)
    assert [get_scaling_factor(c) for c in targets] == pytest.approx(
        expected, rel=1e-12, abs=0
    )


@pytest.mark.unit
def test_inlet_override_reaches_outlet_and_factors_built_from_it():
    m, cr = build_crusher(PATH_CONFIGS["classification_recycle"], feed=FEED_POS)
    overrides = ComponentMap()
    overrides[cr.properties_in] = _MarkedStateScaler()
    cr.default_scaler().scale_model(cr, submodel_scalers=overrides)
    inlet = cr.properties_in[0].flow_mass_sized_comp_size
    out = cr.properties_out[0].flow_mass_sized_comp_size
    assert get_scaling_factor(inlet["OreA", 3]) == pytest.approx(
        1 / 7.0, rel=1e-12, abs=0
    )
    # Giving OreA bin 3 a 7 kg/s nominal makes OreA's total nominal 10 kg/s.
    # Outlet and recycle factors use that total.
    for var in (out["OreA", 3], out["OreA", 0], cr.recycle_load[0, "OreA", 0]):
        assert get_scaling_factor(var) == pytest.approx(
            1 / 10.0, rel=1e-12, abs=0
        ), var.name
    # The product-row nominal is 10 (OreA) + 4 (OreB) = 14 kg/s.
    assert get_scaling_factor(cr.product_psd_eqn[0, "OreB", 1]) == pytest.approx(
        1 / 14.0, rel=1e-12, abs=0
    )
    # The power estimate reads fixed feed values, not inlet factors.
    m_ref, ref = build_crusher(PATH_CONFIGS["classification_recycle"], feed=FEED_POS)
    ref.default_scaler().scale_model(ref)
    assert get_scaling_factor(cr.power[0]) == get_scaling_factor(ref.power[0])


@pytest.mark.unit
@pytest.mark.parametrize("factor_source", ["input_based", "current_values"])
def test_outlet_override_applies_in_either_mode(factor_source):
    m, cr = build_crusher(PATH_CONFIGS["classification_recycle"], feed=FEED_POS)
    overrides = ComponentMap()
    overrides[cr.properties_out] = _MarkedStateScaler(factor_source=factor_source)
    CrusherSolidPSDScaler(factor_source=factor_source).scale_model(
        cr, submodel_scalers=overrides
    )
    assert get_scaling_factor(
        cr.properties_out[0].flow_mass_sized_comp_size["OreA", 3]
    ) == pytest.approx(1 / 7.0, rel=1e-12, abs=0)
    assert get_scaling_factor(
        cr.properties_in[0].flow_mass_sized_comp_size["OreA", 3]
    ) == pytest.approx(1 / 3.0, rel=1e-12, abs=0)


@pytest.mark.unit
def test_constraint_routine_names_a_missing_power_factor():
    m, cr = build_crusher(PATH_CONFIGS["classification_recycle"])
    scaler = CrusherSolidPSDScaler()
    scaler.variable_scaling_routine(cr)
    del_scaling_factor(cr.power[0])
    with pytest.raises(ConfigurationError, match=re.escape(cr.power[0].name)):
        scaler.constraint_scaling_routine(cr)


@pytest.mark.component
@pytest.mark.solver
def test_scaling_improves_size_spec_conditioning():
    m, cr = scale_initialize_solve(GGS_SIZE_SPEC)
    # Some no-coarsening rows use the P80 margin, making their Jacobian rows
    # parallel to the P80 constraint; diagnostics ignores those known pairs.
    DiagnosticsToolbox(m).assert_no_numerical_warnings(ignore_parallel_components=True)
    scaled = jacobian_cond(m, scaled=True)
    assert scaled < jacobian_cond(m, scaled=False)
    # Measured conditioning number 28.
    assert scaled < 1e3


@pytest.mark.component
@pytest.mark.solver
@pytest.mark.parametrize("config_name", sorted(SOLVE_TEST_CONFIGS))
def test_scaled_configs_solve_without_numerical_warnings(config_name):
    # Some configurations are already well conditioned without scaling, so check
    # numerical warnings and bound the scaled condition number.
    m, cr = scale_initialize_solve(
        SOLVE_TEST_CONFIGS[config_name],
        power_kw=FIXED_POWER_KW.get(config_name),
        all_phases=True,
    )
    DiagnosticsToolbox(m).assert_no_numerical_warnings()
    assert jacobian_cond(m, scaled=True) < 1e3


@pytest.mark.component
@pytest.mark.solver
def test_size_spec_scaling_holds_after_large_feed_changes():
    # Keep the original scaling factors when all inlet flows are raised or
    # lowered by a factor of 100.
    m, cr = scale_initialize_solve(GGS_SIZE_SPEC)
    feed = [
        (var, value(var))
        for var in SolidPSDScaler.iter_flow_var_data(cr.properties_in[0])
    ]
    for multiple in (100.0, 0.01):
        for var, flow in feed:
            var.fix(multiple * flow)
        assert_optimal_termination(get_solver("ipopt_v2").solve(m))
        # The same P80-margin rows remain parallel after the feed changes.
        DiagnosticsToolbox(m).assert_no_numerical_warnings(
            ignore_parallel_components=True
        )
