#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Tests for tabular crusher products and per-component no-coarsening."""

import pytest

from pyomo.environ import value

from idaes.core.initialization import InitializationStatus
from idaes.core.util.exceptions import ConfigurationError, InitializationError
from idaes.core.util.model_statistics import degrees_of_freedom

from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models.tests.crusher_test_support import (
    ZERO_POWER_SOLVER_OPTIONS,
    solved_flow_tolerance,
    crusher,
    fix_feed,
    flowsheet,
    one_mineral_flowsheet,
)


@pytest.mark.component
@pytest.mark.solver
def test_individual_tabular_tables_and_passthrough():
    tables = {
        "OreA": [0.75, 0.25, 0.0, 0.0],
        "OreB": [0.25, 0.5, 0.25, 0.0],
        "OreC": [0.5, 0.5, 0.0, 0.0],
    }
    m = flowsheet(
        species=["OreA", "OreB", "OreC"],
        density={"OreA": 2800.0, "OreB": 4200.0, "OreC": 3000.0, "Inert": 2000.0},
        unsized=["Inert"],
    )
    cr = crusher(
        m, psd_method="tabular", tabular_psd_by_comp=tables, bond_work_index=14.0
    )
    fix_feed(
        cr.properties_in[0],
        {"OreA": [0, 0, 0, 10.0], "OreB": [0, 0, 0, 20.0], "OreC": [0, 0, 0, 0.0]},
        liquid=50.0,
        unsized={"Inert": 7.0},
    )
    native.CrusherSolidPSDInitializer(
        solver_options=ZERO_POWER_SOLVER_OPTIONS
    ).initialize(cr)

    for s, oracle, m_s in (
        ("OreA", [7.5, 2.5, 0.0, 0.0], 10.0),
        ("OreB", [5.0, 10.0, 5.0, 0.0], 20.0),
    ):
        for k, exp in enumerate(oracle):
            assert value(
                cr.properties_out[0].flow_mass_sized_comp_size[s, k]
            ) == pytest.approx(exp, abs=solved_flow_tolerance(m_s))
    for k in range(4):
        assert (
            abs(value(cr.properties_out[0].flow_mass_sized_comp_size["OreC", k]))
            <= 1e-12
        )
    out = cr.properties_out[0]
    assert value(out.flow_mass_unsized_comp["Inert"]) == pytest.approx(7.0, rel=1e-10)
    assert value(out.flow_mass_liquid_comp["H2O"]) == pytest.approx(50.0, rel=1e-10)
    assert value(out.temperature) == pytest.approx(298.15, rel=1e-10)
    assert value(out.pressure) == pytest.approx(101325.0, rel=1e-10)
    # When OreC receives feed, the crusher applies OreC's own table.
    for k, v in enumerate([0, 0, 0, 5.0]):
        cr.properties_in[0].flow_mass_sized_comp_size["OreC", k].fix(v)
    native.CrusherSolidPSDInitializer().initialize(cr)
    for k, exp in enumerate([2.5, 2.5, 0.0, 0.0]):
        assert value(
            cr.properties_out[0].flow_mass_sized_comp_size["OreC", k]
        ) == pytest.approx(exp, abs=solved_flow_tolerance(5.0))


def _two_interval_flowsheet():
    return flowsheet(edges=[1e-3, 4e-3, 16e-3])


@pytest.mark.component
@pytest.mark.solver
def test_per_mineral_no_coarsening():
    feed = {"OreA": [90.0, 10.0], "OreB": [0.0, 100.0]}
    # The shared table leaves 60 kg/s of OreA below the first boundary,
    # versus 90 kg/s in its feed. The mixture is finer overall, but OreA is
    # coarser and must be rejected.
    m = _two_interval_flowsheet()
    cr = crusher(m, psd_method="tabular", tabular_psd=[0.6, 0.4], bond_work_index=14.0)
    fix_feed(cr.properties_in[0], feed)
    with pytest.raises(ConfigurationError, match=r"no-coarsening.*OreA"):
        native.CrusherSolidPSDInitializer().initialize(cr)
    # The same rule applies to component-specific tables.
    m = _two_interval_flowsheet()
    cr = crusher(
        m,
        psd_method="tabular",
        tabular_psd_by_comp={"OreA": [0.6, 0.4], "OreB": [0.6, 0.4]},
        bond_work_index=14.0,
    )
    fix_feed(cr.properties_in[0], feed)
    with pytest.raises(ConfigurationError, match=r"no-coarsening.*OreA"):
        native.CrusherSolidPSDInitializer().initialize(cr)
    # An unchanged OreA is a valid boundary case.
    m = _two_interval_flowsheet()
    cr = crusher(
        m,
        psd_method="tabular",
        tabular_psd_by_comp={"OreA": [0.875, 0.125], "OreB": [0.6, 0.4]},
        bond_work_index=14.0,
    )
    fix_feed(cr.properties_in[0], {"OreA": [87.5, 12.5], "OreB": [0.0, 100.0]})
    init = native.CrusherSolidPSDInitializer(solver_options=ZERO_POWER_SOLVER_OPTIONS)
    init.initialize(cr)
    assert init.summary[cr]["status"] == InitializationStatus.Ok
    for s, oracle, m_s in (
        ("OreA", [87.5, 12.5], 100.0),
        ("OreB", [60.0, 40.0], 100.0),
    ):
        for k, exp in enumerate(oracle):
            assert value(
                cr.properties_out[0].flow_mass_sized_comp_size[s, k]
            ) == pytest.approx(exp, abs=solved_flow_tolerance(m_s))
    # Changing the feed makes the existing product coarser; validation
    # and a new initialization both reject it.
    m = _two_interval_flowsheet()
    cr = crusher(
        m,
        psd_method="tabular",
        tabular_psd_by_comp={"OreA": [0.6, 0.4], "OreB": [0.6, 0.4]},
        bond_work_index=14.0,
    )
    fix_feed(cr.properties_in[0], {"OreA": [0.0, 100.0], "OreB": [0.0, 100.0]})
    native.CrusherSolidPSDInitializer().initialize(cr)
    for k, v in enumerate([90.0, 10.0]):
        cr.properties_in[0].flow_mass_sized_comp_size["OreA", k].fix(v)
    assert value(cr.properties_out[0].flow_mass_sized_comp_size["OreA", 0]) - value(
        cr.properties_in[0].flow_mass_sized_comp_size["OreA", 0]
    ) == pytest.approx(-30.0, abs=solved_flow_tolerance(100.0))
    with pytest.raises(InitializationError, match=r"no-coarsening.*OreA"):
        cr.validate_solved_flows()
    with pytest.raises(ConfigurationError, match=r"no-coarsening.*OreA"):
        native.CrusherSolidPSDInitializer().initialize(cr)
    # A product that gains mass in finer intervals is allowed.
    m = one_mineral_flowsheet()
    cr = crusher(
        m, psd_method="tabular", tabular_psd=[0.2, 0.4, 0.4, 0.0], bond_work_index=14.0
    )
    fix_feed(cr.properties_in[0], {"OreA": [0.0, 0.0, 50.0, 50.0]})
    init = native.CrusherSolidPSDInitializer(solver_options=ZERO_POWER_SOLVER_OPTIONS)
    init.initialize(cr)
    assert init.summary[cr]["status"] == InitializationStatus.Ok
    for k, exp in enumerate([20.0, 40.0, 40.0, 0.0]):
        assert value(
            cr.properties_out[0].flow_mass_sized_comp_size["OreA", k]
        ) == pytest.approx(exp, abs=solved_flow_tolerance(100.0))
    # A lower product P80 is not enough: below the first boundary, product
    # flow is 40 kg/s versus 50 kg/s of feed.
    m = one_mineral_flowsheet()
    cr = crusher(
        m, psd_method="tabular", tabular_psd=[0.4, 0.6, 0.0, 0.0], bond_work_index=14.0
    )
    fix_feed(cr.properties_in[0], {"OreA": [50.0, 0.0, 0.0, 50.0]})
    with pytest.raises(ConfigurationError, match=r"no-coarsening.*OreA"):
        native.CrusherSolidPSDInitializer().initialize(cr)


@pytest.mark.component
@pytest.mark.solver
def test_one_interval_tabular_operates_through_the_initializer():
    # One interval preserves the feed and yields zero crushing power.
    m = one_mineral_flowsheet(edges=[1e-3, 2e-3])
    cr = crusher(m, psd_method="tabular", tabular_psd=[1.0])
    fix_feed(cr.properties_in[0], {"OreA": [5.0]})
    assert degrees_of_freedom(cr) == 0
    init = native.CrusherSolidPSDInitializer(solver_options=ZERO_POWER_SOLVER_OPTIONS)
    assert init.initialize(cr) == InitializationStatus.Ok
    cr.validate_solved_flows()
    assert value(
        cr.properties_out[0].flow_mass_sized_comp_size["OreA", 0]
    ) == pytest.approx(5.0, abs=solved_flow_tolerance(5.0))
    assert value(cr.power[0]) == pytest.approx(0.0, abs=1e-9)
