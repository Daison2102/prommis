#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Tests for Whiten classification recycle, including loads and product flows."""

import pytest

from pyomo.environ import value

from idaes.core.initialization import InitializationStatus
from idaes.core.util.exceptions import ConfigurationError

from prommis.comminution.core.size_mesh import geometric_series
from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models.tests.crusher_test_support import (
    CLASSIFICATION_RECYCLE_REF,
    EDGES_X,
    FEED_X,
    HC_EDGES,
    solved_flow_tolerance,
    crusher,
    fix_feed,
    flowsheet,
    one_mineral_flowsheet,
    x_individual,
)

# Breakage matrix for the three-interval Whiten recycle cases.
HC_B_RECYCLE = [[1.0, 0.0, 0.4], [0.0, 1.0, 0.6], [0.0, 0.0, 0.0]]


@pytest.mark.unit
def test_recycle_config_interaction_negatives():
    for sel, params in (
        ("king", {"CSS": 0.025, "alpha1": 0.5, "alpha2": 1.2, "n": 1.0}),
        ("austin", {"S1": 0.5, "d1": 1e-3, "a": 0.5}),
        ("vogel_peukert", {"f_mat": 1.0, "xw_min": 0.1, "v": 10.0}),
        ("user", [0.0, 0.0, 1.0]),
    ):
        m = one_mineral_flowsheet(edges=HC_EDGES)
        with pytest.raises(
            ConfigurationError, match="requires selection_function='whiten'"
        ):
            crusher(
                m,
                psd_method="selection_breakage",
                recycle="classification_recycle",
                selection_function=sel,
                selection_params=params,
                breakage_function="user",
                breakage_params=[list(r) for r in HC_B_RECYCLE],
            )
    # Explicit recycle settings are valid only for selection_breakage.
    for method, extra in (
        ("tabular", {"tabular_psd": [0.25, 0.25, 0.25, 0.25]}),
        ("distribution_function", {"shape_exponent": 1.5, "target_product_p80": 5e-3}),
    ):
        for rec in ("single_pass", "classification_recycle"):
            m = one_mineral_flowsheet()
            with pytest.raises(
                ConfigurationError,
                match="'recycle' is consumed only by psd_method='selection_breakage'",
            ):
                crusher(m, psd_method=method, recycle=rec, **extra)
    # n_stress_events applies to single_pass selection_breakage only
    m = one_mineral_flowsheet(edges=HC_EDGES)
    with pytest.raises(
        ConfigurationError,
        match=(
            "n_stress_events can differ from 1 only with "
            "psd_method='selection_breakage' and recycle='single_pass'"
        ),
    ):
        crusher(
            m,
            psd_method="selection_breakage",
            recycle="classification_recycle",
            selection_function="whiten",
            selection_params={"K1": 2e-3, "K2": 8e-3, "K3": 2.3},
            breakage_function="user",
            breakage_params=[list(r) for r in HC_B_RECYCLE],
            n_stress_events=3,
        )
    m = one_mineral_flowsheet(edges=HC_EDGES)
    with pytest.raises(
        ConfigurationError,
        match=(
            "n_stress_events can differ from 1 only with "
            "psd_method='selection_breakage' and recycle='single_pass'"
        ),
    ):
        crusher(
            m,
            psd_method="tabular",
            tabular_psd=[1 / 3, 1 / 3, 1 / 3],
            n_stress_events=2,
        )


def coarsest_recycled_unit(breakage):
    """Build a three-interval Whiten crusher with full coarsest-bin recycle."""
    m = one_mineral_flowsheet(edges=[1e-3, 2e-3, 4e-3, 8e-3])
    cr = crusher(
        m,
        psd_method="selection_breakage",
        recycle="classification_recycle",
        selection_function="whiten",
        selection_params={"K1": 3e-3, "K2": 4e-3, "K3": 2.3},
        breakage_function="user",
        breakage_params=breakage,
    )
    return m, cr


@pytest.mark.unit
def test_recycle_rejects_singular_denominator():
    # The coarsest interval is fully recycled. If it retains all its mass,
    # its recycle denominator 1 - b_22 C_2 is zero.
    with pytest.raises(
        ConfigurationError,
        match=r"Whiten recycle for 'OreA': 1 - b_jj C_j = \S+ at interval 2 "
        r"must be finite and at least",
    ):
        coarsest_recycled_unit([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    m, cr = coarsest_recycled_unit([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    assert [value(cr.classification[k]) for k in range(3)] == [0.0, 0.0, 1.0]


@pytest.mark.component
@pytest.mark.solver
def test_classification_recycle_per_mineral_loads_solved():
    m = flowsheet(edges=EDGES_X)
    cr = crusher(m, bond_work_index=14.0, **x_individual("classification_recycle"))
    fix_feed(cr.properties_in[0], FEED_X)
    native.CrusherSolidPSDInitializer().initialize(cr)
    band = solved_flow_tolerance(8.0)
    for s, loads, products in (
        ("OreA", [40 / 13, 64 / 13, 128 / 13], [40 / 13, 32 / 13, 32 / 13]),
        ("OreB", [72 / 35, 96 / 35, 128 / 7], [72 / 35, 48 / 35, 32 / 7]),
    ):
        for k in range(3):
            assert value(cr.recycle_load[0, s, k]) == pytest.approx(loads[k], abs=band)
            assert value(
                cr.properties_out[0].flow_mass_sized_comp_size[s, k]
            ) == pytest.approx(products[k], abs=band)


@pytest.mark.component
@pytest.mark.solver
def test_zero_mineral_row_classification_recycle_matches_one_species_control():
    # OreB has a different density but no feed.
    subject_m = flowsheet()
    subject = crusher(subject_m, bond_work_index=14.0, **CLASSIFICATION_RECYCLE_REF)
    rows = {"OreA": [0.5, 1.0, 2.0, 6.5], "OreB": [0.0, 0.0, 0.0, 0.0]}
    fix_feed(subject.properties_in[0], rows, liquid=1.0)
    control_m = flowsheet(species=["OreA"], density={"OreA": 2800.0})
    control = crusher(control_m, bond_work_index=14.0, **CLASSIFICATION_RECYCLE_REF)
    fix_feed(control.properties_in[0], {"OreA": rows["OreA"]}, liquid=1.0)
    for unit in (subject, control):
        init = native.CrusherSolidPSDInitializer()
        init.initialize(unit)
        assert init.summary[unit]["status"] == InitializationStatus.Ok
    for k in range(4):
        assert (
            abs(value(subject.properties_out[0].flow_mass_sized_comp_size["OreB", k]))
            <= 1e-12
        )
        assert abs(value(subject.recycle_load[0, "OreB", k])) <= 1e-12
    assert subject.validate_solved_flows() is None
    assert value(subject.product_p80[0]) > 0
    assert value(subject.product_p80[0]) == pytest.approx(
        value(control.product_p80[0]), rel=1e-8
    )
    assert value(subject.power[0]) > 0
    assert value(subject.power[0]) == pytest.approx(value(control.power[0]), rel=1e-8)


@pytest.mark.component
def test_recycle_closure_policy_verdicts(caplog):
    # The coarsest interval retains 31/32 of its recycled mass, so a
    # column-2 sum error grows by a factor of 32 against the 1e-9 limit.
    caplog.set_level("WARNING")
    m, cr = coarsest_recycled_unit(
        [[1.0, 0.0, 0.03125], [0.0, 1.0, 0.0], [0.0, 0.0, 0.96875]]
    )
    assert cr.validate_crushing_coefficients() is None
    assert "exceeds the allowance" not in caplog.text

    # A deficit of 1e-11 gives a correction of 3.2e-10: accepted with a warning.
    m, cr = coarsest_recycled_unit(
        [[1.0, 0.0, 0.03125 - 1e-11], [0.0, 1.0, 0.0], [0.0, 0.0, 0.96875]]
    )
    assert cr.validate_crushing_coefficients() is None
    assert caplog.text.count("exceeds the allowance") == 1
    # A deficit of 1e-10 gives 3.2e-9 and is rejected.
    m, cr = coarsest_recycled_unit(
        [[1.0, 0.0, 0.03125 - 1e-10], [0.0, 1.0, 0.0], [0.0, 0.0, 0.96875]]
    )
    with pytest.raises(ConfigurationError, match="accumulated closure correction"):
        cr.validate_crushing_coefficients()
    # An extra 1e-11 sent to interval 1 leaves the total correction
    # inside the limit but makes the finest product coefficient negative.
    # The basis-product check rejects it.
    m, cr = coarsest_recycled_unit(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.03125 + 1e-11], [0.0, 0.0, 0.96875]]
    )
    with pytest.raises(ConfigurationError, match="basis product of column 2"):
        cr.validate_crushing_coefficients()

    # Every denominator passes, but the long recycle chain makes the
    # rounding allowance alone exceed the correction limit.
    n = 90
    edges = list(geometric_series(51.2e-3, 0.05e-3, 2 ** (1 / 9)))
    m = one_mineral_flowsheet(edges=edges)
    size_char = m.fs.pp.size_char_m
    k1, k2 = size_char[0] * 1.01, size_char[1] * 0.99
    b = [[0.0] * n for _ in range(n)]
    b[0][0] = 1.0
    for k in range(1, n):
        b[k - 1][k] = 0.0005
        b[k][k] = 0.9995
    cr = crusher(
        m,
        psd_method="selection_breakage",
        recycle="classification_recycle",
        selection_function="whiten",
        selection_params={"K1": k1, "K2": k2, "K3": 1.0},
        breakage_function="user",
        breakage_params=[list(r) for r in b],
    )
    assert [value(cr.classification[k]) for k in range(n)] == [0.0] + [1.0] * (n - 1)
    with pytest.raises(ConfigurationError, match="conditioning") as exc:
        cr.validate_crushing_coefficients()
    assert "alone exceeds the correction limit" in str(exc.value)
