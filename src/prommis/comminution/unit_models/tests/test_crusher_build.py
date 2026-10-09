#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Build, export and configuration-guard tests for CrusherSolidPSD."""

import io
import math
import re

import numpy
import pytest

from pyomo.environ import units, value
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import degrees_of_freedom

import prommis.comminution as pkg
import prommis.comminution.unit_models as um
from prommis.comminution.functions.distributions import RR_VALIDITY_THRESHOLD
from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models.tests.crusher_test_support import (
    CLASSIFICATION_RECYCLE_REF,
    DENSITY_TWO,
    EDGES_X,
    LUCKIE_AUSTIN_REF,
    PENDULUM_REF,
    SINGLE_STRESS_REF,
    WHITEN_B4,
    WHITEN_REF,
    crusher,
    fix_feed,
    flowsheet,
    one_mineral_flowsheet,
)
from prommis.solid_handling.crusher_solids_properties import CoalRefuseParameters


@pytest.mark.unit
def test_default_initializer_and_scaler_bindings():
    m = one_mineral_flowsheet()
    cr = crusher(m, **SINGLE_STRESS_REF)
    assert cr.default_initializer is native.CrusherSolidPSDInitializer
    assert cr.default_scaler is native.CrusherSolidPSDScaler
    assert "CrusherSolidPSD" in um.__all__ and "CrusherSolidPSD" in pkg.__all__
    assert um.CrusherSolidPSD is native.CrusherSolidPSD
    assert pkg.CrusherSolidPSD is native.CrusherSolidPSD


@pytest.mark.unit
def test_work_index_domain_at_build():
    lo, hi = native.WORK_INDEX_BOUNDS
    for bad in (lo - 0.001, hi + 0.001, 0.0, -1.0, float("inf"), float("nan")):
        m = one_mineral_flowsheet()
        with pytest.raises(ConfigurationError, match="bond_work_index"):
            crusher(m, bond_work_index=bad, **SINGLE_STRESS_REF)
    # the tabular route applies the same domain
    m = one_mineral_flowsheet()
    with pytest.raises(ConfigurationError, match="bond_work_index"):
        crusher(
            m,
            psd_method="tabular",
            tabular_psd=[0.25, 0.25, 0.25, 0.25],
            bond_work_index=0.0,
        )
    m = one_mineral_flowsheet()
    with pytest.raises(ConfigurationError, match="bond_work_index.*boolean"):
        crusher(m, bond_work_index=True, **SINGLE_STRESS_REF)
    for good in (lo, hi, 14.0, 105.93):
        m = one_mineral_flowsheet()
        cr = crusher(m, bond_work_index=good, **SINGLE_STRESS_REF)
        assert cr.bond_work_index.fixed and cr.bond_work_index.value == good
        assert cr.bond_work_index.lb == lo and cr.bond_work_index.ub == hi
    # the config gate and the Var domain read the same constant
    assert (lo, hi) == (1.0, 150.0)


@pytest.mark.unit
def test_n_stress_events_domain(caplog, monkeypatch):
    accepted = [1, 2, 1000]
    rejected = [
        1001,
        2**53 + 1,
        10**10000,
        True,
        numpy.bool_(True),
        "3",
        b"3",
        1e300,
        3.0,
        0,
        -1,
        {"OreA": 2},
    ]
    for n in accepted:
        m = one_mineral_flowsheet()
        crusher(m, **dict(SINGLE_STRESS_REF, n_stress_events=n))
    for n in rejected:
        m = one_mineral_flowsheet()
        with pytest.raises(ConfigurationError, match="n_stress_events"):
            crusher(m, **dict(SINGLE_STRESS_REF, n_stress_events=n))
    for n in (True, numpy.bool_(True)):
        m = one_mineral_flowsheet()
        with pytest.raises(ConfigurationError, match="not a boolean"):
            crusher(m, **dict(SINGLE_STRESS_REF, n_stress_events=n))

    # any integer-like object is accepted through __index__ and stored as int
    class IndexOnly:
        def __index__(self):
            return 2

    for n in (IndexOnly(), numpy.int64(2)):
        m = one_mineral_flowsheet()
        cr = crusher(m, **dict(SINGLE_STRESS_REF, n_stress_events=n))
        assert cr.n_stress_events == 2
        assert set(cr.psd_stage) == {(0, 1, "OreA", k) for k in range(4)}
    # more than one event is allowed only on selection_breakage + single_pass
    for cfg, match in (
        (
            dict(psd_method="tabular", tabular_psd=[0.25] * 4, n_stress_events=2),
            "n_stress_events",
        ),
        (
            dict(
                psd_method="distribution_function",
                shape_exponent=1.5,
                n_stress_events=2,
            ),
            "n_stress_events can differ",
        ),
        (dict(CLASSIFICATION_RECYCLE_REF, n_stress_events=3), "n_stress_events"),
    ):
        m = one_mineral_flowsheet()
        with pytest.raises(ConfigurationError, match=match):
            crusher(m, **cfg)
    # Three events create eight stage variable entries: (3 - 1) * 4 * 1 * 1.
    monkeypatch.setattr(native, "STAGE_VARIABLE_WARNING_THRESHOLD", 7)
    caplog.set_level("WARNING", logger=native._log.name)
    caplog.clear()
    crusher(one_mineral_flowsheet(), **dict(SINGLE_STRESS_REF, n_stress_events=3))
    records = [r for r in caplog.records if r.name == native._log.name]
    assert len(records) == 1
    text = records[0].message
    assert "declares 8 stage variable entries" in text
    assert "warning threshold of 7" in text
    # The warning gives a count, not a machine-dependent resource estimate.
    assert "bytes" not in text and " s " not in text
    assert "time and memory" in text
    caplog.clear()
    monkeypatch.setattr(native, "STAGE_VARIABLE_WARNING_THRESHOLD", 8)
    crusher(one_mineral_flowsheet(), **dict(SINGLE_STRESS_REF, n_stress_events=3))
    assert not [r for r in caplog.records if r.name == native._log.name]


@pytest.mark.unit
def test_individual_input_form_guards():
    m = flowsheet(edges=EDGES_X)
    with pytest.raises(ConfigurationError, match="tabular_psd.*tabular_psd_by_comp"):
        crusher(
            m,
            psd_method="tabular",
            tabular_psd=[0.5, 0.25, 0.25],
            tabular_psd_by_comp={
                "OreA": [0.5, 0.25, 0.25],
                "OreB": [0.5, 0.25, 0.25],
            },
        )
    S_A = [0.0, 0.5, 1.0]
    B_A = [[1.0, 0.5, 0.25], [0.0, 0.5, 0.5], [0.0, 0.0, 0.25]]
    user_single_stress = dict(
        psd_method="selection_breakage",
        recycle="single_pass",
        selection_function="user",
        breakage_function="user",
    )
    m = flowsheet(edges=EDGES_X)
    with pytest.raises(
        ConfigurationError, match="selection_params.*selection_params_by_comp"
    ):
        crusher(
            m,
            selection_params=S_A,
            selection_params_by_comp={"OreA": S_A, "OreB": S_A},
            breakage_params=B_A,
            **user_single_stress,
        )
    m = flowsheet(edges=EDGES_X)
    with pytest.raises(
        ConfigurationError, match="breakage_params.*breakage_params_by_comp"
    ):
        crusher(
            m,
            selection_params=S_A,
            breakage_params=B_A,
            breakage_params_by_comp={"OreA": B_A, "OreB": B_A},
            **user_single_stress,
        )
    m = flowsheet(edges=EDGES_X)
    with pytest.raises(ConfigurationError, match="tabular_psd_by_comp"):
        crusher(
            m,
            selection_params=S_A,
            breakage_params=B_A,
            tabular_psd_by_comp={
                "OreA": [0.5, 0.25, 0.25],
                "OreB": [0.5, 0.25, 0.25],
            },
            **user_single_stress,
        )
    m = flowsheet(edges=EDGES_X)
    with pytest.raises(ConfigurationError, match="selection_params_by_comp"):
        crusher(
            m,
            psd_method="tabular",
            tabular_psd=[0.5, 0.25, 0.25],
            selection_params_by_comp={"OreA": S_A, "OreB": S_A},
        )
    # classification_recycle: the classification vector is common
    m = flowsheet(edges=EDGES_X)
    with pytest.raises(ConfigurationError, match="classification vector is common"):
        crusher(
            m,
            psd_method="selection_breakage",
            recycle="classification_recycle",
            selection_function="whiten",
            selection_params_by_comp={"OreA": WHITEN_REF, "OreB": WHITEN_REF},
            breakage_function="user",
            breakage_params=B_A,
        )
    tab = [0.5, 0.25, 0.25]
    for key, common in (
        ("tabular_psd_by_comp", None),
        ("selection_params_by_comp", "selection_params"),
        ("breakage_params_by_comp", "breakage_params"),
    ):
        cfg = (
            dict(psd_method="tabular")
            if common is None
            else dict(user_single_stress, selection_params=S_A, breakage_params=B_A)
        )
        if common is not None:
            cfg.pop(common)
        with pytest.raises(ConfigurationError, match=key + ".*dict"):
            crusher(flowsheet(edges=EDGES_X), **dict(cfg, **{key: []}))
    cases = {
        "OreB": ({"OreA": tab}, "missing.*OreB"),
        "OreC": ({"OreA": tab, "OreB": tab, "OreC": tab}, "unknown.*OreC"),
        "H2O": ({"OreA": tab, "OreB": tab, "H2O": tab}, "H2O.*liquid"),
        "nonstring": ({"OreA": tab, "OreB": tab, 3: tab}, "unknown"),
    }
    for _, (mapping, pattern) in cases.items():
        m = flowsheet(edges=EDGES_X)
        with pytest.raises(ConfigurationError, match=pattern):
            crusher(m, psd_method="tabular", tabular_psd_by_comp=mapping)
    m = flowsheet(
        edges=EDGES_X, unsized=["Inert"], density=dict(DENSITY_TWO, Inert=2000.0)
    )
    with pytest.raises(ConfigurationError, match="Inert.*unsized"):
        crusher(
            m,
            psd_method="tabular",
            tabular_psd_by_comp={"OreA": tab, "OreB": tab, "Inert": tab},
        )
    m = flowsheet(edges=EDGES_X, vapors=["Air"])
    with pytest.raises(ConfigurationError, match="Air.*vapor"):
        crusher(
            m,
            psd_method="tabular",
            tabular_psd_by_comp={"OreA": tab, "OreB": tab, "Air": tab},
        )
    # per-component row rules name the component
    for bad_row in (
        [0.5, 0.5],
        [0.5, -0.25, 0.75],
        [0.5, float("nan"), 0.25],
        [50.0, 25.0, 25.0],
    ):
        m = flowsheet(edges=EDGES_X)
        with pytest.raises(ConfigurationError, match="OreB"):
            crusher(
                m,
                psd_method="tabular",
                tabular_psd_by_comp={"OreA": tab, "OreB": bad_row},
            )
    m1 = flowsheet(edges=EDGES_X)
    original_order_crusher = crusher(
        m1,
        psd_method="tabular",
        tabular_psd_by_comp={"OreA": tab, "OreB": [0.25, 0.5, 0.25]},
    )
    m2 = flowsheet(edges=EDGES_X)
    reordered_crusher = crusher(
        m2,
        psd_method="tabular",
        tabular_psd_by_comp={"OreB": [0.25, 0.5, 0.25], "OreA": tab},
    )
    for idx in original_order_crusher.tabular_psd:
        assert (
            original_order_crusher.tabular_psd[idx].value
            == reordered_crusher.tabular_psd[idx].value
        )
    # Machine-setting parameters such as CSS and v must match across components.
    king_a = {"CSS": 2e-3, "alpha1": 0.5, "alpha2": 1.2, "n": 1.0}
    m = flowsheet(edges=EDGES_X)
    with pytest.raises(ConfigurationError, match="CSS.*OreA.*OreB|CSS.*OreB.*OreA"):
        crusher(
            m,
            psd_method="selection_breakage",
            recycle="single_pass",
            selection_function="king",
            selection_params_by_comp={
                "OreA": king_a,
                "OreB": dict(king_a, CSS=3e-3),
            },
            breakage_function="luckie_austin",
            breakage_params=LUCKIE_AUSTIN_REF,
        )
    m = flowsheet(edges=EDGES_X)
    crusher(
        m,
        psd_method="selection_breakage",
        recycle="single_pass",
        selection_function="king",
        selection_params_by_comp={"OreA": king_a, "OreB": dict(king_a, alpha1=0.7)},
        breakage_function="luckie_austin",
        breakage_params=LUCKIE_AUSTIN_REF,
    )
    vp_a = {"f_mat": 1.0, "xw_min": 0.0, "v": 10.0}
    m = flowsheet(edges=EDGES_X)
    with pytest.raises(ConfigurationError, match="'v'"):
        crusher(
            m,
            psd_method="selection_breakage",
            recycle="single_pass",
            n_stress_events=1,
            selection_function="vogel_peukert",
            selection_params_by_comp={"OreA": vp_a, "OreB": dict(vp_a, v=20.0)},
            breakage_function="luckie_austin",
            breakage_params=LUCKIE_AUSTIN_REF,
        )
    m = flowsheet(edges=EDGES_X)
    crusher(
        m,
        psd_method="selection_breakage",
        recycle="single_pass",
        n_stress_events=1,
        selection_function="vogel_peukert",
        selection_params_by_comp={"OreA": vp_a, "OreB": dict(vp_a, f_mat=2.0)},
        breakage_function="luckie_austin",
        breakage_params=LUCKIE_AUSTIN_REF,
    )


# Invalid build configurations. ``edges`` and ``two_species`` configure the
# flowsheet; all other options configure the crusher.
_TABULAR_FOUR = dict(psd_method="tabular", tabular_psd=[0.25, 0.5, 0.25, 0.0])
_DISTRIBUTION = dict(psd_method="distribution_function", shape_exponent=1.5)
_BUILD_REJECTIONS = [
    (
        "distribution_without_shape_exponent",
        dict(psd_method="distribution_function"),
        "shape_exponent",
    ),
    (
        "shape_exponent_zero",
        dict(_DISTRIBUTION, shape_exponent=0.0, target_product_p80=5e-3),
        "shape_exponent.*positive",
    ),
    (
        "shape_exponent_negative",
        dict(_DISTRIBUTION, shape_exponent=-1.5, target_product_p80=5e-3),
        "shape_exponent.*positive",
    ),
    (
        "negative_target_product_p80",
        dict(_DISTRIBUTION, target_product_p80=-1e-3, edges=[1e-3, 2e-3, 4e-3]),
        "must be a positive size",
    ),
    ("tabular_without_table", dict(psd_method="tabular"), "tabular_psd"),
    (
        "tabular_wrong_length",
        dict(psd_method="tabular", tabular_psd=[0.5, 0.5]),
        "length",
    ),
    (
        "tabular_sum_outside_band",
        dict(psd_method="tabular", tabular_psd=[0.3] * 4),
        "deviates",
    ),
    (
        "tabular_non_sequence",
        dict(psd_method="tabular", tabular_psd=1),
        "tabular_psd must be a sequence",
    ),
    # an unordered set is not a sequence, so its rows cannot be reordered silently
    (
        "tabular_set",
        dict(psd_method="tabular", tabular_psd={0.4, 0.3, 0.2, 0.1}),
        "must be a sequence",
    ),
    # a 0-d array has the sequence attributes but cannot be iterated
    (
        "tabular_zero_d_array",
        dict(psd_method="tabular", tabular_psd=numpy.array(1.0)),
        "must be a sequence",
    ),
    # Converting this mapping to a list would mistake its keys for fractions.
    (
        "tabular_species_row_as_mapping",
        dict(
            psd_method="tabular",
            tabular_psd_by_comp={"OreA": {0: 0.0, 1: 1.0}, "OreB": [0.0, 1.0]},
            edges=[1e-3, 4e-3, 16e-3],
            two_species=True,
        ),
        "not a mapping",
    ),
    (
        "tabular_single_pass_recycle",
        dict(psd_method="tabular", tabular_psd=[0.25] * 4, recycle="single_pass"),
        "recycle",
    ),
    (
        "tabular_classification_recycle",
        dict(
            psd_method="tabular",
            tabular_psd=[0.25] * 4,
            recycle="classification_recycle",
        ),
        "recycle",
    ),
    (
        "selection_breakage_with_tabular_psd",
        dict(SINGLE_STRESS_REF, tabular_psd=[0.25, 0.5, 0.25, 0.0], two_species=True),
        "^"
        + re.escape(
            "psd_method='selection_breakage' does not use these options: "
            "['tabular_psd']. Remove them from the configuration."
        )
        + "$",
    ),
    (
        "distribution_with_selection_params",
        dict(_DISTRIBUTION, selection_params=WHITEN_REF, two_species=True),
        "^"
        + re.escape(
            "psd_method='distribution_function' does not use these options: "
            "['selection_params']. Remove them from the configuration."
        )
        + "$",
    ),
    (
        "tabular_with_shape_exponent",
        dict(_TABULAR_FOUR, shape_exponent=1.5, two_species=True),
        "^"
        + re.escape(
            "psd_method='tabular' does not use these options: ['shape_exponent']. "
            "Remove them from the configuration."
        )
        + "$",
    ),
    # several stray options are reported together, sorted
    (
        "tabular_with_three_foreign_options",
        dict(
            _TABULAR_FOUR,
            target_product_p80=5e-3,
            selection_params=WHITEN_REF,
            breakage_function="vogel",
            two_species=True,
        ),
        "^"
        + re.escape(
            "psd_method='tabular' does not use these options: "
            "['breakage_function', 'selection_params', 'target_product_p80']. "
            "Remove them from the configuration."
        )
        + "$",
    ),
    # recycle and n_stress_events keep their own rules, checked first
    (
        "tabular_recycle_before_foreign_options",
        dict(
            _TABULAR_FOUR,
            recycle="single_pass",
            target_product_p80=5e-3,
            two_species=True,
        ),
        "'recycle' is consumed only",
    ),
    (
        "distribution_n_stress_events_before_foreign_options",
        dict(
            _DISTRIBUTION,
            n_stress_events=2,
            tabular_psd=[0.25, 0.5, 0.25, 0.0],
            two_species=True,
        ),
        "n_stress_events can differ",
    ),
    (
        "size_spec_on_one_interval",
        dict(_DISTRIBUTION, target_product_p80=1.5e-3, edges=[1e-3, 2e-3]),
        "at least two size intervals",
    ),
    (
        "power_spec_on_one_interval",
        dict(_DISTRIBUTION, edges=[1e-3, 2e-3]),
        "at least two size intervals",
    ),
    # the default configuration is selection_breakage + classification_recycle
    # + whiten, which needs K1 and K2
    ("default_configuration", dict(), "K1"),
]

# Successful cases: case ID, options, and an optional check on the built unit.
_BUILD_CONTROLS = [
    (
        "size_spec_on_two_intervals",
        dict(_DISTRIBUTION, target_product_p80=2e-3, edges=[1e-3, 2e-3, 4e-3]),
        None,
    ),
    (
        "power_spec_on_two_intervals",
        dict(_DISTRIBUTION, edges=[1e-3, 2e-3, 4e-3]),
        None,
    ),
    (
        "tabular_on_one_interval",
        dict(psd_method="tabular", tabular_psd=[1.0], edges=[1e-3, 2e-3]),
        None,
    ),
    (
        "tabular_recycle_unset",
        dict(psd_method="tabular", tabular_psd=[0.25] * 4),
        None,
    ),
    (
        "distribution_recycle_unset",
        dict(_DISTRIBUTION, target_product_p80=5e-3),
        None,
    ),
    (
        "default_path_with_whiten_parameters",
        dict(
            selection_params={"K1": 2e-3, "K2": 8e-3, "K3": 2.3},
            breakage_function="user",
            breakage_params=WHITEN_B4,
        ),
        None,
    ),
    # An unset option is not stray; equipment supplies function presets only for
    # selection-breakage.
    (
        "tabular_with_equipment",
        dict(_TABULAR_FOUR, crusher_equipment="jaw", two_species=True),
        lambda cr: cr.config.crusher_stage == "primary",
    ),
    (
        "selection_breakage_with_unset_target",
        dict(SINGLE_STRESS_REF, target_product_p80=None, two_species=True),
        lambda cr: cr.config.n_stress_events == 1,
    ),
    # a table inside the 1e-5 band is renormalized to sum 1
    (
        "tabular_renormalized_inside_band",
        dict(psd_method="tabular", tabular_psd=[0.25, 0.25, 0.25, 0.249999]),
        lambda cr: math.fsum(value(cr.tabular_psd["OreA", k]) for k in range(4))
        == pytest.approx(1.0, rel=1e-12),
    ),
    (
        "tabular_renormalization_keeps_zero",
        dict(psd_method="tabular", tabular_psd=[0.0, 0.3, 0.3, 0.400002]),
        lambda cr: value(cr.tabular_psd["OreA", 0]) == 0.0,
    ),
]


def _crusher_from_case(kwargs):
    """Build a crusher from a case on a fresh flowsheet; return the model and unit."""
    cfg = dict(kwargs)
    edges = cfg.pop("edges", None)
    m = (
        flowsheet(edges=edges)
        if cfg.pop("two_species", False)
        else one_mineral_flowsheet(edges=edges)
    )
    return m, crusher(m, **cfg)


@pytest.mark.unit
def test_build_rejects_configurations_outside_the_spec():
    failures = []
    for case_id, kwargs, match in _BUILD_REJECTIONS:
        try:
            with pytest.raises(ConfigurationError, match=match):
                _crusher_from_case(kwargs)
        except (Exception, pytest.fail.Exception) as exc:
            failures.append(f"{case_id}: {exc}")
    for case_id, kwargs, check in _BUILD_CONTROLS:
        try:
            m, cr = _crusher_from_case(kwargs)
            if check is not None:
                assert check(cr)
            pp = cr.config.property_package
            rows = {
                s: [0.0] * (len(pp.size_interval_set) - 1) + [10.0]
                for s in pp.sized_solid_list
            }
            fix_feed(cr.properties_in[0], rows)
            expected_dof = int(
                cr.config.psd_method == "distribution_function"
                and cr.config.target_product_p80 is None
            )
            assert degrees_of_freedom(m) == expected_dof
            if expected_dof:
                cr.power[0].fix(3.0)
                assert degrees_of_freedom(m) == 0
            assert_units_consistent(m)
        except (Exception, pytest.fail.Exception) as exc:
            failures.append(f"{case_id}: {type(exc).__name__}: {exc}")
    assert not failures, "\n".join(failures)


@pytest.mark.unit
def test_common_material_input_diagnostics():
    tab = [0.5, 0.25, 0.25]
    S_A = [0.0, 0.5, 1.0]
    B_A = [[1.0, 0.5, 0.25], [0.0, 0.5, 0.5], [0.0, 0.0, 0.25]]
    user_single_stress = dict(
        psd_method="selection_breakage",
        recycle="single_pass",
        selection_function="user",
        breakage_function="user",
    )
    # A component-indexed mapping in the common slot points users to the
    # per-component option.
    with pytest.raises(ConfigurationError, match="tabular_psd_by_comp"):
        crusher(
            flowsheet(edges=EDGES_X),
            psd_method="tabular",
            tabular_psd={"OreA": tab, "OreB": tab},
        )
    # Converting this mapping to a list would mistake its keys for fractions.
    with pytest.raises(ConfigurationError, match="tabular_psd_by_comp"):
        crusher(
            flowsheet(edges=[1e-3, 4e-3, 16e-3]),
            psd_method="tabular",
            tabular_psd={0: 0.25, 1: 0.75},
        )
    with pytest.raises(ConfigurationError, match="selection_params_by_comp"):
        crusher(
            flowsheet(edges=EDGES_X),
            selection_params={"OreA": S_A, "OreB": S_A},
            breakage_params=B_A,
            **user_single_stress,
        )
    with pytest.raises(ConfigurationError, match="breakage_params_by_comp"):
        crusher(
            flowsheet(edges=EDGES_X),
            selection_params=S_A,
            breakage_params={"OreA": B_A, "OreB": B_A},
            **user_single_stress,
        )
    # common parametric dictionaries keep the schema error and add the hint
    with pytest.raises(ConfigurationError) as exc:
        crusher(
            flowsheet(edges=EDGES_X),
            **dict(
                SINGLE_STRESS_REF,
                selection_params={"OreA": WHITEN_REF, "OreB": WHITEN_REF},
            ),
        )
    text = str(exc.value)
    assert "requires selection_params with keys" in text
    assert "selection_params_by_comp" in text
    with pytest.raises(ConfigurationError) as exc:
        crusher(
            flowsheet(edges=EDGES_X),
            **dict(
                SINGLE_STRESS_REF,
                breakage_params={"OreA": LUCKIE_AUSTIN_REF, "OreB": LUCKIE_AUSTIN_REF},
            ),
        )
    text = str(exc.value)
    assert "requires breakage_params with keys" in text
    assert "breakage_params_by_comp" in text
    # Component names matching parameter keys must not make valid common data
    # look like per-component input.
    names = ["K1", "K2", "t10", "appearance_table"]
    density = {name: 2800.0 for name in names}
    crusher(
        flowsheet(edges=EDGES_X, species=names, density=density), **SINGLE_STRESS_REF
    )
    table = {
        "10": {"t75": 1.0, "t50": 1.5, "t25": 3.0, "t4": 20.0, "t2": 40.0},
        "20": {"t75": 2.0, "t50": 3.0, "t25": 6.0, "t4": 35.0, "t2": 60.0},
    }
    crusher(
        flowsheet(edges=EDGES_X, species=names, density=density),
        **dict(
            CLASSIFICATION_RECYCLE_REF,
            breakage_function="t10_appearance",
            breakage_params={"appearance_table": table, "t10": 15.0},
        ),
    )
    # Recycle still reports the parameter error, but cannot suggest the
    # forbidden per-component Whiten option.
    with pytest.raises(ConfigurationError) as exc:
        crusher(
            flowsheet(edges=EDGES_X),
            **dict(
                CLASSIFICATION_RECYCLE_REF,
                selection_params={"OreA": WHITEN_REF, "OreB": WHITEN_REF},
            ),
        )
    text = str(exc.value)
    assert "requires selection_params with keys" in text
    assert "common Whiten" in text
    assert "selection_params_by_comp" not in text
    with pytest.raises(ConfigurationError) as exc:
        crusher(flowsheet(edges=EDGES_X), **dict(SINGLE_STRESS_REF, selection_params=1))
    text = str(exc.value)
    assert "as a mapping" in text and "selection_params_by_comp" in text
    with pytest.raises(ConfigurationError) as exc:
        crusher(flowsheet(edges=EDGES_X), **dict(SINGLE_STRESS_REF, breakage_params=1))
    text = str(exc.value)
    assert "as a mapping" in text and "breakage_params_by_comp" in text
    with pytest.raises(ConfigurationError, match=r"selection_params_by_comp\['OreA'\]"):
        crusher(
            flowsheet(edges=EDGES_X),
            **dict(
                SINGLE_STRESS_REF,
                selection_params=None,
                selection_params_by_comp={"OreA": 1, "OreB": WHITEN_REF},
            ),
        )
    # a domain error on common data carries the hint too (components named K1, K2)
    with pytest.raises(ConfigurationError, match="selection_params_by_comp"):
        crusher(
            flowsheet(
                edges=EDGES_X,
                species=["K1", "K2"],
                density={"K1": 2800.0, "K2": 2800.0},
            ),
            **dict(
                SINGLE_STRESS_REF, selection_params={"K1": WHITEN_REF, "K2": WHITEN_REF}
            ),
        )
    with pytest.raises(ConfigurationError, match="t10 must be finite"):
        crusher(
            flowsheet(edges=EDGES_X),
            **dict(
                CLASSIFICATION_RECYCLE_REF,
                breakage_function="t10_appearance",
                breakage_params={"t10": float("nan"), "appearance_table": table},
            ),
        )
    # stray keys of mixed type (int and str) are reported, not a sort crash
    with pytest.raises(ConfigurationError) as exc:
        crusher(
            flowsheet(edges=EDGES_X),
            **dict(
                SINGLE_STRESS_REF,
                selection_params={**WHITEN_REF, 1: 0.0, "typo": 0.0},
            ),
        )
    text = str(exc.value)
    assert "unknown" in text and "'typo'" in text
    assert "selection_params_by_comp" in text
    # a table-domain error on common data carries the hint too
    with pytest.raises(ConfigurationError) as exc:
        crusher(
            flowsheet(edges=EDGES_X),
            **dict(
                CLASSIFICATION_RECYCLE_REF,
                breakage_function="t10_appearance",
                breakage_params={"t10": 10.0, "appearance_table": {"10": table["10"]}},
            ),
        )
    text = str(exc.value)
    assert "at least two rows" in text
    assert "breakage_params_by_comp" in text


COARSE_ONE_MINERAL = {"OreA": [0.0, 0.0, 0.0, 10.0]}


@pytest.mark.unit
def test_property_package_type_checked():
    # A different property-package type must fail the crusher's type check.
    m = one_mineral_flowsheet()
    m.fs.wrong = CoalRefuseParameters()
    with pytest.raises(ConfigurationError, match="SolidPSD"):
        m.fs.bad = native.CrusherSolidPSD(
            property_package=m.fs.wrong,
            psd_method="tabular",
            tabular_psd=[0.25, 0.25, 0.25, 0.25],
        )
    # control: the flowsheet default package resolves before the type check
    m = flowsheet()
    m.fs.config.default_property_package = m.fs.pp
    m.fs.cr = native.CrusherSolidPSD(
        psd_method="tabular", tabular_psd=[0.25, 0.5, 0.25, 0.0]
    )
    assert m.fs.cr.config.property_package is m.fs.pp


@pytest.mark.unit
def test_distribution_function_d80_out_of_mesh_rejected_at_build():
    for bad_d80 in (20e-3, 1.0e-3):  # above x_N, below the bin-0 floor 1.8e-3
        m = one_mineral_flowsheet()
        with pytest.raises(ConfigurationError, match="attainable mesh band"):
            crusher(
                m,
                psd_method="distribution_function",
                shape_exponent=1.5,
                target_product_p80=bad_d80,
            )
    m = one_mineral_flowsheet()
    with pytest.raises(ConfigurationError, match="d_max must be <= the top mesh edge"):
        crusher(
            m,
            psd_method="distribution_function",
            distribution_shape="gates_gaudin_schuhmann",
            shape_exponent=1.5,
            target_product_p80=15e-3,  # d_max = 17.4e-3 > 16e-3
        )
    # A valid target gives each characteristic size a top-edge validity bound:
    # Rosin-Rammler uses 0.95 passing; GGS uses the top mesh edge.
    assert RR_VALIDITY_THRESHOLD == 0.95
    m = one_mineral_flowsheet()
    cr = crusher(
        m,
        psd_method="distribution_function",
        shape_exponent=1.5,
        target_product_p80=5e-3,
    )
    expected_ub = 16e-3 / ((-math.log(0.05)) ** (1.0 / 1.5))
    assert cr.d_63[0].lb == pytest.approx(1e-9)
    assert math.isclose(cr.d_63[0].ub, expected_ub, rel_tol=1e-12, abs_tol=0.0)
    m2 = one_mineral_flowsheet()
    cr2 = crusher(
        m2,
        psd_method="distribution_function",
        distribution_shape="gates_gaudin_schuhmann",
        shape_exponent=1.5,
        target_product_p80=5e-3,
    )
    assert cr2.d_max[0].lb == pytest.approx(1e-9)
    assert math.isclose(cr2.d_max[0].ub, 16e-3, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.component
def test_power_spec_requires_fixed_power():
    # The fixed-power check runs before the degrees-of-freedom check.
    m = one_mineral_flowsheet()
    cr = crusher(
        m,
        psd_method="distribution_function",
        distribution_shape="rosin_rammler",
        shape_exponent=1.5,
        bond_work_index=12.0,
    )
    # A fixed feed leaves power as the one free operating specification.
    fix_feed(cr.properties_in[0], COARSE_ONE_MINERAL)
    assert degrees_of_freedom(m) == 1
    with pytest.raises(ConfigurationError, match="is not fixed"):
        native.CrusherSolidPSDInitializer().initialize(cr)
    cr.power[0].fix(6.0)
    assert degrees_of_freedom(m) == 0


# Each case gives the configuration and degrees of freedom after fixing the
# feed. A power specification leaves one; the other paths leave zero.
PATH_SHAPE_SCOPES = {
    "tabular": (dict(psd_method="tabular", tabular_psd=[0.25, 0.25, 0.25, 0.25]), 0),
    "rr_size_spec": (
        dict(
            psd_method="distribution_function",
            distribution_shape="rosin_rammler",
            shape_exponent=1.5,
            target_product_p80=5e-3,
        ),
        0,
    ),
    "ggs_power_spec": (
        dict(
            psd_method="distribution_function",
            distribution_shape="gates_gaudin_schuhmann",
            shape_exponent=1.5,
        ),
        1,
    ),
    "single_stress": (dict(SINGLE_STRESS_REF), 0),
    "repeated_stress": (dict(SINGLE_STRESS_REF, n_stress_events=3), 0),
    "classification_recycle": (dict(CLASSIFICATION_RECYCLE_REF), 0),
}


@pytest.mark.build
@pytest.mark.component
@pytest.mark.parametrize("scope", sorted(PATH_SHAPE_SCOPES))
def test_every_path_shape_scope_is_square_and_consistent(scope):
    """Check degrees of freedom, units, and structure for each product path."""
    cfg, expected_dof = PATH_SHAPE_SCOPES[scope]
    m = one_mineral_flowsheet()
    cr = crusher(m, bond_work_index=12.0, **cfg)
    fix_feed(cr.properties_in[0], COARSE_ONE_MINERAL)
    assert cr.power[0].lb == 0
    assert_units_equivalent(cr.power[0], units.kW)
    assert degrees_of_freedom(m) == expected_dof
    if expected_dof:
        cr.power[0].fix(3.0)
        assert degrees_of_freedom(m) == 0
    assert_units_consistent(m)
    DiagnosticsToolbox(m).assert_no_structural_warnings(ignore_evaluation_errors=True)


def _report_line(text, label):
    """Return the reported value and unit for ``label``."""
    match = re.search(
        rf"^\s*{re.escape(label)}\s*:\s*(\S+)\s*:\s*(\S+)", text, re.MULTILINE
    )
    assert match, f"no {label!r} line in the report"
    return float(match.group(1)), match.group(2)


@pytest.mark.unit
def test_report_smoke():
    m = one_mineral_flowsheet()
    cr = crusher(m, psd_method="tabular", tabular_psd=[0.25, 0.25, 0.25, 0.25])
    buf = io.StringIO()
    cr.report(ostream=buf)
    assert "Power" in buf.getvalue()
    assert "Product P80" in buf.getvalue()
    # With distinct inlet and outlet rows, the report must show the product P80.
    fix_feed(cr.properties_in[0], COARSE_ONE_MINERAL)
    for k, flow in enumerate([10.0, 0.0, 0.0, 0.0]):
        cr.properties_out[0].flow_mass_sized_comp_size["OreA", k].set_value(flow)
    buf = io.StringIO()
    cr.report(ostream=buf)
    text = buf.getvalue()
    assert re.search(r"^\s*Units\s+Inlet\s+Outlet\s*$", text, re.MULTILINE)
    product_p80, unit = _report_line(text, "Product P80")
    assert unit == "meter"
    assert product_p80 == pytest.approx(value(cr.product_p80[0]), rel=1e-4)
    assert product_p80 != pytest.approx(
        value(cr.properties_in[0].percentile_size[0.8]), rel=1e-2
    )
    # The pendulum law reports its power separately from unit power.
    m = flowsheet()
    cr = crusher(m, **PENDULUM_REF)
    cr.power[0].set_value(123.0)  # kW
    buf = io.StringIO()
    cr.report(ostream=buf)
    text = buf.getvalue()
    assert "Pendulum power" in text
    power, unit = _report_line(text, "Power")
    assert unit == "watt" and power == pytest.approx(1.23e5, rel=1e-4)
    pendulum, unit = _report_line(text, "Pendulum power")
    assert unit == "watt"
    assert pendulum == pytest.approx(
        value(units.convert(cr.pendulum_power[0], to_units=units.W)), rel=1e-4
    )
