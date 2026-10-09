#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Crusher selection and breakage tests, including per-component product rows."""

import math

import pytest

from pyomo.environ import value

from idaes.core.initialization import InitializationStatus
from idaes.core.util.exceptions import ConfigurationError, InitializationError
from idaes.core.util.model_statistics import degrees_of_freedom

from prommis.comminution.core.size_mesh import geometric_series
from prommis.comminution.functions.breakage import breakage_matrix
from prommis.comminution.functions.selection import selection_value
from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models.tests.crusher_test_support import (
    B_A,
    B_B,
    EDGES_X,
    FEED_X,
    HC_EDGES,
    LUCKIE_AUSTIN_REF,
    PARAMETRIC_SETS,
    S_A,
    S_B,
    ZERO_POWER_SOLVER_OPTIONS,
    ZERO_BOTTOM,
    ZERO_EDGES,
    ZERO_WHITEN,
    solved_flow_tolerance,
    crusher,
    fix_feed,
    flowsheet,
    one_mineral_flowsheet,
    x_individual,
)

# Hand-calculated single-pass selection and breakage data for HC_EDGES.
HC_S = [0.0, 0.5, 1.0]
HC_B = [[1.0, 1.0, 0.4], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]]


@pytest.mark.unit
def test_user_format_negatives():
    def _build(sel=None, brk=None):
        m = one_mineral_flowsheet(edges=HC_EDGES)
        cr = crusher(
            m,
            psd_method="selection_breakage",
            recycle="single_pass",
            selection_function="user",
            selection_params=sel if sel is not None else list(HC_S),
            breakage_function="user",
            breakage_params=brk if brk is not None else [list(r) for r in HC_B],
        )
        return m, cr  # Keep the parent model alive with the unit.

    # A breakage column more than 1e-9 above or below one fails at build time.
    # Column 1 sums to 0.99.
    deficit_col = [
        [1.0, 0.99, 0.4],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
    ]
    with pytest.raises(ConfigurationError, match="sum of column 1 minus 1"):
        _build(brk=deficit_col)
    # Column 1 sums to 1.01.
    excess_col = [[1.0, 1.0, 0.4], [0.0, 0.01, 0.6], [0.0, 0.0, 0.0]]
    with pytest.raises(ConfigurationError, match="sum of column 1 minus 1"):
        _build(brk=excess_col)
    # Test both signs at the 1e-9 limit: values inside stay unchanged;
    # the next representable value outside is rejected.
    for sign in (1, -1):
        for outside in (False, True):
            if sign > 0:
                tail = math.nextafter(1e-9, math.inf) if outside else 1e-9
                brk = [[1.0, 1.0, 0.4], [0.0, tail, 0.6], [0.0, 0.0, 0.0]]
            else:
                diagonal = 1.0 - 1e-9
                if outside:
                    diagonal = math.nextafter(diagonal, -math.inf)
                brk = [[1.0, 0.0, 0.4], [0.0, diagonal, 0.6], [0.0, 0.0, 0.0]]
            defect = math.fsum([brk[0][1], brk[1][1], -1.0])
            assert (abs(defect) > 1e-9) is outside
            if outside:
                with pytest.raises(ConfigurationError, match="sum of column 1 minus 1"):
                    _build(brk=brk)
            else:
                supplied = [list(row) for row in brk]
                m, cr = _build(brk=brk)
                assert value(cr.breakage["OreA", 1, 1]) == supplied[1][1]
                assert brk == supplied
    above_diag = [[1.0, 1.0, 0.4], [0.0, 0.0, 0.6], [0.1, 0.0, 0.0]]
    with pytest.raises(ConfigurationError, match="daughter i > parent j"):
        _build(brk=above_diag)
    neg_entry = [[1.0, 1.2, 0.4], [0.0, -0.2, 0.6], [0.0, 0.0, 0.0]]
    with pytest.raises(ConfigurationError, match="finite and >= 0"):
        _build(brk=neg_entry)
    with pytest.raises(ConfigurationError, match=r"must lie in \[0, 1\]"):
        _build(sel=[0.0, 1.5, 1.0])
    with pytest.raises(ConfigurationError, match="contain 3 values"):
        _build(sel=[0.0, 0.5])
    with pytest.raises(ConfigurationError, match="as a list or tuple"):
        _build(sel=1)
    with pytest.raises(ConfigurationError, match="3x3 matrix"):
        _build(brk=[1, 2, 3])


# Build-time parameter cases on HC_EDGES: (case ID, crusher options, error match).
_VOGEL_B = {"q": 0.05, "dprime": 0.005}
_RULE_WHITEN = {"K1": 2e-3, "K2": 8e-3}
_RULE_KING = {"CSS": 0.025, "alpha1": 0.5, "alpha2": 1.2, "n": 1.0}
_RULE_AUSTIN = {"S1": 0.5, "d1": 4e-3, "a": 0.5}
_RULE_VP = {"f_mat": 1.0, "xw_min": 0.0, "v": 10.0}
_RULE_APPEARANCE = {
    "10": {"t75": 1.0, "t50": 1.5, "t25": 3.0, "t4": 20.0, "t2": 40.0},
    "20": {"t75": 2.0, "t50": 3.0, "t25": 6.0, "t4": 35.0, "t2": 60.0},
}
_RULE_ECS = {"sizes": [2e-3, 8e-3], "t10_rows": [10, 20], "ecs": [[0.4, 0.3]] * 2}


def _single_pass(sel_fn, sel, brk_fn="vogel", brk=None):
    """Return single-pass crusher options, with Vogel breakage by default."""
    return dict(
        psd_method="selection_breakage",
        recycle="single_pass",
        selection_function=sel_fn,
        selection_params=sel,
        breakage_function=brk_fn,
        breakage_params=_VOGEL_B if brk is None else brk,
    )


def _vogel_peukert(params):
    """Return options with Vogel-Peukert selection and Luckie-Austin breakage."""
    return _single_pass("vogel_peukert", params, "luckie_austin", LUCKIE_AUSTIN_REF)


def _t10_recycle(table, t10=15.0, ecs_table=None):
    """Return Whiten-recycle options with t10 appearance breakage.

    Use the pendulum power law when ecs_table is provided.
    """
    kwargs = dict(
        psd_method="selection_breakage",
        recycle="classification_recycle",
        selection_function="whiten",
        selection_params={"K1": 2e-3, "K2": 8e-3, "K3": 2.3},
        breakage_function="t10_appearance",
        breakage_params={"appearance_table": table, "t10": t10},
    )
    if ecs_table is not None:
        kwargs["breakage_params"]["ecs_table"] = ecs_table
        kwargs["power_law"] = "pendulum"
        kwargs["pendulum_power_params"] = {"power_factor": 1.3, "no_load_power": 20.0}
    return kwargs


def _missing(table, row, marker):
    """Return a copy of ``table`` without ``marker`` in ``row``."""
    edited = {key: dict(markers) for key, markers in table.items()}
    del edited[row][marker]
    return edited


def _assert_build_rules(rejections, controls):
    """Build each case on a fresh model; collect every rule that fails."""
    failures = []
    for case_id, kwargs, match in rejections:
        m = one_mineral_flowsheet(edges=HC_EDGES)
        try:
            with pytest.raises(ConfigurationError, match=match):
                crusher(m, **kwargs)
        except (Exception, pytest.fail.Exception) as exc:
            failures.append(f"{case_id}: {exc}")
    for case_id, kwargs in controls:
        m = one_mineral_flowsheet(edges=HC_EDGES)
        try:
            crusher(m, **kwargs)
        except (Exception, pytest.fail.Exception) as exc:
            failures.append(f"{case_id}: {exc}")
    assert not failures, "\n".join(failures)


SELECTION_REJECTIONS = [
    (
        "whiten_K1_zero",
        _single_pass("whiten", dict(_RULE_WHITEN, K1=0.0)),
        "whiten requires K1 > 0 and K2 > 0",
    ),
    (
        "whiten_K2_negative",
        _single_pass("whiten", dict(_RULE_WHITEN, K2=-1e-3)),
        "whiten requires K1 > 0 and K2 > 0",
    ),
    (
        "whiten_K3_zero",
        _single_pass("whiten", dict(_RULE_WHITEN, K3=0.0)),
        "whiten requires K3 > 0",
    ),
    (
        "whiten_K2_below_1.01_K1",
        _single_pass("whiten", {"K1": 0.2, "K2": 0.2}),
        "K2 >= 1.01",
    ),
    (
        "whiten_K2_below_1.01_K1_recycle",
        dict(
            psd_method="selection_breakage",
            recycle="classification_recycle",
            selection_function="whiten",
            selection_params={"K1": 4e-3, "K2": 4e-3, "K3": 2.3},
            breakage_function="user",
            breakage_params=[[1.0, 0.0, 0.4], [0.0, 1.0, 0.6], [0.0, 0.0, 0.0]],
        ),
        "K2 >= 1.01",
    ),
    (
        "whiten_string",
        _single_pass("whiten", {"K1": "0.002", "K2": "0.008"}),
        "not a string",
    ),
    (
        "whiten_boolean",
        _single_pass("whiten", {"K1": 2e-3, "K2": True}),
        "boolean",
    ),
    (
        "whiten_stray_k3",
        _single_pass("whiten", {"K1": 2e-3, "K2": 8e-3, "k3": 2.0}),
        "unknown selection_params",
    ),
    (
        "whiten_params_none",
        _single_pass("whiten", None),
        r"selection_params with keys \['K1', 'K2'\]; missing \['K1', 'K2'\]",
    ),
    (
        "king_CSS_zero",
        _single_pass("king", dict(_RULE_KING, CSS=0.0)),
        "CSS > 0",
    ),
    (
        "king_alpha_order",
        _single_pass("king", dict(_RULE_KING, alpha1=1.2, alpha2=0.5)),
        "alpha1 < alpha2",
    ),
    (
        "king_n_zero",
        _single_pass("king", dict(_RULE_KING, n=0.0)),
        "king requires n > 0",
    ),
    (
        "king_missing_alpha",
        _single_pass("king", {"CSS": 0.025, "n": 1.0}),
        r"selection_params with keys \['CSS', 'alpha1', 'alpha2', 'n'\]; "
        r"missing \['alpha1', 'alpha2'\]",
    ),
    (
        "king_stray_m",
        _single_pass("king", dict(_RULE_KING, m=2.0)),
        "unknown selection_params",
    ),
    (
        "austin_S1_negative",
        _single_pass("austin", dict(_RULE_AUSTIN, S1=-0.1)),
        "austin requires 0 <= S1 <= 1",
    ),
    (
        "austin_S1_above_one",
        _single_pass("austin", dict(_RULE_AUSTIN, S1=1.1)),
        "austin requires 0 <= S1 <= 1",
    ),
    (
        "austin_a_negative",
        _single_pass("austin", dict(_RULE_AUSTIN, a=-0.1)),
        "austin requires 0 <= a <= 1",
    ),
    (
        "austin_a_above_one",
        _single_pass("austin", dict(_RULE_AUSTIN, a=1.5)),
        "austin requires 0 <= a <= 1",
    ),
    (
        "austin_d1_zero",
        _single_pass("austin", {"S1": 0.5, "d1": 0.0, "a": 0.5}),
        "d1 > 0",
    ),
    *(
        (
            f"vogel_peukert_missing_{key}",
            _vogel_peukert({k: v for k, v in _RULE_VP.items() if k != key}),
            "f_mat.*xw_min.*v",
        )
        for key in _RULE_VP
    ),
    (
        "vogel_peukert_stray_key",
        _vogel_peukert(dict(_RULE_VP, k=1.0)),
        "allowed keys",
    ),
    ("vogel_peukert_f_mat_zero", _vogel_peukert(dict(_RULE_VP, f_mat=0.0)), "f_mat"),
    (
        "vogel_peukert_xw_min_negative",
        _vogel_peukert(dict(_RULE_VP, xw_min=-5e-324)),
        "xw_min",
    ),
    (
        "vogel_peukert_v_zero",
        _single_pass("vogel_peukert", dict(_RULE_VP, xw_min=0.1, v=0.0)),
        "v > 0",
    ),
    (
        "whiten_by_species_missing_K1",
        dict(
            psd_method="selection_breakage",
            recycle="single_pass",
            selection_function="whiten",
            selection_params_by_comp={"OreA": {"K2": 8e-3}},
            breakage_function="vogel",
            breakage_params=_VOGEL_B,
        ),
        r"selection_params_by_comp\['OreA'\] with keys \['K1', 'K2'\]; "
        r"missing \['K1'\]",
    ),
]

SELECTION_CONTROLS = [
    ("whiten", _single_pass("whiten", dict(_RULE_WHITEN, K3=2.0))),
    ("king", _single_pass("king", _RULE_KING)),
    ("austin", _single_pass("austin", _RULE_AUSTIN)),
    ("vogel_peukert_f_mat_tiny", _vogel_peukert(dict(_RULE_VP, f_mat=5e-324))),
    ("vogel_peukert_xw_min_zero", _vogel_peukert(dict(_RULE_VP, xw_min=0.0))),
    ("vogel_peukert_v_tiny", _vogel_peukert(dict(_RULE_VP, v=5e-324))),
]


@pytest.mark.unit
def test_selection_params_rejected_at_build():
    _assert_build_rules(SELECTION_REJECTIONS, SELECTION_CONTROLS)


BREAKAGE_REJECTIONS = [
    (
        "luckie_austin_phi_negative",
        _single_pass(
            "whiten", _RULE_WHITEN, "luckie_austin", dict(LUCKIE_AUSTIN_REF, phi=-0.1)
        ),
        "luckie_austin breakage requires 0 <= phi <= 1",
    ),
    (
        "luckie_austin_phi_above_one",
        _single_pass(
            "whiten", _RULE_WHITEN, "luckie_austin", dict(LUCKIE_AUSTIN_REF, phi=1.1)
        ),
        "luckie_austin breakage requires 0 <= phi <= 1",
    ),
    (
        "luckie_austin_gamma_negative",
        _single_pass(
            "user",
            [0.0, 0.5, 1.0],
            "luckie_austin",
            {"phi": 0.5, "gamma": -1.0, "beta": 3.0},
        ),
        "gamma",
    ),
    (
        "reid_stewart_phi_negative",
        _single_pass(
            "whiten", _RULE_WHITEN, "reid_stewart", dict(LUCKIE_AUSTIN_REF, phi=-0.1)
        ),
        "reid_stewart breakage requires 0 <= phi <= 1",
    ),
    (
        "reid_stewart_phi_above_one",
        _single_pass(
            "whiten", _RULE_WHITEN, "reid_stewart", dict(LUCKIE_AUSTIN_REF, phi=1.1)
        ),
        "reid_stewart breakage requires 0 <= phi <= 1",
    ),
    (
        "vogel_dprime_zero",
        _single_pass("user", [0.0, 0.5, 1.0], "vogel", {"q": 0.05, "dprime": 0.0}),
        "dprime",
    ),
    (
        "vogel_dprime_string",
        _single_pass("user", [0.0, 0.5, 1.0], "vogel", {"q": 0.05, "dprime": "0.005"}),
        "not a string",
    ),
    (
        "vogel_stray_Q",
        _single_pass("user", [0.0, 0.5, 1.0], "vogel", dict(_VOGEL_B, Q=0.1)),
        "unknown breakage_params",
    ),
    (
        "vogel_missing_q",
        _single_pass("user", [0.0, 0.5, 1.0], "vogel", {"dprime": 0.005}),
        r"breakage_params with keys \['q', 'dprime'\]; missing \['q'\]",
    ),
    (
        "t10_appearance_single_pass",
        _single_pass(
            "whiten",
            _RULE_WHITEN,
            "t10_appearance",
            {"appearance_table": _RULE_APPEARANCE, "t10": 15.0},
        ),
        "valid only in the Whiten",
    ),
    (
        "appearance_table_list",
        _t10_recycle(list(_RULE_APPEARANCE.values())),
        "appearance_table must be a dict",
    ),
    (
        "t10_below_range",
        _t10_recycle(_RULE_APPEARANCE, 5.0),
        "outside the appearance-table rows",
    ),
    ("t10_nan", _t10_recycle(_RULE_APPEARANCE, float("nan")), "t10 must be finite"),
    (
        "missing_t2",
        _t10_recycle(_missing(_RULE_APPEARANCE, "10", "t2")),
        "missing marker",
    ),
    (
        "row_none",
        _t10_recycle({"10": None, "20": _RULE_APPEARANCE["20"]}),
        "must be a dict",
    ),
    (
        "duplicate_key",
        _t10_recycle(
            {
                **_RULE_APPEARANCE,
                10: {"t75": 9.0, "t50": 9.0, "t25": 9.0, "t4": 9.0, "t2": 9.0},
            }
        ),
        "duplicate",
    ),
    (
        "non_integer_key",
        _t10_recycle({10.9: _RULE_APPEARANCE["10"], "20": _RULE_APPEARANCE["20"]}),
        "integer-like",
    ),
    (
        "unknown_marker",
        _t10_recycle(
            {
                "10": dict(_RULE_APPEARANCE["10"], t20=10.0),
                "20": _RULE_APPEARANCE["20"],
            }
        ),
        r"unknown markers \['t20'\]",
    ),
    (
        "ecs_empty",
        _t10_recycle(_RULE_APPEARANCE, ecs_table=dict(_RULE_ECS, ecs=[])),
        "must be nonempty lists or tuples",
    ),
    (
        "ecs_row_width",
        _t10_recycle(_RULE_APPEARANCE, ecs_table=dict(_RULE_ECS, ecs=[[0.4], [0.8]])),
        "one value per size",
    ),
    (
        "ecs_row_count",
        _t10_recycle(_RULE_APPEARANCE, ecs_table=dict(_RULE_ECS, ecs=[[0.4, 0.3]])),
        "one 'ecs' row per t10 row",
    ),
]

BREAKAGE_CONTROLS = [
    ("vogel", _single_pass("user", [0.0, 0.5, 1.0], "vogel", _VOGEL_B)),
    (
        "luckie_austin",
        _single_pass("whiten", _RULE_WHITEN, "luckie_austin", LUCKIE_AUSTIN_REF),
    ),
    ("t10_appearance_recycle", _t10_recycle(_RULE_APPEARANCE)),
]


@pytest.mark.unit
def test_breakage_params_rejected_at_build():
    _assert_build_rules(BREAKAGE_REJECTIONS, BREAKAGE_CONTROLS)


# Compare generated operators with the named kernels at characteristic sizes.
_ANALYTIC_SELECTION = {
    "king": {"CSS": 4.0e-3, "alpha1": 0.5, "alpha2": 1.2, "n": 2.0},
    "austin": {"S1": 0.5, "d1": 4.0e-3, "a": 0.5},
    "vogel_peukert": {"f_mat": 1.0, "xw_min": 0.1, "v": 10.0},
}

# Hand values of each selection row at size_char = 2**(k + 0.5) mm on HC_EDGES.
_ANALYTIC_SELECTION_ROWS = {
    "king": [0.0, 1.0 - ((4.8 - 2.0 * math.sqrt(2.0)) / 2.8) ** 2, 1.0],
    "austin": [2.0 ** ((k - 3.5) / 2.0) for k in range(3)],
    "vogel_peukert": [
        0.0,
        1.0 - math.exp(-(50.0 * 2.0**1.5 * 1e-3 - 0.1)),
        1.0 - math.exp(-(50.0 * 2.0**2.5 * 1e-3 - 0.1)),
    ],
}


def _stored_breakage(cr, s, n_int):
    return [[value(cr.breakage[s, i, j]) for j in range(n_int)] for i in range(n_int)]


@pytest.mark.unit
def test_generated_operators_follow_named_kernels_at_d_char():
    models = []  # Keep each parent model alive while comparing its unit.

    def build(m, **cfg):
        models.append(m)
        return crusher(m, **cfg)

    def zero_edge_flowsheet(species=("OreA",)):
        return flowsheet(
            edges=ZERO_EDGES,
            species=list(species),
            density={s: {"OreA": 2800.0, "OreB": 4200.0}[s] for s in species},
            bottom_size=ZERO_BOTTOM,
        )

    # With a zero finest edge, selection still uses each interval's
    # characteristic size rather than zero.
    vogel = {"q": 0.5, "dprime": 2.0e-3}
    base = dict(
        psd_method="selection_breakage",
        selection_function="whiten",
        selection_params=ZERO_WHITEN,
        breakage_function="vogel",
        breakage_params=vogel,
    )
    configs = {
        "single_stress": dict(base, recycle="single_pass", n_stress_events=1),
        "repeated_stress": dict(base, recycle="single_pass", n_stress_events=3),
        "classification_recycle": dict(base, recycle="classification_recycle"),
    }
    built = {path: build(zero_edge_flowsheet(), **cfg) for path, cfg in configs.items()}
    size_char = built["single_stress"].config.property_package.size_char_m
    expected_sel = [selection_value("whiten", d, ZERO_WHITEN) for d in size_char]
    assert expected_sel[0] == pytest.approx(0.13090, abs=1e-5)
    for path, cr in built.items():
        row = cr.classification if path == "classification_recycle" else None
        for k in range(3):
            stored = row[k] if row is not None else cr.selection["OreA", k]
            assert value(stored) == expected_sel[k]
        assert value(cr.breakage["OreA", 0, 2]) == pytest.approx(0.10768, abs=1e-5)
        assert value(cr.breakage["OreA", 1, 2]) == pytest.approx(0.14518, abs=1e-5)

    # Compare each selection kernel with the hand-calculated values above.
    for name, params in _ANALYTIC_SELECTION.items():
        m = one_mineral_flowsheet(edges=HC_EDGES)
        cr = build(m, **_single_pass(name, params, "user", [list(r) for r in HC_B]))
        assert [value(cr.selection["OreA", k]) for k in range(3)] == pytest.approx(
            _ANALYTIC_SELECTION_ROWS[name], rel=1e-12
        )

    # Stored breakage matrices must match their kernels exactly and differ by kernel.
    kernel_params = {"phi": 0.5, "gamma": 1.5, "beta": 4.0}
    stored = {}
    for name in ("luckie_austin", "reid_stewart"):
        m = one_mineral_flowsheet(edges=HC_EDGES)
        cr = build(m, **_single_pass("whiten", _RULE_WHITEN, name, kernel_params))
        expected = breakage_matrix(name, HC_EDGES, m.fs.pp.size_char_m, kernel_params)
        stored[name] = _stored_breakage(cr, "OreA", 3)
        assert stored[name] == [list(r) for r in expected]
    assert stored["luckie_austin"] != stored["reid_stewart"]

    # Each component uses its own parameters. Compare stored entries exactly:
    # rebalancing a column would shift OreB's [2, 2] entry.
    _, _, _, _, sel, _, brk = next(
        p for p in PARAMETRIC_SETS if p[0] == "zero_edge_vogel"
    )
    m = zero_edge_flowsheet(("OreA", "OreB"))
    cr = build(
        m,
        **dict(
            base,
            recycle="single_pass",
            selection_params=None,
            selection_params_by_comp=sel,
        ),
    )
    rows = {s: [value(cr.selection[s, k]) for k in range(3)] for s in ("OreA", "OreB")}
    for s in rows:
        assert rows[s] == [selection_value("whiten", d, sel[s]) for d in size_char]
    assert rows["OreA"] != rows["OreB"]
    m = zero_edge_flowsheet(("OreA", "OreB"))
    cr = build(
        m,
        **dict(
            base,
            recycle="classification_recycle",
            breakage_params=None,
            breakage_params_by_comp=brk,
        ),
    )
    matrices = {s: _stored_breakage(cr, s, 3) for s in ("OreA", "OreB")}
    for s, matrix in matrices.items():
        expected = breakage_matrix("vogel", ZERO_EDGES, size_char, brk[s])
        assert matrix == [list(r) for r in expected]
    assert matrices["OreA"] != matrices["OreB"]


@pytest.mark.component
@pytest.mark.solver
def test_independent_individuality_four_combinations():
    # Compare solved rows within the model's flow tolerance.
    band = solved_flow_tolerance(8.0)
    combos = {
        ("common", "common"): {"OreA": [2, 4, 2], "OreB": [2, 4, 2]},
        ("individual", "common"): {"OreA": [2, 4, 2], "OreB": [1, 2, 5]},
        ("common", "individual"): {"OreA": [2, 4, 2], "OreB": [1, 1, 6]},
        ("individual", "individual"): {"OreA": [2, 4, 2], "OreB": [0.5, 0.5, 7]},
    }
    for (sel_form, brk_form), oracle in combos.items():
        cfg = dict(
            psd_method="selection_breakage",
            recycle="single_pass",
            n_stress_events=1,
            selection_function="user",
            breakage_function="user",
        )
        if sel_form == "common":
            cfg["selection_params"] = list(S_A)
        else:
            cfg["selection_params_by_comp"] = {"OreA": list(S_A), "OreB": list(S_B)}
        if brk_form == "common":
            cfg["breakage_params"] = [list(r) for r in B_A]
        else:
            cfg["breakage_params_by_comp"] = {
                "OreA": [list(r) for r in B_A],
                "OreB": [list(r) for r in B_B],
            }
        m = flowsheet(edges=EDGES_X)
        cr = crusher(m, bond_work_index=14.0, **cfg)
        fix_feed(cr.properties_in[0], FEED_X)
        assert degrees_of_freedom(m) == 0
        assert (
            native.CrusherSolidPSDInitializer().initialize(cr)
            == InitializationStatus.Ok
        )
        product = {
            idx: value(var)
            for idx, var in cr.properties_out[0].flow_mass_sized_comp_size.items()
        }
        for s, rows in oracle.items():
            for k, exp in enumerate(rows):
                assert product[(s, k)] == pytest.approx(exp, abs=band)
            assert sum(product[(s, k)] for k in range(3)) == pytest.approx(
                8.0, abs=band
            )
    # Changing OreB's breakage data must leave OreA's product unchanged.
    m = flowsheet(edges=EDGES_X)
    both = crusher(m, bond_work_index=14.0, **x_individual("single_stress"))
    fix_feed(both.properties_in[0], FEED_X)
    assert degrees_of_freedom(m) == 0
    assert (
        native.CrusherSolidPSDInitializer().initialize(both) == InitializationStatus.Ok
    )
    base = {
        idx: value(var)
        for idx, var in both.properties_out[0].flow_mass_sized_comp_size.items()
    }
    m2 = flowsheet(edges=EDGES_X)
    cfg = x_individual("single_stress")
    cfg["breakage_params_by_comp"]["OreB"] = [list(r) for r in B_A]
    swapped = crusher(m2, bond_work_index=14.0, **cfg)
    fix_feed(swapped.properties_in[0], FEED_X)
    assert degrees_of_freedom(m2) == 0
    assert (
        native.CrusherSolidPSDInitializer().initialize(swapped)
        == InitializationStatus.Ok
    )
    changed = {
        idx: value(var)
        for idx, var in swapped.properties_out[0].flow_mass_sized_comp_size.items()
    }
    for k in range(3):
        assert changed[("OreA", k)] == pytest.approx(base[("OreA", k)], abs=2.0 * band)
    for k, exp in enumerate([1, 2, 5]):
        assert changed[("OreB", k)] == pytest.approx(exp, abs=band)


@pytest.mark.component
@pytest.mark.solver
@pytest.mark.parametrize("path", ["repeated_stress"])
def test_per_mineral_conservation_and_separation(path):
    m = flowsheet(edges=EDGES_X)
    cr = crusher(m, bond_work_index=14.0, **x_individual(path))
    fix_feed(cr.properties_in[0], FEED_X)
    native.CrusherSolidPSDInitializer().initialize(cr)
    assert cr.validate_solved_flows() is None
    # These products follow two stress events; stage 1 still holds the
    # single-event result.
    for s, product, stage1 in (
        ("OreA", [7 / 2, 4, 1 / 2], [2, 4, 2]),
        ("OreB", [31 / 32, 29 / 32, 49 / 8], [1 / 2, 1 / 2, 7]),
    ):
        m_s = value(cr.properties_in[0].flow_mass_sized_comp[s])
        band = solved_flow_tolerance(m_s)
        total = sum(
            value(cr.properties_out[0].flow_mass_sized_comp_size[s, k])
            for k in range(3)
        )
        assert total == pytest.approx(m_s, abs=band)
        for k in range(3):
            assert value(
                cr.properties_out[0].flow_mass_sized_comp_size[s, k]
            ) == pytest.approx(product[k], abs=band)
            assert value(cr.psd_stage[0, 1, s, k]) == pytest.approx(stage1[k], abs=band)


@pytest.mark.component
@pytest.mark.solver
@pytest.mark.parametrize("path", ["single_stress"])
def test_zero_to_positive_mineral_feed(path):
    m = flowsheet(edges=EDGES_X)
    cr = crusher(m, bond_work_index=14.0, **x_individual(path))
    fix_feed(cr.properties_in[0], {"OreA": [0, 0, 8.0], "OreB": [0, 0, 0.0]})
    init = native.CrusherSolidPSDInitializer()
    init.initialize(cr)
    assert init.summary[cr]["status"] == InitializationStatus.Ok
    for k in range(3):
        assert (
            abs(value(cr.properties_out[0].flow_mass_sized_comp_size["OreB", k]))
            <= 1e-12
        )
    for k, exp in enumerate([2.0, 4.0, 2.0]):
        assert value(
            cr.properties_out[0].flow_mass_sized_comp_size["OreA", k]
        ) == pytest.approx(exp, abs=solved_flow_tolerance(8.0))
    ore_a_before = [
        value(cr.properties_out[0].flow_mass_sized_comp_size["OreA", k])
        for k in range(3)
    ]
    for k, v in enumerate([0, 0, 8.0]):
        cr.properties_in[0].flow_mass_sized_comp_size["OreB", k].fix(v)
    init2 = native.CrusherSolidPSDInitializer()
    init2.initialize(cr)
    assert init2.summary[cr]["status"] == InitializationStatus.Ok
    for k, exp in enumerate([0.5, 0.5, 7.0]):
        assert value(
            cr.properties_out[0].flow_mass_sized_comp_size["OreB", k]
        ) == pytest.approx(exp, abs=solved_flow_tolerance(8.0))
    for k, exp in enumerate([2.0, 4.0, 2.0]):
        assert value(
            cr.properties_out[0].flow_mass_sized_comp_size["OreA", k]
        ) == pytest.approx(exp, abs=solved_flow_tolerance(8.0))
        assert value(
            cr.properties_out[0].flow_mass_sized_comp_size["OreA", k]
        ) == pytest.approx(ore_a_before[k], abs=solved_flow_tolerance(8.0))


def _solved_three_event_unit():
    """Build and solve a three-event crusher on EDGES_X; return its model and unit."""
    m = flowsheet(edges=EDGES_X)
    cr = crusher(
        m,
        bond_work_index=14.0,
        **dict(x_individual("repeated_stress"), n_stress_events=3),
    )
    fix_feed(cr.properties_in[0], FEED_X)
    native.CrusherSolidPSDInitializer().initialize(cr)
    return m, cr


@pytest.mark.component
@pytest.mark.solver
@pytest.mark.parametrize("case", ["later_stage_sign"])
def test_closure_residual_first_and_later_stage(case):
    # Stage 1 balances against the inlet component flow; later stages
    # balance against the preceding stage's total.
    m, cr = _solved_three_event_unit()
    assert cr.validate_solved_flows() is None
    t = m.fs.time.first()
    order = list(m.fs.pp.size_interval_set)

    def _shift(r, amount):
        target = cr.psd_stage[t, r, "OreA", 0]
        target.set_value(target.value + amount)

    inlet_mass = sum(
        value(cr.properties_in[t].flow_mass_sized_comp_size["OreA", k]) for k in order
    )
    band = solved_flow_tolerance(inlet_mass)
    # Stage 1 rises by 0.9 tolerance and stage 2 falls by 0.9.
    # Stage 2 is then 1.8 tolerances below its incoming stage and fails.
    _shift(1, 0.9 * band)
    _shift(2, -0.9 * band)
    with pytest.raises(
        InitializationError,
        match=r"stage mass balance failed.*stage 2, component 'OreA'",
    ):
        cr.validate_solved_flows()
    # Shifting both stages equally keeps stage 2 balanced with stage 1;
    # only stage 1 fails against the inlet.
    _shift(1, -0.9 * band)
    _shift(2, 0.9 * band)
    _shift(1, 1e-3)
    _shift(2, 1e-3)
    with pytest.raises(
        InitializationError,
        match=r"stage mass balance failed at t = 0\.0, stage 1, component 'OreA'",
    ) as exc:
        cr.validate_solved_flows()
    assert "stage 2" not in str(exc.value)


@pytest.mark.component
@pytest.mark.solver
@pytest.mark.parametrize("n_int", [1])
def test_one_interval_identity_crusher_preserves_feed_and_has_zero_duty(n_int):
    # Full selection with identity breakage preserves the single feed interval.
    edges = geometric_series(2.0e-3, 1.0e-3, 2)
    assert len(edges) == n_int + 1
    identity = [[1.0 if i == j else 0.0 for j in range(n_int)] for i in range(n_int)]
    m = one_mineral_flowsheet(edges=edges)
    cr = crusher(
        m,
        psd_method="selection_breakage",
        recycle="single_pass",
        n_stress_events=1,
        selection_function="user",
        selection_params=[1.0] * n_int,
        breakage_function="user",
        breakage_params=identity,
    )
    feed = [float(k + 1) for k in range(n_int)]
    fix_feed(cr.properties_in[0], {"OreA": feed})
    assert cr.validate_crushing_coefficients() is None
    assert degrees_of_freedom(m) == 0
    init = native.CrusherSolidPSDInitializer(solver_options=ZERO_POWER_SOLVER_OPTIONS)
    assert init.initialize(cr) == InitializationStatus.Ok
    band = solved_flow_tolerance(sum(feed))
    for k in range(n_int):
        assert value(
            cr.properties_out[0].flow_mass_sized_comp_size["OreA", k]
        ) == pytest.approx(feed[k], abs=band)
    assert value(cr.power[0]) == pytest.approx(0.0, abs=1e-8)
