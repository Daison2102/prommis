#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Test crusher operator arithmetic, validation, and component flows."""

from copy import deepcopy

import pytest

from pyomo.environ import units, value

from idaes.core.initialization import InitializationStatus
from idaes.core.util.exceptions import ConfigurationError

from prommis.comminution.functions.selection import selection_value
from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models.tests.crusher_test_support import (
    COMMON_B,
    COMMON_S,
    WHITEN_REF,
    ZERO_POWER_SOLVER_OPTIONS,
    solved_flow_tolerance,
    crusher,
    fix_feed,
    flowsheet,
    recycle_reference,
    stress_reference,
    stress_unit,
)


@pytest.mark.unit
def test_stress_operator_mass_balance_checks(caplog):
    # A 1e-10 column shortfall exceeds the mass-balance limit after 100 events.
    caplog.set_level("WARNING")
    selection = [1.0, 1.0]
    cr = stress_unit(2, selection, [[1.0, 0.03125], [0.0, 0.96875]], 100)
    assert cr.validate_crushing_coefficients() is None
    assert "exceeds the allowance" not in caplog.text

    # One event keeps the shortfall below the limit but above the rounding allowance.
    deficit = [[1.0, 0.03125 - 1e-10], [0.0, 0.96875]]
    supplied = deepcopy(deficit)
    cr = stress_unit(2, selection, deficit, 1)
    assert cr.validate_crushing_coefficients() is None
    assert caplog.text.count("exceeds the allowance") == 1
    assert deficit == supplied
    cr = stress_unit(2, selection, deficit, 100)
    with pytest.raises(
        ConfigurationError, match="accumulated closure correction"
    ) as exc:
        cr.validate_crushing_coefficients()
    assert "column 1" in str(exc.value)
    assert "100 stress events" in str(exc.value)

    # Other bins receive over 100% of the parent, leaving a negative finest coefficient.
    excess = [[1.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.5 + 1e-10]]
    cr = stress_unit(3, [1.0] * 3, excess, 1)
    with pytest.raises(ConfigurationError, match="effective finest coefficient") as exc:
        cr.validate_crushing_coefficients()
    assert "column 2" in str(exc.value)


# Different feed shapes expose an incorrect aggregate product split by
# mass share, even when the total product mass is correct.
COMMON_RECYCLE_B = [
    [1.0, 0.6, 0.6, 0.6],
    [0.0, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.0],
    [0.0, 0.0, 0.0, 0.4],
]
COMMON_FEED = {"OreA": [0.0, 0.5, 1.5, 4.0], "OreB": [0.2, 0.0, 0.8, 3.0]}
COMMON_EVENTS = {"single_stress": 1, "repeated_stress": 3}


def common_operator_unit(path):
    """Build a crusher for ``path`` and fix the shared feed."""
    m = flowsheet()
    if path == "classification_recycle":
        cr = crusher(
            m,
            psd_method="selection_breakage",
            recycle="classification_recycle",
            selection_function="whiten",
            selection_params=WHITEN_REF,
            breakage_function="user",
            breakage_params=COMMON_RECYCLE_B,
        )
    else:
        cr = crusher(
            m,
            psd_method="selection_breakage",
            recycle="single_pass",
            selection_function="user",
            selection_params=COMMON_S,
            breakage_function="user",
            breakage_params=COMMON_B,
            n_stress_events=COMMON_EVENTS[path],
        )
    fix_feed(cr.properties_in[0], COMMON_FEED)
    return m, cr


def common_whiten_classification(cr):
    """Return Whiten classification fractions from the shared selection inputs."""
    pp = cr.config.property_package
    return [
        selection_value(
            "whiten",
            value(units.convert(pp.size_char[k], to_units=units.m)),
            WHITEN_REF,
        )
        for k in pp.size_interval_set
    ]


def common_operator_reference(
    cr, path, feed, selection=None, breakage=None, classification=None
):
    """Return reference working rows and product flows for one feed.

    Working rows are indexed by variable name and stage index, if any.
    """
    working = {}
    if path == "classification_recycle":
        load, product = recycle_reference(
            COMMON_RECYCLE_B if breakage is None else breakage,
            (
                common_whiten_classification(cr)
                if classification is None
                else classification
            ),
            feed,
        )
        working["recycle_load"] = {(): load}
        return working, product
    selection = COMMON_S if selection is None else selection
    breakage = COMMON_B if breakage is None else breakage
    events = COMMON_EVENTS[path]
    if events > 1:
        working["psd_stage"] = {
            (r,): stress_reference(selection, breakage, feed, r)
            for r in range(1, events)
        }
    return working, stress_reference(selection, breakage, feed, events)


@pytest.mark.component
@pytest.mark.solver
@pytest.mark.parametrize(
    "path", ["single_stress", "repeated_stress", "classification_recycle"]
)
def test_common_operator_keeps_minerals_separate(path):
    m, cr = common_operator_unit(path)
    init = native.CrusherSolidPSDInitializer(solver_options=ZERO_POWER_SOLVER_OPTIONS)
    assert init.initialize(cr) == InitializationStatus.Ok
    order = list(cr.config.property_package.size_interval_set)
    references = {}
    for s, feed in COMMON_FEED.items():
        working, product = common_operator_reference(cr, path, feed)
        references[s] = product
        band = solved_flow_tolerance(sum(feed))
        targets = [
            (cr.properties_out[0].flow_mass_sized_comp_size[s, k], product[k])
            for k in order
        ]
        for name, rows in working.items():
            component = getattr(cr, name)
            targets += [
                (component[(0, *index, s, k)], row[k])
                for index, row in rows.items()
                for k in order
            ]
        for target, expected in targets:
            solved = value(units.convert(target, to_units=units.kg / units.s))
            assert solved == pytest.approx(expected, abs=band)
    aggregate = [sum(feed[k] for feed in COMMON_FEED.values()) for k in order]
    _, blind = common_operator_reference(cr, path, aggregate)
    for s, feed in COMMON_FEED.items():
        share = sum(feed) / sum(aggregate)
        assert max(abs(share * blind[k] - references[s][k]) for k in order) > 0.1
