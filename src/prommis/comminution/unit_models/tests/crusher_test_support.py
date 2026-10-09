#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Shared builders, test data, and reference arithmetic for crusher tests."""

from pyomo.environ import ConcreteModel, Param, Var, units, value

from idaes.core import FlowsheetBlock

from prommis.comminution.core.size_mesh import geometric_series
from prommis.comminution.properties.solid_psd_properties import (
    SolidPSDParameterBlock,
)
from prommis.comminution.unit_models import crusher as native
from prommis.comminution.unit_models.crusher import CrusherSolidPSD

# Reference mesh: four intervals from 1 to 16 mm.
EDGES_REF = geometric_series(16e-3, 1e-3, 2)
# Test mesh with geometric-mean interval sizes of 1, 3, and 4 mm.
EDGES_X = [0.4e-3, 2.5e-3, 3.6e-3, (40.0 / 9.0) * 1e-3]
SPECIES_TWO = ["OreA", "OreB"]
DENSITY_TWO = {"OreA": 2800.0, "OreB": 4200.0}


def flowsheet(
    edges=None, species=None, density=None, unsized=None, vapors=None, bottom_size=None
):
    """A flowsheet with one solid PSD property package."""
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    kwargs = dict(
        size_edges=list(EDGES_REF if edges is None else edges),
        sized_solid_component_list=list(SPECIES_TWO if species is None else species),
        solid_density=dict(DENSITY_TWO if density is None else density),
    )
    if unsized:
        kwargs["unsized_solid_component_list"] = list(unsized)
    if vapors:
        kwargs["vapor_component_list"] = list(vapors)
    if bottom_size is not None:
        kwargs["bottom_size"] = bottom_size
    m.fs.pp = SolidPSDParameterBlock(**kwargs)
    return m


def one_mineral_flowsheet(edges=None):
    """A flowsheet whose package carries the one component OreA."""
    return flowsheet(edges=edges, species=["OreA"], density={"OreA": 2800.0})


def two_time_flowsheet(edges=None, species=None, density=None):
    """A flowsheet on the two-point time set [0.0, 1.0]."""
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False, time_set=[0.0, 1.0], time_units=units.s)
    m.fs.pp = SolidPSDParameterBlock(
        size_edges=list(EDGES_REF if edges is None else edges),
        sized_solid_component_list=list(species or ["OreA"]),
        solid_density=dict(density or {"OreA": 2800.0}),
    )
    return m


def crusher(m, **cfg):
    """Build ``m.fs.cr`` on the flowsheet's package with ``cfg``."""
    m.fs.cr = CrusherSolidPSD(property_package=m.fs.pp, **cfg)
    return m.fs.cr


def stress_unit(n_int, selection, breakage, events):
    """Build a single-pass crusher with the given selection and breakage data."""
    edges = geometric_series(1e-3 * 2**n_int, 1e-3, 2)
    assert len(edges) == n_int + 1
    m = one_mineral_flowsheet(edges=edges)
    return crusher(
        m,
        psd_method="selection_breakage",
        recycle="single_pass",
        selection_function="user",
        selection_params=selection,
        breakage_function="user",
        breakage_params=breakage,
        n_stress_events=events,
    )


def fix_feed(
    feed, rows, liquid=0.0, unsized=None, temperature=298.15, pressure=101325.0
):
    """Fix a feed from ``rows = {component: [kg/s per interval]}``."""
    for s, row in rows.items():
        for k, v in enumerate(row):
            feed.flow_mass_sized_comp_size[s, k].fix(v)
    for j in feed.params.liquid_list:
        feed.flow_mass_liquid_comp[j].fix(liquid)
    for u, v in (unsized or {}).items():
        feed.flow_mass_unsized_comp[u].fix(v)
    feed.temperature.fix(temperature)
    feed.pressure.fix(pressure)


def state_snapshot(block):
    """Return variable and mutable parameter values as string, along with whether
    each variable is fixed.
    """
    return {
        obj.name: (repr(obj.value), obj.fixed)
        for obj in block.component_data_objects(Var)
    } | {
        obj.name: repr(obj.value)
        for param in block.component_objects(Param)
        if param.mutable
        for obj in param.values()
    }


def solved_flow_tolerance(flow):
    """Return the solved-flow tolerance."""
    return 1e-9 + 1e-6 * abs(flow)


WHITEN_REF = {"K1": 1.0e-3, "K2": 16.0e-3, "K3": 2.3}
LUCKIE_AUSTIN_REF = {"phi": 0.4, "gamma": 0.9, "beta": 4.0}
SINGLE_STRESS_REF = dict(
    psd_method="selection_breakage",
    recycle="single_pass",
    n_stress_events=1,
    selection_function="whiten",
    selection_params=WHITEN_REF,
    breakage_function="luckie_austin",
    breakage_params=LUCKIE_AUSTIN_REF,
)
CLASSIFICATION_RECYCLE_REF = dict(SINGLE_STRESS_REF, recycle="classification_recycle")

RR = native.DistributionShape.rosin_rammler
GGS = native.DistributionShape.gates_gaudin_schuhmann

# one configuration per product path, on the reference mesh
PATH_CONFIGS = {
    "tabular": dict(psd_method="tabular", tabular_psd=[0.25, 0.5, 0.25, 0.0]),
    "distribution_size_spec": dict(
        psd_method="distribution_function", shape_exponent=1.5, target_product_p80=5e-3
    ),
    "distribution_power_spec": dict(
        psd_method="distribution_function", shape_exponent=1.5
    ),
    "single_stress": dict(SINGLE_STRESS_REF),
    "repeated_stress": dict(SINGLE_STRESS_REF, n_stress_events=3),
    "classification_recycle": dict(CLASSIFICATION_RECYCLE_REF),
}

# Synthetic drop-weight data for pendulum power tests.
PENDULUM_APPEARANCE = {
    "10": {"t75": 1.0, "t50": 1.5, "t25": 3.0, "t4": 20.0, "t2": 40.0},
    "20": {"t75": 2.0, "t50": 3.0, "t25": 6.0, "t4": 35.0, "t2": 60.0},
}
PENDULUM_ECS_A = {
    "sizes": [2e-3, 8e-3],
    "t10_rows": [10, 20, 30],
    "ecs": [[0.4, 0.3], [0.8, 0.6], [1.1, 0.9]],
}
PENDULUM_REF = dict(
    CLASSIFICATION_RECYCLE_REF,
    breakage_function="t10_appearance",
    breakage_params={
        "appearance_table": PENDULUM_APPEARANCE,
        "t10": 15.0,
        "ecs_table": PENDULUM_ECS_A,
    },
    power_law="pendulum",
    pendulum_power_params={"power_factor": 1.3, "no_load_power": 20.0},
)

# a classification-recycle breakage matrix on the reference mesh
WHITEN_B4 = [
    [1.0, 0.0, 0.4, 0.3],
    [0.0, 1.0, 0.6, 0.3],
    [0.0, 0.0, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0],
]

# Three-interval mesh and feed for hand calculations.
HC_EDGES = [1e-3, 2e-3, 4e-3, 8e-3]
HC_FEED = {"OreA": [1.0, 2.0, 3.0]}

# Disable IPOPT bound relaxation.
ZERO_POWER_SOLVER_OPTIONS = {
    "tol": 1e-10,
    "constr_viol_tol": 1e-10,
    "bound_relax_factor": 0,
    "bound_push": 1e-8,
    "bound_frac": 1e-8,
}
# reference feed on EDGES_REF: all mass in the top interval
FEED_REF = {"OreA": [0.0, 0.0, 0.0, 6.0], "OreB": [0.0, 0.0, 0.0, 4.0]}
# Rounded smooth 80%-passing size of FEED_REF (m) for power tests.
F80_LIVE_REF = 0.01439999
# Bond duty for 10 kg/s of FEED_REF at Wi=12 kWh/t, using the Rosin-Rammler
# P80 at the largest valid d_63 (n=1.5, top-edge passing 0.95).
REFERENCE_DUTY_KW = 6.011346
# Each component enters with 8 kg/s in the coarsest interval.
FEED_X = {"OreA": [0.0, 0.0, 8.0], "OreB": [0.0, 0.0, 8.0]}

# User selection and breakage data for the EDGES_X test mesh.
S_A = (0.0, 0.5, 1.0)
S_B = (0.0, 0.25, 0.5)
B_A = ((1.0, 0.5, 0.25), (0.0, 0.5, 0.5), (0.0, 0.0, 0.25))
B_B = ((1.0, 0.25, 0.125), (0.0, 0.75, 0.125), (0.0, 0.0, 0.75))
WHITEN_X = {"K1": 1.0e-3, "K2": 5.0e-3, "K3": 1.0}  # C = [0, 0.5, 0.75] on X


def x_common(path):
    """Return common inputs for ``path`` on the EDGES_X test mesh."""
    base = dict(
        psd_method="selection_breakage",
        selection_function="user",
        breakage_function="user",
    )
    if path == "single_stress":
        return dict(
            base,
            recycle="single_pass",
            n_stress_events=1,
            selection_params=list(S_A),
            breakage_params=[list(r) for r in B_A],
        )
    if path == "repeated_stress":
        return dict(
            base,
            recycle="single_pass",
            n_stress_events=2,
            selection_params=list(S_A),
            breakage_params=[list(r) for r in B_A],
        )
    return dict(
        psd_method="selection_breakage",
        recycle="classification_recycle",
        selection_function="whiten",
        selection_params=WHITEN_X,
        breakage_function="user",
        breakage_params=[list(r) for r in B_A],
    )


def x_individual(path):
    """Return EDGES_X inputs with breakage data for each component.

    Single-pass paths also use selection data for each component.
    """
    cfg = x_common(path)
    if path in ("single_stress", "repeated_stress"):
        cfg.pop("selection_params")
        cfg["selection_params_by_comp"] = {"OreA": list(S_A), "OreB": list(S_B)}
    cfg.pop("breakage_params")
    cfg["breakage_params_by_comp"] = {
        "OreA": [list(r) for r in B_A],
        "OreB": [list(r) for r in B_B],
    }
    return cfg


# Per-component parameter cases for zero-edge and reference meshes.
ZERO_EDGES = [0.0, 1.0e-3, 2.0e-3, 4.0e-3]
ZERO_BOTTOM = 0.5e-3
ZERO_WHITEN = {"K1": 0.5e-3, "K2": 4.0e-3, "K3": 2.3}

PARAMETRIC_SETS = [
    (
        "zero_edge_luckie_austin",
        ZERO_EDGES,
        ZERO_BOTTOM,
        "whiten",
        {"OreA": ZERO_WHITEN, "OreB": {"K1": 0.5e-3, "K2": 3.0e-3, "K3": 2.0}},
        "luckie_austin",
        {
            "OreA": {"phi": 0.4, "gamma": 0.9, "beta": 4.0},
            "OreB": {"phi": 0.6, "gamma": 1.1, "beta": 3.0},
        },
    ),
    (
        "zero_edge_vogel",
        ZERO_EDGES,
        ZERO_BOTTOM,
        "whiten",
        {"OreA": ZERO_WHITEN, "OreB": {"K1": 0.5e-3, "K2": 3.0e-3, "K3": 2.0}},
        "vogel",
        {"OreA": {"q": 0.5, "dprime": 2.0e-3}, "OreB": {"q": 0.7, "dprime": 1.5e-3}},
    ),
    (
        "parity_whiten_luckie_austin",
        list(EDGES_REF),
        None,
        "whiten",
        {"OreA": WHITEN_REF, "OreB": {"K1": 1.0e-3, "K2": 12.0e-3, "K3": 2.0}},
        "luckie_austin",
        {"OreA": LUCKIE_AUSTIN_REF, "OreB": {"phi": 0.5, "gamma": 1.0, "beta": 3.5}},
    ),
]

# Starting-value test helpers.


class StartingValueRecorder(native.CrusherSolidPSDInitializer):
    """Record each proposed value and setter outcome by target name.

    Each ``applied`` entry is ``(target_name, proposed_value, disposition)``,
    where disposition is ``"written"`` or ``"skipped_fixed"``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.applied = []

    def _seed_if_unfixed(self, model, target, starting_value):
        disposition = super()._seed_if_unfixed(model, target, starting_value)
        self.applied.append((target.name, starting_value, disposition))
        return disposition

    def written(self, target):
        """Return the recorded value written to ``target``, or ``None`` if none
        was written.
        """
        name = getattr(target, "name", target)
        for seen, starting_value, disposition in self.applied:
            if seen == name and disposition == "written":
                return starting_value
        return None

    def skipped(self):
        """The names of the fixed targets the setter refused to write."""
        return [name for name, _, d in self.applied if d == "skipped_fixed"]


def starting_values(cr, initializer=None):
    """Return proposed starting values at every time point, indexed by target name.

    Leave model values unchanged and propagate operating-condition errors.
    """
    init = initializer or native.CrusherSolidPSDInitializer()
    return {
        target.name: proposed
        for target, proposed in init._calculate_starting_values(
            cr, cr._prepare_crushing_data()
        )
    }


def proposed_product(cr, t=0):
    """Return proposed outlet size flows at ``t``, indexed by
    ``(component, interval)``.
    """
    proposed = starting_values(cr)
    pp = cr.config.property_package
    out = cr.properties_out[t]
    return {
        (s, k): proposed[out.flow_mass_sized_comp_size[s, k].name]
        for s in pp.sized_solid_list
        for k in pp.size_interval_set
    }


def feed_percentile(cr, t=0):
    """The live feed 80%-passing size in metres the property package reports."""
    return value(cr.properties_in[t].percentile_size[0.8])


def sized_feed_flow(cr, t=0):
    """The sized feed mass flow in kg/s."""
    return value(
        units.convert(cr.properties_in[t].flow_mass_sized, to_units=units.kg / units.s)
    )


def product_percentile(cr, rows, target=0.8):
    """Return the smooth percentile of product ``rows`` in metres."""
    pp = cr.config.property_package
    order = list(pp.size_interval_set)
    return pp.size_at_passing_from_flows(
        {s: [rows[s, k] for k in order] for s in pp.sized_solid_list}, target=target
    )


# Expected stress and recycle products.
def stress_reference(selection, breakage, feed, events):
    """Return the product row after ``events`` stress events.

    Each event balances the finest bin against its incoming total.
    """
    n = len(feed)
    a = [
        [
            float(k == j) * (1.0 - selection[j]) + breakage[k][j] * selection[j]
            for j in range(n)
        ]
        for k in range(n)
    ]
    row = list(feed)
    for _ in range(events):
        product = [sum(a[k][j] * row[j] for j in range(n)) for k in range(n)]
        product[0] = sum(row) - sum(product[1:])
        row = product
    return row


def recycle_reference(breakage, classification, feed):
    """Return recycle loads and product rows.

    The finest product bin balances the total product against the feed.
    """
    n = len(feed)
    load = [0.0] * n
    for k in reversed(range(n)):
        coarser = sum(
            breakage[k][j] * classification[j] * load[j] for j in range(k + 1, n)
        )
        load[k] = (feed[k] + coarser) / (1.0 - breakage[k][k] * classification[k])
    product = [(1.0 - classification[k]) * load[k] for k in range(n)]
    product[0] = sum(feed) - sum(product[1:])
    return load, product


# Shared four-interval selection and breakage data for two-component tests.
COMMON_S = [0.0, 0.3, 0.7, 1.0]
COMMON_B = [
    [1.0, 0.5, 0.25, 0.125],
    [0.0, 0.5, 0.5, 0.125],
    [0.0, 0.0, 0.25, 0.25],
    [0.0, 0.0, 0.0, 0.5],
]
