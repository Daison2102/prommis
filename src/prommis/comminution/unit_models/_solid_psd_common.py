#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Shared helpers for unit models using the solid PSD property package.

State construction, pass-through constraints, and scaling assume one inlet
and one outlet. The constraints pass unsized solids, liquids, vapor,
temperature, and pressure through unchanged; each unit defines its own
sized-solid equations. The property-package check and parameter-rebuild
functions also work for units with different numbers of inlets or outlets.
Registered Params rebuild in registration order.
"""

__author__ = "Daison Yancy Caballero"

from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import VarData
from pyomo.environ import units, value

from idaes.core.initialization import InitializerBase
from idaes.core.scaling import CustomScalerBase
from idaes.core.util.exceptions import ConfigurationError

from prommis.comminution.core.unit_utils import (
    PROPAGATED_STATE_VAR_NAMES,
    FactorSource,
    declare_factor_source,
    delegate_state_block_scaling,
    fixed_magnitude_or_none,
    propagate_scaling_factors,
    required_scaling_factor,
    run_product_block_overrides,
    scale_constraints_at_time,
    scale_rows_by_largest_term,
)
from prommis.comminution.properties.solid_psd_properties import (
    SolidPSDParameterData,
    SolidPSDScaler,
)

_PASSTHROUGH_ROWS = (
    ("liquid_passthrough_eqn", "flow_mass_liquid_comp"),
    ("unsized_passthrough_eqn", "flow_mass_unsized_comp"),
    ("vapor_passthrough_eqn", "flow_mass_vapor_comp"),
)


def require_solid_psd_package(config, unit_name):
    """Return the configured solid PSD package or raise ``ConfigurationError``."""
    pp = config.property_package
    if not isinstance(pp, SolidPSDParameterData):
        raise ConfigurationError(
            f"{unit_name} requires a SolidPSDParameterBlock property package."
        )
    return pp


def build_solid_psd_state_blocks(unit):
    """Build the two state blocks and the ``inlet``/``outlet`` ports."""
    pp = unit.config.property_package
    time = unit.flowsheet().time
    unit.properties_in = pp.build_state_block(
        time, defined_state=True, **unit.config.property_package_args
    )
    unit.properties_out = pp.build_state_block(
        time, defined_state=False, **unit.config.property_package_args
    )
    unit.add_port("inlet", unit.properties_in)
    unit.add_port("outlet", unit.properties_out)
    return unit.properties_in, unit.properties_out


def add_passthrough_constraints(unit):
    """Add pass-through equations for unsized solids, liquids, vapor,
    temperature, and pressure.
    """
    pp = unit.config.property_package
    time = unit.flowsheet().time

    @unit.Constraint(
        time,
        pp.liquid_list,
        doc="Per-component liquid mass flow passes through unchanged.",
    )
    def liquid_passthrough_eqn(b, t, j):
        return (
            b.properties_out[t].flow_mass_liquid_comp[j]
            == b.properties_in[t].flow_mass_liquid_comp[j]
        )

    if pp.unsized_solid_list:

        @unit.Constraint(
            time, pp.unsized_solid_list, doc="Unsized solids pass through unchanged."
        )
        def unsized_passthrough_eqn(b, t, u):
            return (
                b.properties_out[t].flow_mass_unsized_comp[u]
                == b.properties_in[t].flow_mass_unsized_comp[u]
            )

    if pp.vapor_list:

        @unit.Constraint(
            time,
            pp.vapor_list,
            doc="Vapor component mass flow passes through unchanged.",
        )
        def vapor_passthrough_eqn(b, t, v):
            return (
                b.properties_out[t].flow_mass_vapor_comp[v]
                == b.properties_in[t].flow_mass_vapor_comp[v]
            )

    @unit.Constraint(time, doc="Temperature pass-through.")
    def temperature_eqn(b, t):
        return b.properties_out[t].temperature == b.properties_in[t].temperature

    @unit.Constraint(time, doc="Pressure pass-through.")
    def pressure_eqn(b, t):
        return b.properties_out[t].pressure == b.properties_in[t].pressure


def report_applicability_warnings(logger, unit_name, messages, emit_warning=True):
    """Return applicability messages and optionally log them to ``logger``."""
    messages = list(messages)
    if emit_warning:
        for msg in messages:
            logger.warning("%s applicability: %s", unit_name, msg)
    return messages


class SolidPSDUnitInitializerBase(InitializerBase):
    """Provide fixed-aware assignment and pass-through seeding for
    solid PSD units.
    """

    CONFIG = InitializerBase.CONFIG()

    def _seed_if_unfixed(self, model, target, value):
        """Set an unfixed ``VarData`` and return ``"written"`` or ``"skipped_fixed"``.

        Raise ``TypeError`` with the component name if ``target`` is not ``VarData``.
        """
        if not isinstance(target, VarData):
            name = getattr(target, "name", repr(target))
            raise TypeError(
                f"seed target {name} is not a VarData; only Var data may be seeded."
            )
        if target.fixed:
            return "skipped_fixed"
        target.set_value(value)
        return "written"

    def _seed_passthrough(self, model, t):
        """Seed unfixed outlet flows, temperature, and pressure from the inlet."""
        pp = model.config.property_package
        feed = model.properties_in[t]
        out = model.properties_out[t]
        for unsized in pp.unsized_solid_list:
            self._seed_if_unfixed(
                model,
                out.flow_mass_unsized_comp[unsized],
                value(feed.flow_mass_unsized_comp[unsized]),
            )
        for liquid in pp.liquid_list:
            self._seed_if_unfixed(
                model,
                out.flow_mass_liquid_comp[liquid],
                value(feed.flow_mass_liquid_comp[liquid]),
            )
        for vapor in pp.vapor_list:
            self._seed_if_unfixed(
                model,
                out.flow_mass_vapor_comp[vapor],
                value(feed.flow_mass_vapor_comp[vapor]),
            )
        self._seed_if_unfixed(model, out.temperature, value(feed.temperature))
        self._seed_if_unfixed(model, out.pressure, value(feed.pressure))


class SolidPSDUnitScalerBase(CustomScalerBase):
    """Scale shared state blocks and pass-through constraints.

    Input-based mode scales inlet flows from fixed values or fallback factors.
    Each outlet sized flow uses its component's summed inlet nominal unless
    its factor is retained or set by an outlet scaler. Other outlet factors
    propagate from the inlet. Current-values mode scales both state blocks
    from their current values.
    """

    CONFIG = declare_factor_source(CustomScalerBase.CONFIG())
    _STATE_VAR_NAMES = PROPAGATED_STATE_VAR_NAMES
    INPUT_BASED_DEFAULT_POWER_FACTOR = 1.0

    def variable_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """Scale inlet and outlet state variables using the configured factor source."""
        if self.config.factor_source == FactorSource.input_based:
            self._scale_states_input_based(model, overwrite, submodel_scalers)
            return
        delegate_state_block_scaling(
            self,
            (model.properties_in, model.properties_out),
            SolidPSDScaler,
            submodel_scalers,
            "variable_scaling_routine",
            overwrite=overwrite,
        )

    def constraint_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """Delegate state-block scaling and scale pass-through constraints."""
        delegate_state_block_scaling(
            self,
            (model.properties_in, model.properties_out),
            SolidPSDScaler,
            submodel_scalers,
            "constraint_scaling_routine",
            overwrite=overwrite,
        )
        # Scale each pass-through row from its own inlet and outlet terms, so
        # a zero-flow component does not inherit the total feed scale.
        for name, var_name in _PASSTHROUGH_ROWS:
            scale_rows_by_largest_term(
                self,
                model,
                name,
                lambda idx, v=var_name: [
                    getattr(blk[idx[0]], v)[idx[1]]
                    for blk in (model.properties_in, model.properties_out)
                ],
                overwrite=overwrite,
            )
        for t in model.flowsheet().time:
            feed = model.properties_in[t]
            scale_constraints_at_time(
                self,
                model,
                [
                    (
                        "temperature_eqn",
                        required_scaling_factor(self, feed.temperature),
                    ),
                    ("pressure_eqn", required_scaling_factor(self, feed.pressure)),
                ],
                t,
                overwrite=overwrite,
            )

    def _scale_states_input_based(self, model, overwrite, submodel_scalers):
        """Scale the inlet, apply any outlet scaler, then fill the outlet.

        Keep existing outlet factors when ``overwrite`` is false, and always keep
        factors set by an outlet scaler. Propagate inlet factors for the remaining
        outlet variables, then use each component's summed inlet nominal for
        sized outlet flows.
        """
        delegate_state_block_scaling(
            self,
            (model.properties_in,),
            SolidPSDScaler,
            submodel_scalers,
            "variable_scaling_routine",
            overwrite=overwrite,
        )
        overridden = run_product_block_overrides(
            self, (model.properties_out,), submodel_scalers, overwrite=overwrite
        )
        kept = ComponentSet()
        if overridden or not overwrite:
            kept = ComponentSet(
                v
                for t in model.flowsheet().time
                for name in self._STATE_VAR_NAMES
                if hasattr(model.properties_out[t], name)
                for v in getattr(model.properties_out[t], name).values()
                if self.get_scaling_factor(v) is not None
            )
        for t in model.flowsheet().time:
            propagate_scaling_factors(
                self,
                model.properties_in[t],
                model.properties_out[t],
                self._STATE_VAR_NAMES,
                overwrite=overwrite and not overridden,
            )
            self._set_outlet_solid_factors(model, t, kept)

    @staticmethod
    def _component_nominal_totals(nominals):
        """Return each sized component's summed per-bin nominals."""
        totals = {}
        for (s, _), nom in nominals.items():
            totals[s] = totals.get(s, 0.0) + nom
        return totals

    def _set_outlet_solid_factors(self, model, t, kept):
        """Scale outlet sized flows not in ``kept`` from inlet nominals.

        Size reduction can fill bins nearly empty in the feed, so use each
        component's summed inlet nominal rather than the matching bin nominal.
        """
        feed = model.properties_in[t].flow_mass_sized_comp_size
        out = model.properties_out[t].flow_mass_sized_comp_size
        totals = self._component_nominal_totals(
            {idx: 1.0 / required_scaling_factor(self, feed[idx]) for idx in feed}
        )
        for idx in out:
            if out[idx] not in kept:
                self.set_variable_scaling_factor(
                    out[idx], 1.0 / totals[idx[0]], overwrite=True
                )

    @staticmethod
    def _fixed_feed_solid_si(model, t):
        """Return absolute fixed sized-feed flows in kg/s, or ``None`` if any
        flow is unset or unfixed.
        """
        feed = model.properties_in[t]
        out = {}
        for idx in feed.flow_mass_sized_comp_size:
            v = fixed_magnitude_or_none(
                feed.flow_mass_sized_comp_size[idx], units.kg / units.s
            )
            if v is None:
                return None
            out[idx] = abs(v)
        return out
