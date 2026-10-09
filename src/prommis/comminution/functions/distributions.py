#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Assumed-shape cumulative distribution forms and mesh discretization.

Supported shapes are Rosin-Rammler (``Q(d) = 1 - exp(-(d/d_63)^n)``) and
Gates-Gaudin-Schuhmann (``Q(d) = (min(d, d_max)/d_max)^n``), with a smooth
approximation to the GGS cap. Per-interval retained fractions come from
``core.psd_math.fractions_from_cdf``. Material below the lowest mesh edge
is included in bin 0, and the fractions sum to 1.

When ``distribution_fractions`` receives a numeric characteristic size, it requires
Rosin-Rammler ``Q(x_N) >= RR_VALIDITY_THRESHOLD`` (0.95) or GGS
``d_max <= x_N``. For a Pyomo characteristic size, the unit model checks its
evaluated value numerically.

References:

[1] Rosin, P., & Rammler, E. (1933). The laws governing the fineness of
    powdered coal. J. Inst. Fuel, 7, 29-36.

[2] Schuhmann, R., Jr. (1940). Principles of comminution, I: size
    distribution and surface calculations. AIME Technical Publication
    No. 1189.
"""

__author__ = "Daison Yancy Caballero"

import math
import numbers
from collections.abc import Callable
from typing import NamedTuple

from pyomo.environ import exp

from idaes.core.util.math import smooth_max, smooth_min

from prommis.comminution.core.config_utils import _is_real_number
from prommis.comminution.core.psd_math import P80_PASSING, fractions_from_cdf
from prommis.comminution.core.smooth_functions import EPS_SMOOTH

# Minimum Rosin-Rammler passing at the top mesh edge.
RR_VALIDITY_THRESHOLD = 0.95

# Safety limit: raise if the CDF check keeps failing.
_MAX_ROUNDING_STEPS = 100

# Fixed smoothing of the outer ``smooth_max`` that keeps the GGS base nonnegative
# before a non-integer power.
_GGS_BASE_FLOOR = 1e-9


def rosin_rammler_cdf(d, d63, n):
    """Rosin-Rammler cumulative passing ``1 - exp(-(d/d63)^n)``.

    See Rosin and Rammler (1933).
    """
    return 1 - exp(-((d / d63) ** n))


def ggs_cdf(d, d_max, n, eps=EPS_SMOOTH):
    """Gates-Gaudin-Schuhmann passing with a smooth cap near ``d_max``.

    The ideal law is ``Q(d) = (min(d, d_max)/d_max)^n`` (Schuhmann, 1940).
    ``smooth_min`` approximates the upper cap on ``d/d_max`` using ``eps``.
    The outer ``smooth_max`` uses ``_GGS_BASE_FLOOR`` (1e-9) to keep the
    base nonnegative before a non-integer power, even where ``smooth_min``
    undershoots zero on ultra-fine edges. The cap is approximate, so
    ``Q(d_max)`` can be slightly below one.
    """
    ratio = d / d_max
    clamped = smooth_max(0.0, smooth_min(1.0, ratio, eps), _GGS_BASE_FLOOR)
    return clamped**n


def _rr_upper_bound(x_top, n):
    """Return the top-edge limit for Rosin-Rammler ``d_63``, adjusted for rounding."""
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError("shape exponent n must be a finite positive number")
    upper = x_top / ((-math.log(1.0 - RR_VALIDITY_THRESHOLD)) ** (1.0 / n))
    for _ in range(_MAX_ROUNDING_STEPS):
        if not (
            math.isfinite(upper)
            and upper > 0.0
            and rosin_rammler_cdf(x_top, upper, n) < RR_VALIDITY_THRESHOLD
        ):
            return upper
        upper = math.nextafter(upper, 0.0)
    raise ValueError(
        f"Rosin-Rammler d_63 upper bound for mesh top {x_top!r} and exponent "
        f"{n!r} still fails the {RR_VALIDITY_THRESHOLD} passing threshold after "
        f"{_MAX_ROUNDING_STEPS} rounding steps."
    )


def _rr_check_top_edge(top, shape_size, n, threshold):
    q_top = rosin_rammler_cdf(top, shape_size, n)
    if q_top < threshold:
        raise ValueError(
            f"Rosin-Rammler passing at the top edge is Q(x_N) = {q_top!r}, "
            f"below the required {threshold}; d_63 is too large for this mesh."
        )


def _ggs_check_top_edge(top, shape_size, n, threshold):
    if shape_size > top:
        raise ValueError(
            "Gates-Gaudin-Schuhmann d_max must be <= the top mesh edge "
            f"(d_max = {shape_size:.6g} > x_N = {top:.6g})."
        )


def _form(shape):
    """Return the ``ShapeForm`` of ``shape``."""
    try:
        return SHAPES[shape]
    except KeyError:
        raise ValueError(f"unknown distribution shape {shape!r}") from None


def p80_ratio(shape, n):
    """Return P80 divided by the shape's characteristic size for exponent ``n``."""
    return _form(shape).p80_ratio(n)


def shape_size_from_p80(shape, p80, n):
    """Return the characteristic size whose analytic P80 is ``p80``."""
    return p80 / p80_ratio(shape, n)


def shape_size_upper_bound(shape, x_top, n):
    """Return the mesh-based upper limit for a distribution's characteristic size.

    For Rosin-Rammler, the limit must meet ``RR_VALIDITY_THRESHOLD`` (0.95)
    at the top edge. If rounding puts passing just below that threshold,
    move the limit slightly lower.
    For GGS, return the top mesh edge.

    Raises:
        ValueError: If ``shape`` is unknown, the Rosin-Rammler exponent is
            not finite and positive, or the limit still fails the passing
            check after the retry limit.
    """
    return _form(shape).upper_bound(x_top, n)


def shape_cdf(shape, d, shape_size, n, eps=EPS_SMOOTH):
    """Return the cumulative passing of ``shape`` at size ``d``.

    Accept floats or Pyomo expressions; ``eps`` smooths the GGS cap only.
    """
    return _form(shape).cdf(d, shape_size, n, eps)


def distribution_fractions(shape, edges, shape_size, n, eps=EPS_SMOOTH):
    """Discretize an assumed-shape cumulative form onto a mesh.

    Args:
        shape: ``"rosin_rammler"`` or ``"gates_gaudin_schuhmann"``.
        edges: ``N+1`` ascending numeric size edges, in the characteristic
            size's length unit.
        shape_size: shape characteristic size (``d_63`` for RR, ``d_max`` for GGS);
            a real number (Python or NumPy int or float) on the pure-function
            path, a Pyomo expression inside a unit.
        n: positive finite real shape exponent.
        eps: smoothing parameter for the GGS cap.

    Returns:
        list of ``N`` retained fractions (floats or Pyomo expressions) that
        sum to 1.

    Raises:
        ValueError: for an unknown shape, a non-positive/non-finite numeric
            characteristic size or exponent, or failure of the numeric
            shape-validity rule
            (:func:`assert_shape_valid`).
    """
    if isinstance(shape_size, numbers.Real):
        assert_shape_valid(shape, edges, shape_size, n)
    elif not (_is_real_number(n) and math.isfinite(n) and n > 0.0):
        # A Pyomo characteristic size is not evaluated here; check only the exponent.
        raise ValueError("shape exponent n must be a finite positive number")
    form = _form(shape)
    return fractions_from_cdf(lambda x: form.cdf(x, shape_size, n, eps), edges)


def assert_shape_valid(shape, edges, shape_size, n, threshold=RR_VALIDITY_THRESHOLD):
    """Validate a numeric characteristic size against the top mesh edge.

    Rosin-Rammler requires ``Q(x_N) >= threshold``; GGS requires
    ``d_max <= x_N``.

    Args:
        threshold: minimum Rosin-Rammler passing at the top edge (default
            ``RR_VALIDITY_THRESHOLD``, 0.95).

    Raises:
        ValueError: if the shape is unknown, the characteristic size or exponent fails a
            numeric check, or a top-edge rule is violated.
    """
    top = float(edges[-1])
    if isinstance(shape_size, bool) or not (
        math.isfinite(float(shape_size)) and float(shape_size) > 0.0
    ):
        raise ValueError(
            "characteristic size (d_63/d_max) must be a finite positive number"
        )
    if not (_is_real_number(n) and math.isfinite(n) and n > 0.0):
        raise ValueError("shape exponent n must be a finite positive number")
    _form(shape).check_top_edge(top, float(shape_size), n, threshold)


class ShapeForm(NamedTuple):
    """Calculations and characteristic-size labels for one PSD shape."""

    shape_size_name: str  # characteristic-size name: d_63 or d_max
    shape_size_doc: str  # description used for the characteristic-size Var
    cdf: Callable  # (d, characteristic_size, n, eps) -> cumulative passing
    p80_ratio: Callable  # (n) -> P80 / characteristic size
    upper_bound: Callable  # (x_top, n) -> mesh-based characteristic-size limit
    check_top_edge: Callable  # (top, size, n, threshold) -> raises if invalid


SHAPES = {
    "rosin_rammler": ShapeForm(
        shape_size_name="d_63",
        shape_size_doc="Rosin-Rammler size at 63.2% passing",
        cdf=lambda d, shape_size, n, eps: rosin_rammler_cdf(d, shape_size, n),
        p80_ratio=lambda n: (-math.log(0.2)) ** (1.0 / n),
        upper_bound=_rr_upper_bound,
        check_top_edge=_rr_check_top_edge,
    ),
    "gates_gaudin_schuhmann": ShapeForm(
        shape_size_name="d_max",
        shape_size_doc="Gates-Gaudin-Schuhmann maximum size",
        cdf=ggs_cdf,
        p80_ratio=lambda n: P80_PASSING ** (1.0 / n),
        upper_bound=lambda x_top, n: float(x_top),
        check_top_edge=_ggs_check_top_edge,
    ),
}

SHAPE_SIZE_NAME = {name: form.shape_size_name for name, form in SHAPES.items()}
