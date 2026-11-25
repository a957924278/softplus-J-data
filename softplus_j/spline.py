"""Optimizable periodic cubic-spline parameterization for soft connectivity."""

from __future__ import annotations

import numpy as np
from desc.backend import jnp
from desc.io import IOAble
from desc.optimizable import Optimizable, optimizable_parameter
from desc.utils import errorif


class SplineZetaMin(Optimizable, IOAble):
    """Periodic cubic-spline control knots defining zeta_min(alpha) and zeta_max(alpha).

    With ``symmetry=True``, the free control knots span ``(0, pi)`` and are mirrored
    under stellarator symmetry in the soft-connectivity compute layer. With
    ``symmetry=False``, the free knots span the full ``[0, 2*pi)`` directly.
    """

    _io_attrs_ = [
        "_NFP",
        "_n_control",
        "_alpha_knots",
        "_zeta_min_knots",
        "_zeta_max_knots",
        "_symmetry",
    ]
    _static_attrs = Optimizable._static_attrs + [
        "_NFP",
        "_n_control",
        "_alpha_knots",
        "_symmetry",
    ]

    def __init__(
        self,
        n_control=8,
        NFP=1,
        alpha_knots=None,
        zeta_min_knots=None,
        zeta_max_knots=None,
        symmetry=True,
    ):
        self._NFP = int(NFP)
        self._n_control = int(n_control)
        self._symmetry = bool(symmetry)
        if alpha_knots is None:
            period = np.pi if self._symmetry else 2 * np.pi
            alpha_knots = (np.arange(n_control) + 0.5) * period / n_control
        self._alpha_knots = np.asarray(alpha_knots, dtype=float)
        errorif(
            self._alpha_knots.size != self._n_control,
            ValueError,
            "alpha_knots must have length n_control.",
        )
        if zeta_min_knots is None:
            zeta_min_knots = np.full(n_control, np.pi / NFP)
        zeta_min_knots = np.asarray(zeta_min_knots, dtype=float)
        errorif(
            zeta_min_knots.size != self._n_control,
            ValueError,
            "zeta_min_knots must have length n_control.",
        )
        self._zeta_min_knots = zeta_min_knots
        if zeta_max_knots is None:
            zeta_max_knots = np.zeros(n_control)
        zeta_max_knots = np.asarray(zeta_max_knots, dtype=float)
        errorif(
            zeta_max_knots.size != self._n_control,
            ValueError,
            "zeta_max_knots must have length n_control.",
        )
        self._zeta_max_knots = self._wrap_zeta_max_near_zero(zeta_max_knots)

    def _set_up(self):
        """Validate serialized SplineZetaMin state after loading."""
        self._symmetry = bool(self._symmetry)
        self._NFP = int(self._NFP)
        self._n_control = int(self._n_control)
        self._alpha_knots = np.asarray(self._alpha_knots, dtype=float)
        self._zeta_min_knots = np.asarray(self._zeta_min_knots, dtype=float)
        self._zeta_max_knots = self._wrap_zeta_max_near_zero(self._zeta_max_knots)

    def _wrap_zeta_max_near_zero(self, val):
        """Wrap zeta_max values to [-pi/NFP, pi/NFP)."""
        span = 2 * np.pi / self._NFP
        return np.mod(np.asarray(val, dtype=float) + span / 2, span) - span / 2

    def reduced_knots(self, params=None):
        """Return free knot dictionary consumed by the soft-connectivity compute layer."""
        if params is None:
            params = self.params_dict
        return {
            "reduced_alpha_knots": jnp.asarray(self._alpha_knots),
            "zeta_min_knots": jnp.clip(
                jnp.asarray(params["zeta_min_knots"]), 0.0, 2 * jnp.pi / self._NFP
            ),
            "zeta_max_knots": jnp.asarray(params["zeta_max_knots"]),
        }

    @property
    def symmetry(self):
        """Whether stellarator symmetry is assumed for the knot layout."""
        return self._symmetry

    @optimizable_parameter
    @property
    def zeta_min_knots(self):
        """Low-field minimum toroidal angle knots."""
        return self._zeta_min_knots

    @zeta_min_knots.setter
    def zeta_min_knots(self, val):
        val = np.asarray(val, dtype=float)
        errorif(
            val.size != self._n_control,
            ValueError,
            "zeta_min_knots must have length n_control.",
        )
        self._zeta_min_knots = val

    @optimizable_parameter
    @property
    def zeta_max_knots(self):
        """High-field maximum toroidal angle knots."""
        return self._zeta_max_knots

    @zeta_max_knots.setter
    def zeta_max_knots(self, val):
        val = np.asarray(val, dtype=float)
        errorif(
            val.size != self._n_control,
            ValueError,
            "zeta_max_knots must have length n_control.",
        )
        self._zeta_max_knots = self._wrap_zeta_max_near_zero(val)

    @property
    def alpha_knots_full(self):
        """Full alpha knots spanning [0, 2*pi)."""
        if not self._symmetry:
            return self._alpha_knots
        return np.concatenate([self._alpha_knots, self._alpha_knots + np.pi])
