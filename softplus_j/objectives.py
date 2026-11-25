"""DESC Objective classes for direct J-invariance and soft-connectivity optimization."""

from __future__ import annotations

import numpy as np
from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.objectives.normalization import compute_scaling_factors
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import Timer, errorif, warnif

from .compute import (
    SOFTPLUS_SHARPNESS,
    boozer_second_adiabatic_invariant_alpha_derivative_from_data,
    boozer_soft_connectivity_penalty_from_data,
)


class SecondAdiabaticInvariantAlphaDerivative(_Objective):
    """Differentiable Boozer-space action residual dJ_tilde/dalpha (f_J).

    Evaluates the analytical field-line derivative of the softplus-regularized
    second adiabatic invariant proxy on a fixed single-field-period toroidal grid
    with log-sum-exp smooth surface extrema.
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = "r"
    _units = "(1/T)"
    _print_value_fmt = "dJ/dalpha: "
    _static_attrs = _Objective._static_attrs + [
        "_data_keys",
        "_grid",
        "_nzeta",
        "_num_alpha",
        "_soft_extrema_tau",
        "_softplus_sharpness",
        "_stop_grad_sigma_B",
        "_t",
        "_zeta_max",
        "_zeta_min",
        "M_booz",
        "N_booz",
    ]

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        M_booz=None,
        N_booz=None,
        num_alpha=48,
        t=None,
        nzeta=100,
        zeta_min=0.0,
        zeta_max=None,
        softplus_sharpness=SOFTPLUS_SHARPNESS,
        soft_extrema_tau=0.1,
        stop_grad_sigma_B=False,
        name="Second adiabatic invariant alpha derivative",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.M_booz = M_booz
        self.N_booz = N_booz
        self._num_alpha = num_alpha
        self._t = np.atleast_1d(
            np.asarray(np.linspace(0.05, 0.95, 48) if t is None else t)
        )
        self._nzeta = nzeta
        self._zeta_min = zeta_min
        self._zeta_max = zeta_max
        self._softplus_sharpness = softplus_sharpness
        self._soft_extrema_tau = soft_extrema_tau
        self._stop_grad_sigma_B = bool(stop_grad_sigma_B)

        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays and Boozer transforms."""
        eq = self.things[0]
        M_booz = self.M_booz or 2 * eq.M
        N_booz = self.N_booz or 2 * eq.N

        if self._grid is None:
            grid = LinearGrid(
                rho=1.0, M=2 * M_booz, N=2 * N_booz, NFP=eq.NFP, sym=False
            )
        else:
            grid = self._grid

        errorif(grid.sym, ValueError, f"{self.name} grid must be non-symmetric")
        warnif(
            grid.num_theta < 2 * M_booz,
            RuntimeWarning,
            f"{self.name} grid should resolve Boozer surface variation in theta",
        )
        warnif(
            grid.num_zeta < 2 * N_booz,
            RuntimeWarning,
            f"{self.name} grid should resolve Boozer surface variation in zeta",
        )

        self._data_keys = ["|B|_mn_B", "iota"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(
            self._data_keys,
            obj=eq,
            grid=grid,
            M_booz=M_booz,
            N_booz=N_booz,
        )
        self._dim_f = grid.num_rho * self._num_alpha * self._t.size
        self._constants = {
            "profiles": profiles,
            "quad_weights": jnp.ones(self._dim_f),
            "t": jnp.asarray(self._t),
            "transforms": transforms,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = 1.0 / scales["B"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute dJ_tilde/dalpha on the configured (rho, alpha, t) grid."""
        constants = self._get_deprecated_constants(constants)
        zeta_max = (
            2 * np.pi / constants["transforms"]["grid"].NFP
            if self._zeta_max is None
            else self._zeta_max
        )
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        dJ_dalpha = boozer_second_adiabatic_invariant_alpha_derivative_from_data(
            constants["transforms"]["grid"],
            constants["transforms"]["B"].basis,
            data,
            constants["t"],
            num_alpha=self._num_alpha,
            nzeta=self._nzeta,
            zeta_min=self._zeta_min,
            zeta_max=zeta_max,
            softplus_sharpness=self._softplus_sharpness,
            soft_extrema_tau=self._soft_extrema_tau,
            stop_grad_sigma_B=self._stop_grad_sigma_B,
        )
        return jnp.ravel(dJ_dalpha)


class SoftConnectivity(_Objective):
    """Soft-connectivity regularizer enforcing single-well |B| along field lines (f_SC)."""

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Soft connectivity penalty: "
    _static_attrs = _Objective._static_attrs + [
        "_data_keys",
        "_grid",
        "_num_alpha",
        "_sigmoid_sharpness",
        "_t",
        "_zeta_max",
        "_zeta_min",
        "M_booz",
        "N_booz",
    ]

    def __init__(
        self,
        eq,
        spline,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        M_booz=None,
        N_booz=None,
        num_alpha=50,
        t=None,
        zeta_min=0.0,
        zeta_max=None,
        sigmoid_sharpness=50.0,
        name="soft connectivity",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.M_booz = M_booz
        self.N_booz = N_booz
        self._num_alpha = num_alpha
        self._t = np.atleast_1d(
            np.asarray(np.linspace(0.0, 1.0, 200) if t is None else t)
        )
        self._zeta_min = zeta_min
        self._zeta_max = zeta_max
        self._sigmoid_sharpness = sigmoid_sharpness
        super().__init__(
            things=(eq, spline),
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays and Boozer transforms."""
        eq = self.things[0]
        M_booz = self.M_booz or 2 * eq.M
        N_booz = self.N_booz or 2 * eq.N

        if self._grid is None:
            grid = LinearGrid(
                rho=1.0, M=2 * M_booz, N=2 * N_booz, NFP=eq.NFP, sym=False
            )
        else:
            grid = self._grid

        errorif(grid.sym, ValueError, f"{self.name} grid must be non-symmetric")
        warnif(
            grid.num_theta < 2 * M_booz,
            RuntimeWarning,
            f"{self.name} grid should resolve Boozer surface variation in theta",
        )
        warnif(
            grid.num_zeta < 2 * N_booz,
            RuntimeWarning,
            f"{self.name} grid should resolve Boozer surface variation in zeta",
        )

        self._data_keys = ["|B|_mn_B", "iota"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(
            self._data_keys,
            obj=eq,
            grid=grid,
            M_booz=M_booz,
            N_booz=N_booz,
        )

        self._dim_f = grid.num_rho * self._num_alpha * self._t.size
        self._constants = {
            "profiles": profiles,
            "quad_weights": jnp.ones(self._dim_f),
            "t": jnp.asarray(self._t),
            "transforms": transforms,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            self._normalization = 1.0

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params, spline_params, constants=None):
        """Compute the soft-connectivity penalty on the configured grid."""
        constants = self._get_deprecated_constants(constants)
        zeta_max = (
            2 * np.pi / constants["transforms"]["grid"].NFP
            if self._zeta_max is None
            else self._zeta_max
        )
        spline_knots = self.things[1].reduced_knots(spline_params)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=eq_params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        penalty = boozer_soft_connectivity_penalty_from_data(
            constants["transforms"]["grid"],
            constants["transforms"]["B"].basis,
            data,
            num_alpha=self._num_alpha,
            t=constants["t"],
            zeta_min=self._zeta_min,
            zeta_max=zeta_max,
            reduced_alpha_knots=spline_knots["reduced_alpha_knots"],
            zeta_min_knots=spline_knots["zeta_min_knots"],
            zeta_max_knots=spline_knots["zeta_max_knots"],
            sigmoid_sharpness=self._sigmoid_sharpness,
            spline_symmetry=self.things[1].symmetry,
        )
        return jnp.ravel(penalty)
