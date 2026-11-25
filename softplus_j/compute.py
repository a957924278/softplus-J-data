"""Core JAX/DESC compute functions for differentiable J-invariance and soft connectivity.

Implements Equations (9)--(30) of:
    H. Chen et al., "Direct Optimization of Stellarator Omnigenity from the
    Second Adiabatic Invariant" (Nuclear Fusion, NF-109458).
"""

from __future__ import annotations

import jax
import numpy as np
from desc.backend import jnp, trapezoid, vmap
from interpax import interp1d

SOFTPLUS_SHARPNESS = 100.0


def _softplus_relu(x, beta=SOFTPLUS_SHARPNESS):
    """Smooth positive-part approximation of ``max(x, 0)``: Eq. (9)."""
    return jnp.logaddexp(beta * x, 0.0) / beta


def _softplus_relu_sigmoid(x, beta=SOFTPLUS_SHARPNESS):
    """Derivative of ``_softplus_relu``: ``sigmoid(beta * x)``."""
    return 1.0 / (1.0 + jnp.exp(-beta * x))


def _smoothmax_logsumexp(x, axis, tau):
    """Differentiable log-sum-exp smooth maximum: Eqs. (17)--(18)."""
    tau = jnp.asarray(tau, dtype=x.dtype)
    tau = jnp.maximum(tau, jnp.finfo(x.dtype).eps)
    x_scaled = x / tau
    x_max = jnp.max(x_scaled, axis=axis, keepdims=True)
    lse = x_max + jnp.log(jnp.sum(jnp.exp(x_scaled - x_max), axis=axis, keepdims=True))
    return tau * lse


def _reshape_surface_coefficients(grid, values):
    """Reshape flattened per-surface Boozer Fourier coefficients."""
    return jnp.asarray(values).reshape((grid.num_rho, -1))


def _boozer_B_star_from_t(B_min, B_max, t):
    """Map normalized pitch samples ``t`` in (0, 1) to turning-point field B*."""
    B_min = jnp.atleast_1d(jnp.asarray(B_min))
    B_max = jnp.atleast_1d(jnp.asarray(B_max))
    t = jnp.atleast_1d(jnp.asarray(t))
    return (1.0 - t[None, :]) * B_min[:, None] + t[None, :] * B_max[:, None]


def _boozer_second_adiabatic_surface_proxy(
    basis,
    rho,
    coeff_B,
    coeff_g,
    iota,
    alpha,
    B_star,
    zeta_min,
    zeta_max,
    nzeta,
    softplus_sharpness,
):
    """Fixed-domain softplus-regularized action proxy J_tilde on one surface."""
    zeta = jnp.linspace(zeta_min, zeta_max, nzeta)
    theta = alpha[:, None] + iota * (zeta[None, :] - zeta_min)
    rho2d = jnp.broadcast_to(rho, theta.shape)
    zeta2d = jnp.broadcast_to(zeta[None, :], theta.shape)
    nodes = jnp.stack((rho2d, theta, zeta2d), axis=-1).reshape((-1, 3))

    mat = basis.evaluate(nodes)
    B = (mat @ coeff_B).reshape((alpha.size, nzeta))
    B_star = jnp.atleast_1d(B_star)
    arg = 1.0 - B[None] / B_star[:, None, None]
    cutoff = _softplus_relu(arg, beta=softplus_sharpness)
    sqrt_c = jnp.sqrt(jnp.maximum(cutoff, 1e-30))
    if coeff_g is not None:
        g = (mat @ coeff_g).reshape((alpha.size, nzeta))
        integrand = sqrt_c * jnp.abs(B[None]) * jnp.abs(g[None])
    else:
        integrand = sqrt_c / B[None]
    return trapezoid(integrand, zeta, axis=-1).transpose(1, 0)


def boozer_second_adiabatic_invariant_from_data(
    grid,
    basis,
    data,
    alpha,
    B_star,
    *,
    nzeta=800,
    zeta_min=0.0,
    zeta_max=None,
    softplus_sharpness=SOFTPLUS_SHARPNESS,
):
    """Evaluate smoothed action proxy J_tilde(rho, alpha, t) from Boozer data."""
    rho = jnp.asarray(grid.compress(grid.nodes[:, 0]))
    iota = jnp.asarray(grid.compress(data["iota"]))
    coeff_B = _reshape_surface_coefficients(grid, data["|B|_mn_B"])
    coeff_g = (
        _reshape_surface_coefficients(grid, data["sqrt(g)_Boozer_mn"])
        if "sqrt(g)_Boozer_mn" in data
        else None
    )
    alpha = jnp.asarray(alpha)
    if alpha.ndim == 1:
        alpha = jnp.broadcast_to(alpha[None, :], (rho.size, alpha.size))
    B_star = jnp.asarray(B_star)
    if B_star.ndim == 1:
        B_star = jnp.broadcast_to(B_star[None, :], (rho.size, B_star.size))
    zeta_max = 2 * jnp.pi / grid.NFP if zeta_max is None else zeta_max

    if coeff_g is not None:
        return vmap(
            lambda r, it, cB, cg, pi, a: _boozer_second_adiabatic_surface_proxy(
                basis, r, cB, cg, it, a, pi, zeta_min, zeta_max, nzeta, softplus_sharpness
            )
        )(rho, iota, coeff_B, coeff_g, B_star, alpha)
    return vmap(
        lambda r, it, cB, pi, a: _boozer_second_adiabatic_surface_proxy(
            basis, r, cB, None, it, a, pi, zeta_min, zeta_max, nzeta, softplus_sharpness
        )
    )(rho, iota, coeff_B, B_star, alpha)


def _boozer_second_adiabatic_surface_alpha_deriv(
    basis,
    rho,
    coeff_B,
    iota,
    alpha,
    B_star,
    zeta_min,
    zeta_max,
    nzeta,
    softplus_sharpness,
):
    """Analytical dJ_tilde/dalpha on one flux surface: Eqs. (15)--(16)."""
    zeta = jnp.linspace(zeta_min, zeta_max, nzeta)
    theta = alpha[:, None] + iota * (zeta[None, :] - zeta_min)
    rho2d = jnp.broadcast_to(rho, theta.shape)
    zeta2d = jnp.broadcast_to(zeta[None, :], theta.shape)
    nodes = jnp.stack((rho2d, theta, zeta2d), axis=-1).reshape((-1, 3))

    mat = basis.evaluate(nodes)
    mat_dt = basis.evaluate(nodes, derivatives=np.array([0, 1, 0]))

    B = (mat @ coeff_B).reshape((alpha.size, nzeta))
    dB_dt = (mat_dt @ coeff_B).reshape((alpha.size, nzeta))

    B_star = jnp.atleast_1d(B_star)
    arg = 1.0 - B[None] / B_star[:, None, None]
    cutoff = _softplus_relu(arg, beta=softplus_sharpness)
    sig = _softplus_relu_sigmoid(arg, beta=softplus_sharpness)
    sqrt_c = jnp.sqrt(jnp.maximum(cutoff, 1e-30))

    safe_sqrt_c = jnp.where(cutoff > 0, sqrt_c, 1e-30)
    # Exact analytical df/dB for f(B; B*) = sqrt(sp_eps(1 - B/B*)) / B
    df_dB = jnp.where(
        cutoff > 0,
        -sig / (2.0 * B_star[:, None, None] * safe_sqrt_c * B[None])
        - safe_sqrt_c / (B[None] ** 2),
        0.0,
    )

    integrand = df_dB * dB_dt[None]
    return trapezoid(integrand, zeta, axis=-1).transpose(1, 0)


def boozer_second_adiabatic_invariant_alpha_derivative_analytical(
    basis,
    rho,
    iota,
    coeff_B,
    alpha,
    B_star,
    *,
    nzeta=1000,
    zeta_min=0.0,
    zeta_max=None,
    nfp=1,
    softplus_sharpness=SOFTPLUS_SHARPNESS,
):
    """Analytical dJ_tilde/dalpha across flux surfaces via chain-rule integration."""
    alpha = jnp.asarray(alpha)
    B_star = jnp.asarray(B_star)
    if B_star.ndim == 0:
        B_star = B_star[None, None]
    elif B_star.ndim == 1:
        B_star = B_star[None, :]
    B_star = jnp.broadcast_to(B_star, (rho.size, B_star.shape[-1]))
    zeta_max = 2 * jnp.pi / nfp if zeta_max is None else zeta_max

    return vmap(
        lambda r, it, cB, pi, a: _boozer_second_adiabatic_surface_alpha_deriv(
            basis, r, cB, it, a, pi, zeta_min, zeta_max, nzeta, softplus_sharpness
        )
    )(rho, iota, coeff_B, B_star, alpha)


def boozer_second_adiabatic_invariant_alpha_derivative_from_data(
    grid,
    basis,
    data,
    t,
    *,
    num_alpha=48,
    nzeta=1000,
    zeta_min=0.0,
    zeta_max=None,
    softplus_sharpness=SOFTPLUS_SHARPNESS,
    soft_extrema_tau=0.1,
    stop_grad_sigma_B=False,
):
    """Convenience wrapper for dJ_tilde/dalpha using per-surface Boozer data."""
    rho = jnp.asarray(grid.compress(grid.nodes[:, 0]))
    iota = jnp.asarray(grid.compress(data["iota"]))
    coeff_B = _reshape_surface_coefficients(grid, data["|B|_mn_B"])

    nfp = grid.NFP
    alpha0 = jnp.pi - iota * (jnp.pi / nfp)
    alpha = jnp.linspace(-jnp.pi, 0.0, num_alpha, endpoint=False) + alpha0[:, None]

    A = basis.evaluate(grid.nodes[:: grid.num_rho])
    B_grid = coeff_B @ A.T
    num_eval = B_grid.shape[1]
    log_n = jnp.log(jnp.maximum(jnp.asarray(num_eval, dtype=B_grid.dtype), 2.0))
    B_range = jnp.max(B_grid, axis=1) - jnp.min(B_grid, axis=1)
    B_range = jnp.maximum(B_range, 1e-30)
    tau_eff = (soft_extrema_tau * B_range / log_n)[:, None]
    if stop_grad_sigma_B:
        tau_eff = jax.lax.stop_gradient(tau_eff)
    B_max = _smoothmax_logsumexp(B_grid, axis=1, tau=tau_eff).squeeze(-1)
    B_min = -_smoothmax_logsumexp(-B_grid, axis=1, tau=tau_eff).squeeze(-1)
    B_star = _boozer_B_star_from_t(B_min, B_max, t)[0]
    return boozer_second_adiabatic_invariant_alpha_derivative_analytical(
        basis,
        rho,
        iota,
        coeff_B,
        alpha,
        B_star,
        nzeta=nzeta,
        zeta_min=zeta_min,
        zeta_max=zeta_max,
        nfp=nfp,
        softplus_sharpness=softplus_sharpness,
    )


def _boozer_soft_connectivity_surface(
    basis,
    rho,
    coeff_B,
    iota,
    alpha,
    zeta_min_bound,
    zeta_max_bound,
    t,
    reduced_alpha_knots,
    zeta_min_knots,
    zeta_max_knots,
    sigmoid_sharpness,
    spline_symmetry,
):
    """Structured soft-connectivity penalty on one flux surface: Eqs. (24)--(29)."""
    zeta_span = zeta_max_bound - zeta_min_bound
    alpha_next = alpha + iota * zeta_span

    if reduced_alpha_knots is not None:
        reduced_alpha_knots = jnp.asarray(reduced_alpha_knots)
        if zeta_min_knots is not None:
            zeta_min_knots = jnp.clip(jnp.asarray(zeta_min_knots), 0.0, zeta_span)
            if spline_symmetry:
                alpha_min_knots = jnp.concatenate(
                    [
                        reduced_alpha_knots,
                        jnp.mod(
                            2 * jnp.pi - reduced_alpha_knots - iota * zeta_span,
                            2 * jnp.pi,
                        ),
                    ]
                )
                zeta_min_knots = jnp.concatenate(
                    [zeta_min_knots, zeta_span - zeta_min_knots]
                )
            else:
                alpha_min_knots = reduced_alpha_knots
            min_order = jnp.argsort(alpha_min_knots)
            alpha_min_knots = alpha_min_knots[min_order]
            zeta_min_knots = zeta_min_knots[min_order]
        if zeta_max_knots is not None:
            zeta_max_knots = (
                jnp.mod(jnp.asarray(zeta_max_knots) + 0.5 * zeta_span, zeta_span)
                - 0.5 * zeta_span
            )
            if spline_symmetry:
                alpha_max_knots = jnp.concatenate(
                    [
                        reduced_alpha_knots,
                        jnp.mod(2 * jnp.pi - reduced_alpha_knots, 2 * jnp.pi),
                    ]
                )
                zeta_max_knots = jnp.concatenate(
                    [zeta_max_knots, -zeta_max_knots]
                )
            else:
                alpha_max_knots = reduced_alpha_knots
            max_order = jnp.argsort(alpha_max_knots)
            alpha_max_knots = alpha_max_knots[max_order]
            zeta_max_knots = zeta_max_knots[max_order]

    if zeta_max_knots is not None:
        zeta_max_alpha = interp1d(
            alpha, alpha_max_knots, zeta_max_knots, method="cubic", period=2 * jnp.pi
        )
        zeta_max_next = (
            interp1d(
                alpha_next,
                alpha_max_knots,
                zeta_max_knots,
                method="cubic",
                period=2 * jnp.pi,
            )
            + zeta_span
        )
    else:
        zeta_max_alpha = jnp.full(alpha.shape, zeta_min_bound)
        zeta_max_next = jnp.full(alpha.shape, zeta_max_bound)

    zeta = (1.0 - t[None, :]) * zeta_max_alpha[:, None] + (
        t[None, :] * zeta_max_next[:, None]
    )
    theta = alpha[:, None] + iota * zeta
    rho2d = jnp.broadcast_to(rho, theta.shape)
    nodes = jnp.stack((rho2d, theta, zeta), axis=-1).reshape((-1, 3))
    nt = t.size

    dB_dtheta = (
        basis.evaluate(nodes, derivatives=np.array([0, 1, 0])) @ coeff_B
    ).reshape((alpha.size, nt))
    dB_dzeta = (
        basis.evaluate(nodes, derivatives=np.array([0, 0, 1])) @ coeff_B
    ).reshape((alpha.size, nt))
    dB_dz_line = iota * dB_dtheta + dB_dzeta

    if zeta_min_knots is not None:
        zeta_min_alpha = interp1d(
            alpha, alpha_min_knots, zeta_min_knots, method="cubic", period=2 * jnp.pi
        )
        zeta_min_next = (
            interp1d(
                alpha_next,
                alpha_min_knots,
                zeta_min_knots,
                method="cubic",
                period=2 * jnp.pi,
            )
            + zeta_span
        )
    else:
        zeta_min_alpha = jnp.full(alpha.shape, zeta_min_bound + 0.5 * zeta_span)
        zeta_min_next = zeta_min_alpha + zeta_span

    use_current_min = (zeta_min_alpha > zeta_max_alpha) & (
        zeta_min_alpha < zeta_max_next
    )
    zeta_min_shifted = jnp.where(use_current_min, zeta_min_alpha, zeta_min_next)
    delta = zeta - zeta_min_shifted[:, None]
    sig = jax.nn.sigmoid(sigmoid_sharpness * delta)
    penalty_left = _softplus_relu(dB_dz_line)
    penalty_right = _softplus_relu(-dB_dz_line)
    return sig * penalty_right + (1.0 - sig) * penalty_left


def boozer_soft_connectivity_penalty(
    basis,
    rho,
    iota,
    coeff_B,
    alpha,
    *,
    nzeta=200,
    t=None,
    zeta_min=0.0,
    zeta_max=None,
    nfp=1,
    reduced_alpha_knots=None,
    zeta_min_knots=None,
    zeta_max_knots=None,
    sigmoid_sharpness=50.0,
    spline_symmetry=True,
):
    """Compute the structured soft-connectivity penalty across flux surfaces."""
    has_knots = zeta_min_knots is not None or zeta_max_knots is not None
    if has_knots and reduced_alpha_knots is None:
        raise ValueError(
            "reduced_alpha_knots must be supplied with zeta_min/max spline knots."
        )
    alpha = jnp.asarray(alpha)
    if alpha.ndim == 1:
        alpha = jnp.broadcast_to(alpha[None, :], (rho.size, alpha.size))
    zeta_max = 2 * jnp.pi / nfp if zeta_max is None else zeta_max
    t = jnp.linspace(0.0, 1.0, nzeta) if t is None else jnp.atleast_1d(jnp.asarray(t))

    return vmap(
        lambda r, it, cB, a: _boozer_soft_connectivity_surface(
            basis,
            r,
            cB,
            it,
            a,
            zeta_min,
            zeta_max,
            t,
            reduced_alpha_knots,
            zeta_min_knots,
            zeta_max_knots,
            sigmoid_sharpness,
            spline_symmetry,
        )
    )(rho, iota, coeff_B, alpha)


def boozer_soft_connectivity_penalty_from_data(
    grid,
    basis,
    data,
    *,
    alpha=None,
    num_alpha=50,
    nzeta=200,
    t=None,
    zeta_min=0.0,
    zeta_max=None,
    reduced_alpha_knots=None,
    zeta_min_knots=None,
    zeta_max_knots=None,
    sigmoid_sharpness=50.0,
    spline_symmetry=True,
):
    """Convenience wrapper for soft connectivity using per-surface Boozer data."""
    rho = jnp.asarray(grid.compress(grid.nodes[:, 0]))
    iota = jnp.asarray(grid.compress(data["iota"]))
    coeff_B = _reshape_surface_coefficients(grid, data["|B|_mn_B"])

    nfp = grid.NFP
    if alpha is None:
        alpha0 = jnp.pi - iota * (jnp.pi / nfp)
        alpha_span = jnp.pi if spline_symmetry else 2 * jnp.pi
        alpha = (
            jnp.linspace(-alpha_span, 0.0, num_alpha, endpoint=False) + alpha0[:, None]
        )

    return boozer_soft_connectivity_penalty(
        basis,
        rho,
        iota,
        coeff_B,
        alpha,
        nzeta=nzeta,
        t=t,
        zeta_min=zeta_min,
        zeta_max=zeta_max,
        nfp=nfp,
        reduced_alpha_knots=reduced_alpha_knots,
        zeta_min_knots=zeta_min_knots,
        zeta_max_knots=zeta_max_knots,
        sigmoid_sharpness=sigmoid_sharpness,
        spline_symmetry=spline_symmetry,
    )
