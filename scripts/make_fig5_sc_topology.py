"""Create fig_sc_cheat: SC-off static cheating evidence.

Two panels:
  (a) Multi-well splitting map on the SC-off configuration:
      (α,t) regions where B<B* splits into ≥2 disconnected intervals.
  (b) Hard-J σ_J(t) vs soft-proxy σ_J(t) for the SC-off config,
      demonstrating that the soft proxy under-reports α-variation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from desc.backend import jnp

from fig_common import (
    COLORS,
    boozer_spectral_data,
    count_periodic_true_components,
    data_path,
    figure_path,
    load_equilibrium,
    save_matplotlib,
    setup_matplotlib,
    smooth_J_map,
)

DEFAULT_EQ = data_path("equilibria", "sc_off_beta0.00.h5")
DEFAULT_OUTPUT = figure_path("fig5_sc_topology.pdf")

from softplus_j import _boozer_B_star_from_t

try:
    from jax.numpy import trapezoid as _trapezoid
except ImportError:
    from jax.numpy import trapz as _trapezoid


def trapezoid_(y, x, axis=-1):
    return _trapezoid(y, x, axis=axis)


def expand_about_alpha0(values):
    """Reflect DESC's sampled half-period about the symmetry axis alpha0."""
    num_alpha = values.shape[0]
    delta_half = np.linspace(-np.pi, 0.0, num_alpha, endpoint=False)
    delta_full = np.concatenate((delta_half, -delta_half[::-1]))
    values_full = np.concatenate((values, values[::-1]), axis=0)
    order = np.argsort(delta_full)
    return delta_full[order], values_full[order]


def compute_well_split_map(eq, num_alpha=96, nt=61, nzeta=2000):
    """Return the alpha0-centered map where B<B* splits into multiple wells."""
    nfp = eq.NFP
    bd = boozer_spectral_data(eq, rho=1.0)
    iota = float(bd["iota"][0])
    basis = bd["basis"]
    coeff_B = np.asarray(bd["coeff_B"])

    zeta = np.linspace(0.0, 2 * np.pi / nfp, nzeta)
    t = np.linspace(0.01, 0.99, nt)

    a0 = np.pi - iota * (np.pi / nfp)
    alpha = np.linspace(-np.pi, 0.0, num_alpha, endpoint=False) + a0
    nodes = np.stack(
        np.broadcast_arrays(1.0, alpha[:, None] + iota * zeta[None, :], zeta[None, :]),
        axis=-1,
    ).reshape(-1, 3)
    B = (np.asarray(basis.evaluate(nodes)) @ coeff_B[0]).reshape(num_alpha, nzeta)

    B_min = float(B.min())
    B_max = float(B.max())
    B_star = np.asarray(
        _boozer_B_star_from_t(jnp.array(B_min), jnp.array(B_max), jnp.array(t))
    )

    split = np.zeros((num_alpha, nt), dtype=bool)
    for ia in range(num_alpha):
        for it in range(nt):
            mask = B[ia] < float(B_star[0, it])
            if count_periodic_true_components(mask) >= 2:
                split[ia, it] = True

    alpha_delta, split = expand_about_alpha0(split)
    return alpha_delta, t, split


def compute_hard_J_sigma(eq, num_alpha=96, nt=61, nzeta=2000):
    """Hard-J σ_J(t). (Duplicated minimal version)."""
    from jax import vmap

    nfp = eq.NFP
    bd = boozer_spectral_data(eq, rho=1.0)
    basis = bd["basis"]
    grid_b = bd["grid"]
    rho = jnp.asarray(bd["rho"])
    iota = jnp.asarray(bd["iota"])
    coeff_B = jnp.asarray(bd["coeff_B"])
    coeff_g = jnp.asarray(bd["coeff_g"])

    zeta_max = 2 * np.pi / nfp
    zeta = jnp.linspace(0.0, zeta_max, nzeta)
    t = np.linspace(0.01, 0.99, nt)

    A_full = basis.evaluate(grid_b.nodes[:: grid_b.num_rho])
    B_full = coeff_B @ A_full.T
    B_max_surf = jnp.max(B_full, axis=1)
    B_min_surf = jnp.min(B_full, axis=1)
    B_star = _boozer_B_star_from_t(B_min_surf, B_max_surf, jnp.asarray(t))

    def one_surface(r_it, cB, cg, B_star_s):
        r_, it_ = r_it
        a0 = jnp.pi - it_ * (jnp.pi / nfp)
        alpha_1d = jnp.linspace(-jnp.pi, 0.0, num_alpha, endpoint=False) + a0
        theta = alpha_1d[:, None] + it_ * zeta[None, :]
        rho2d = jnp.broadcast_to(r_, theta.shape)
        zeta2d = jnp.broadcast_to(zeta[None, :], theta.shape)
        nodes = jnp.stack((rho2d, theta, zeta2d), axis=-1).reshape((-1, 3))
        mat = basis.evaluate(nodes)
        B = (mat @ cB).reshape((num_alpha, nzeta))
        g = (mat @ cg).reshape((num_alpha, nzeta))
        arg = 1.0 - B[None, :, :] / B_star_s[:, None, None]
        cutoff = jnp.maximum(arg, 0.0)
        integrand = jnp.sqrt(cutoff) * B[None, :, :] * jnp.abs(g[None, :, :])
        return trapezoid_(integrand, zeta, axis=-1)

    J_hard = np.asarray(
        vmap(one_surface)((rho, iota), coeff_B, coeff_g, B_star)
    )[0]  # (t, alpha)
    J_mean = np.mean(J_hard, axis=1)
    sigma = (np.max(J_hard, axis=1) - np.min(J_hard, axis=1)) / (J_mean + 1e-30)
    return t, sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eq", default=str(DEFAULT_EQ), help="SC-off equilibrium path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    args = parser.parse_args()

    setup_matplotlib(9)
    eq = load_equilibrium(args.eq)

    # Panel (a): well-split map
    print("Computing well-split map...")
    alpha_delta, t, split = compute_well_split_map(eq)

    # Panel (b): hard vs soft sigma
    print("Computing hard-J sigma...")
    t_hard, sigma_hard = compute_hard_J_sigma(eq)

    print("Computing soft-J sigma...")
    t_soft, alpha_soft, J_soft = smooth_J_map(eq, rho=1.0, num_alpha=96, nt=61)
    J_mean_soft = np.mean(J_soft, axis=0)  # mean over alpha → (t,)
    sigma_soft = (np.max(J_soft, axis=0) - np.min(J_soft, axis=0)) / (J_mean_soft + 1e-30)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.85))

    # (a) Well-split map
    ax_a.pcolormesh(
        t,
        alpha_delta,
        split.astype(float),
        cmap=ListedColormap(["#F2F2F2", COLORS["failure"]]),
        vmin=0,
        vmax=1,
        shading="auto",
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(-np.pi, np.pi)
    ax_a.set_yticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    ax_a.set_yticklabels(
        [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    )
    ax_a.axhline(0.0, color="white", lw=0.8, ls="--", alpha=0.8)
    ax_a.set_xlabel(r"pitch $t$")
    ax_a.set_ylabel(r"$\alpha-\alpha_0$")
    ax_a.set_title("(a) Trapped-region topology", loc="left")
    ax_a.legend(
        handles=[
            Patch(facecolor="#F2F2F2", edgecolor="0.6", label="single well"),
            Patch(facecolor=COLORS["failure"], label=r"$\geq 2$ wells"),
        ],
        loc="center right",
        ncols=1,
    )

    # (b) Hard vs soft σ_J
    floor = 1e-3
    ax_b.plot(t_hard, np.maximum(sigma_hard, floor), color=COLORS["failure"], lw=1.7, label=r"hard-cutoff $J$")
    # Interpolate sigma_soft onto same t grid for clean comparison
    sigma_soft_interp = np.interp(t_hard, t_soft, sigma_soft)
    ax_b.plot(t_hard, np.maximum(sigma_soft_interp, floor), color=COLORS["compact"], lw=1.7, ls="--",
              label=r"soft-proxy $\widetilde{J}$")
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel(r"pitch $t$")
    ax_b.set_ylabel(r"$\sigma_J(t)$")
    ax_b.set_title(r"(b) Apparent residual reduction", loc="left")
    ax_b.set_yscale("log")
    ax_b.set_ylim(floor, 10.0)
    ax_b.grid(False)
    ax_b.legend(loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.93, wspace=0.32)
    output = save_matplotlib(fig, args.output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
