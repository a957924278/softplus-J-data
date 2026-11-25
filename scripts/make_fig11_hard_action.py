"""Create fig_hardJ: hard-cutoff J α-variation verification.

Computes normalized σ_J(t) = (max_α J - min_α J) / ⟨J⟩_α
for the matched SC ablation, the two design cases, and W7-X as baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from desc.backend import jnp
from desc.compat import flip_helicity
from desc.examples import get

from fig_common import (
    COLORS,
    boozer_spectral_data,
    data_path,
    figure_path,
    load_equilibrium,
    save_matplotlib,
    setup_matplotlib,
)

DEFAULT_OUTPUT = figure_path("fig11_hard_action.pdf")

CASES = (
    ("SC-off", data_path("equilibria", "sc_off_beta0.00.h5")),
    ("SC-on", data_path("equilibria", "sc_on_beta0.00.h5")),
    ("Compact configuration", data_path("equilibria", "compact_beta0.00.h5")),
    ("Balanced configuration, β=0", data_path("equilibria", "balanced_vac_beta0.00.h5")),
)

try:
    from jax.numpy import trapezoid as _trapezoid
except ImportError:
    from jax.numpy import trapz as _trapezoid


def trapezoid_(y, x, axis=-1):
    return _trapezoid(y, x, axis=axis)


from softplus_j import _boozer_B_star_from_t


def compute_hard_J_sigma(eq, num_alpha=96, nt=61, nzeta=2000):
    """Compute hard-cutoff J and normalized α-variation σ_J(t).

    Returns (t, sigma_J) where sigma_J = (max_α J - min_α J) / mean_α J.
    """
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

    # B* from true min/max on each surface
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
        return trapezoid_(integrand, zeta, axis=-1)  # (t, alpha)

    J_hard = np.asarray(
        vmap(one_surface)((rho, iota), coeff_B, coeff_g, B_star)
    )  # (rho, t, alpha)
    J_hard = J_hard[0]  # single surface: (t, alpha)

    # σ_J(t) = (max_α - min_α) / mean_α
    J_mean = np.mean(J_hard, axis=1)  # (t,)
    sigma = (np.max(J_hard, axis=1) - np.min(J_hard, axis=1)) / (J_mean + 1e-30)
    return t, sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--num-alpha", type=int, default=96, help="Alpha samples.")
    parser.add_argument("--nt", type=int, default=61, help="Pitch samples.")
    parser.add_argument("--nzeta", type=int, default=2000, help="Zeta quadrature.")
    args = parser.parse_args()

    setup_matplotlib(9)

    # W7-X baseline
    print("Computing W7-X baseline...")
    eq_w7x = get("W7-X")
    flip_helicity(eq_w7x)
    t_w7x, sigma_w7x = compute_hard_J_sigma(
        eq_w7x, num_alpha=args.num_alpha, nt=args.nt, nzeta=args.nzeta
    )

    curves = [("W7-X (reference)", t_w7x, sigma_w7x, COLORS["reference"], "-", 1.9)]

    colors = {
        "SC-off": COLORS["failure"],
        "SC-on": COLORS["ablation"],
        "Compact configuration": COLORS["compact"],
        "Balanced configuration, β=0": COLORS["balanced"],
    }
    for label, eq_path in CASES:
        print(f"Computing hard J: {label}")
        eq = load_equilibrium(eq_path)
        t_val, sigma = compute_hard_J_sigma(
            eq, num_alpha=args.num_alpha, nt=args.nt, nzeta=args.nzeta
        )
        curves.append((label, t_val, sigma, colors[label], "-", 2.0))

    fig, ax = plt.subplots(figsize=(5.2, 3.15))
    for label, t_val, sigma, color, ls, lw in curves:
        ax.plot(t_val, np.maximum(sigma, 1e-4), color=color, ls=ls, lw=lw, label=label)

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"pitch  $t$")
    ax.set_ylabel(r"$\sigma_J(t)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e2)
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    output = save_matplotlib(fig, args.output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
