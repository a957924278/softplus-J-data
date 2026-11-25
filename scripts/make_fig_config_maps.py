"""Create per-section configuration comparisons for the Results subsections."""

from __future__ import annotations

import argparse

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import desc.plotting  # noqa: F401 - initialize before setup_matplotlib sets rcParams

from fig_common import (
    boozer_spectral_data,
    data_path,
    figure_path,
    load_equilibrium,
    save_matplotlib,
    setup_matplotlib,
)


def compute_smooth_J_map(
    eq,
    *,
    rho: float = 1.0,
    num_alpha: int = 192,
    nt: int = 96,
    nzeta: int = 1400,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    softplus_sharpness: float = 100.0,
):
    """Compute the smoothed second adiabatic invariant J_tilde(alpha, t).

    Evaluates integral_0^{2pi/NFP} sqrt(sp_eps(1 - B/B*)) * |B| * |sqrt(g)_B| dzeta_B
    with logsumexp smooth extrema (soft_extrema_tau = 0.1), matching the paper's
    original diagnostic formulation.
    """
    from desc.backend import jnp
    from softplus_j import (
        _boozer_B_star_from_t,
        _smoothmax_logsumexp,
        _softplus_relu,
    )

    bd = boozer_spectral_data(eq, rho=rho)
    basis = bd["basis"]
    grid = bd["grid"]
    B_full = np.asarray(basis.evaluate(grid.nodes[:: grid.num_rho]) @ bd["coeff_B"].T).T
    B_range = B_full.max(axis=1) - B_full.min(axis=1)
    tau_eff = (0.1 * B_range / np.log(max(B_full.shape[1], 2)))[:, None]
    B_max = np.asarray(_smoothmax_logsumexp(B_full, axis=1, tau=tau_eff)).squeeze(-1)
    B_min = -np.asarray(_smoothmax_logsumexp(-B_full, axis=1, tau=tau_eff)).squeeze(-1)

    t = np.linspace(0.05, 0.95, nt)
    B_star = np.asarray(
        _boozer_B_star_from_t(jnp.asarray(B_min), jnp.asarray(B_max), jnp.asarray(t))
    )[0]

    iota = float(bd["iota"][0])
    if alpha_min is None and alpha_max is None:
        alpha0 = np.pi - iota * (np.pi / eq.NFP)
        alpha = np.linspace(-np.pi, 0.0, num_alpha, endpoint=False) + alpha0
    elif alpha_min is not None and alpha_max is not None:
        alpha = np.linspace(alpha_min, alpha_max, num_alpha, endpoint=False)
    else:
        raise ValueError("alpha_min and alpha_max must be supplied together")

    zeta = np.linspace(0.0, 2.0 * np.pi / eq.NFP, nzeta)
    theta = alpha[:, None] + iota * zeta[None, :]
    nodes = np.stack(
        (
            np.full_like(theta, rho),
            theta,
            np.broadcast_to(zeta[None, :], theta.shape),
        ),
        axis=-1,
    ).reshape((-1, 3))
    mat = np.asarray(basis.evaluate(nodes))
    B = (mat @ np.asarray(bd["coeff_B"][0])).reshape((num_alpha, nzeta))
    g = (mat @ np.asarray(bd["coeff_g"][0])).reshape((num_alpha, nzeta))

    arg = 1.0 - B[None, :, :] / B_star[:, None, None]
    cutoff = np.asarray(_softplus_relu(jnp.asarray(arg), beta=softplus_sharpness))
    integrand = np.sqrt(cutoff) * B[None, :, :] * np.abs(g[None, :, :])
    J = np.trapezoid(integrand, zeta, axis=-1).T  # (num_alpha, nt)
    return t, alpha, J


CASES = (
    ("SC-off", data_path("equilibria", "sc_off_beta0.00.h5")),
    ("SC-on", data_path("equilibria", "sc_on_beta0.00.h5")),
    ("Compact configuration", data_path("equilibria", "compact_beta0.00.h5")),
    ("Balanced configuration", data_path("equilibria", "balanced_vac_beta0.00.h5")),
)
ROW_LABELS = ("SC-off", "SC-on", "Compact", "Balanced")

# One figure per Results subsection: 5.1 uses the controlled pair, 5.2 and 5.3
# each show a single design configuration.
GROUPS = (
    ("sc", (0, 1), "fig4_config_sc.pdf"),
    ("compact", (2,), "fig7_config_compact.pdf"),
    ("balanced", (3,), "fig9_config_balanced.pdf"),
)

TOP_INCHES = 0.38
BOTTOM_INCHES = 0.46
ROW_INCHES = 1.66


def plot_boundary_no_legend(eq, ax, label_size):
    from desc.plotting import plot_boundary

    plot_boundary(
        eq,
        ax=ax,
        lw=[1.4] * 4,
        size=16,
        xlabel_fontsize=label_size,
        ylabel_fontsize=label_size,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_aspect("equal", adjustable="box")


def boozer_field_map(eq, ntheta=181, nzeta=121):
    """Return a normalized Boozer-surface field map for cross-case comparison."""
    bd = boozer_spectral_data(eq, rho=1.0, keys=("|B|_mn_B", "iota"))
    zeta = np.linspace(0.0, 2.0 * np.pi / eq.NFP, nzeta)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    zz, tt = np.meshgrid(zeta, theta)
    nodes = np.stack((np.ones_like(tt), tt, zz), axis=-1).reshape(-1, 3)
    field = (
        np.asarray(bd["basis"].evaluate(nodes)) @ np.asarray(bd["coeff_B"])[0]
    ).reshape(tt.shape)
    field = (field - field.min()) / (field.max() - field.min())
    return zeta, theta, field, float(bd["iota"][0])


def plot_periodic_fieldline(ax, zeta, theta, **kwargs):
    """Plot a field line without drawing across the periodic theta seam."""
    breaks = np.where(np.abs(np.diff(theta)) > np.pi)[0] + 1
    for segment in np.split(np.arange(theta.size), breaks):
        if segment.size > 1:
            ax.plot(zeta[segment], theta[segment], **kwargs)


def plot_case_group(
    rows,
    case_indices,
    action_derivative_maps,
    alpha0_values,
    output,
):
    """Plot one subsection's configurations with an adaptive derivative scale."""
    label_size = plt.rcParams["axes.labelsize"]
    # Each subsection sets its own upper limit. The derivative magnitudes span
    # more than an order of magnitude between the controlled pair and the design
    # cases, so one shared scale would flatten the SC-off/SC-on contrast.
    derivative_limit = max(
        float(np.nanpercentile(action_derivative_maps[index], 99.5))
        for index in case_indices
    )
    action_norm = Normalize(vmin=0.0, vmax=max(derivative_limit, 1e-12))
    print(f"  colorbar upper limit: {derivative_limit:.4g}")
    nrows = len(case_indices)
    height = TOP_INCHES + BOTTOM_INCHES + ROW_INCHES * nrows
    fig = plt.figure(figsize=(7.25, height))
    grid = fig.add_gridspec(
        nrows,
        3,
        left=0.07,
        right=0.98,
        bottom=BOTTOM_INCHES / height,
        top=1.0 - TOP_INCHES / height,
        wspace=0.42,
        hspace=0.14,
    )
    axes = np.empty((nrows, 3), dtype=object)
    for local_row in range(nrows):
        axes[local_row] = (
            fig.add_subplot(grid[local_row, 0]),
            fig.add_subplot(grid[local_row, 1]),
            fig.add_subplot(grid[local_row, 2]),
        )

    for local_row, case_index in enumerate(case_indices):
        _, eq, t, alpha, J, alpha0 = rows[case_index]
        ax_bnd, ax_booz, ax_j = axes[local_row]
        plot_boundary_no_legend(eq, ax_bnd, label_size)
        ax_bnd.text(
            0.03,
            0.96,
            ROW_LABELS[case_index],
            transform=ax_bnd.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )

        zeta, theta, field, _ = boozer_field_map(eq)
        cs_booz = ax_booz.contour(
            zeta,
            theta,
            field,
            levels=np.linspace(0.05, 0.95, 19),
            cmap="viridis",
            linewidths=0.75,
        )
        nfp = eq.NFP
        ax_booz.set_xticks(
            [0.0, np.pi / nfp, 2.0 * np.pi / nfp],
            [r"$0$", r"$\pi/N_{\mathrm{FP}}$", r"$2\pi/N_{\mathrm{FP}}$"],
        )
        ax_booz.set_yticks([0.0, np.pi, 2.0 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
        ax_booz.set_xlabel(r"$\zeta_B$", fontsize=label_size)
        ax_booz.set_ylabel(r"$\theta_B$", fontsize=label_size)

        im_j = ax_j.imshow(
            action_derivative_maps[case_index],
            origin="lower",
            aspect="auto",
            extent=(
                float(np.min(t)),
                float(np.max(t)),
                alpha0_values[case_index] - np.pi,
                alpha0_values[case_index] + np.pi,
            ),
            cmap="magma",
            norm=action_norm,
        )
        ax_j.set_xlim(float(np.min(t)), float(np.max(t)))
        ax_j.set_ylim(alpha0 - np.pi, alpha0 + np.pi)
        ax_j.set_xticks([0.0, 0.5, 1.0], [r"$0.0$", r"$0.5$", r"$1.0$"])
        ax_j.set_yticks(
            [alpha0 - np.pi, alpha0, alpha0 + np.pi],
            [r"$-\pi$", r"$0$", r"$\pi$"],
        )
        ax_j.set_xlabel(r"pitch $t$", fontsize=label_size)
        ax_j.set_ylabel(r"$\alpha-\alpha_0$", fontsize=label_size)

        if local_row < nrows - 1:
            for ax in (ax_bnd, ax_booz, ax_j):
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    axes[0, 0].set_title("Boundary sections", pad=7)
    axes[0, 1].set_title(r"Normalized $|B|$", pad=7)
    axes[0, 2].set_title(r"$|\partial_\alpha(\widetilde{J}/\widetilde{J}_{\max})|$", pad=7)

    cb_booz = fig.colorbar(
        cs_booz,
        ax=axes[:, 1].tolist(),
        fraction=0.045,
        pad=0.02,
        aspect=30 if nrows > 1 else 15,
        ticks=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    cb_booz.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    cb_j = fig.colorbar(
        im_j,
        ax=axes[:, 2].tolist(),
        fraction=0.045,
        pad=0.02,
        aspect=30 if nrows > 1 else 15,
    )
    cb_j.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])

    saved = save_matplotlib(fig, output)
    print(f"saved: {saved}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default=str(figure_path()),
        help="Directory for the per-subsection outputs.",
    )
    parser.add_argument("--rho", type=float, default=1.0, help="Flux surface for J.")
    parser.add_argument("--num-alpha", type=int, default=192, help="J alpha samples.")
    parser.add_argument("--nt", type=int, default=96, help="J pitch samples.")
    parser.add_argument(
        "--nzeta", type=int, default=1400, help="J zeta quadrature points."
    )
    args = parser.parse_args()

    setup_matplotlib(8.5)
    rows = []
    for label, path in CASES:
        print(f"loading and computing J map: {label}")
        eq = load_equilibrium(path)
        iota = float(eq.compute("iota")["iota"][-1])
        alpha0 = np.mod(np.pi - iota * np.pi / eq.NFP, 2.0 * np.pi)
        t, alpha, J = compute_smooth_J_map(
            eq,
            rho=args.rho,
            num_alpha=args.num_alpha,
            nt=args.nt,
            nzeta=args.nzeta,
            alpha_min=alpha0 - np.pi,
            alpha_max=alpha0 + np.pi,
        )
        rows.append((label, eq, t, alpha, J, alpha0))

    action_derivative_maps = []
    alpha0_values = []
    for _, eq, _, alpha, J, alpha0 in rows:
        # The full alpha interval was evaluated directly above. Keep it
        # unwrapped here so the plot is centered on alpha0 without folding.
        alpha0_values.append(alpha0)
        J_normalized = J / np.nanmax(J)
        action_derivative_maps.append(
            np.abs(np.gradient(J_normalized, alpha, axis=0, edge_order=2))
        )

    outdir = Path(args.outdir)
    for name, case_indices, filename in GROUPS:
        print(f"plotting group: {name}")
        plot_case_group(
            rows,
            case_indices,
            action_derivative_maps,
            alpha0_values,
            outdir / filename,
        )


if __name__ == "__main__":
    main()
