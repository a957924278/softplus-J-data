"""Create the SoftConnectivity spline diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import desc.plotting  # noqa: F401 - initialize before setup_matplotlib sets rcParams
from fig_common import (
    COLORS,
    boozer_spectral_data,
    data_path,
    figure_path,
    full_spline_knots,
    load_equilibrium,
    plot_spline_on_boozer,
    save_matplotlib,
    setup_matplotlib,
)

DEFAULT_EQ = data_path("equilibria", "compact_beta0.00.h5")
DEFAULT_OUTPUT = figure_path("fig3_sc_construction.pdf")
CASE_LABEL = "Compact configuration"


def spline_branch_limits(spline, iota, alpha=0.0):
    """Return consecutive maximum and minimum locations on one field line."""
    from interpax import interp1d

    zeta_span = 2.0 * np.pi / spline._NFP
    alpha_min, zeta_min, alpha_max, zeta_max = full_spline_knots(spline, iota)
    alpha_next = alpha + iota * zeta_span
    minimum = float(
        interp1d(
            alpha, alpha_min, zeta_min, method="cubic", period=2.0 * np.pi
        )
    )
    minimum_next = float(
        interp1d(
            alpha_next,
            alpha_min,
            zeta_min,
            method="cubic",
            period=2.0 * np.pi,
        )
    ) + zeta_span
    branch_start = float(
        interp1d(
            alpha, alpha_max, zeta_max, method="cubic", period=2.0 * np.pi
        )
    )
    branch_end = float(
        interp1d(
            alpha_next,
            alpha_max,
            zeta_max,
            method="cubic",
            period=2.0 * np.pi,
        )
    ) + zeta_span
    branch_minimum = (
        minimum if branch_start < minimum < branch_end else minimum_next
    )
    return branch_start, branch_minimum, branch_end


def optimize_spline(eq, *, n_control, grid_multiplier, maxiter, verbose):
    """Optimize only SplineZetaMin while constraining all equilibrium parameters."""
    from desc.grid import LinearGrid
    from desc.objectives import FixParameters, ObjectiveFunction
    from softplus_j import SoftConnectivity, SplineZetaMin
    from desc.optimize import Optimizer

    spline = SplineZetaMin(n_control=n_control, NFP=eq.NFP)
    sample_count = max(8, grid_multiplier * eq.M)
    grid = LinearGrid(
        rho=1.0,
        M=grid_multiplier * eq.M,
        N=grid_multiplier * eq.N,
        NFP=eq.NFP,
        sym=False,
    )
    objective = ObjectiveFunction(
        (
            SoftConnectivity(
                eq,
                spline=spline,
                grid=grid,
                num_alpha=sample_count,
                t=np.linspace(0.01, 0.99, sample_count),
                weight=1.0,
            ),
        )
    )
    constraints = (FixParameters(eq),)
    optimizer = Optimizer("lsq-exact")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (eq_fixed, spline), result = optimizer.optimize(
            [eq, spline],
            objective,
            constraints=constraints,
            maxiter=maxiter,
            verbose=verbose,
            ftol=1e-3,
            xtol=1e-8,
            copy=True,
        )
    return eq_fixed, spline, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eq", default=str(DEFAULT_EQ), help="DESC equilibrium path.")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Output PNG path."
    )
    parser.add_argument(
        "--n-control", type=int, default=8, help="Spline control points."
    )
    parser.add_argument(
        "--grid-multiplier",
        type=int,
        default=4,
        help="Boozer grid multiplier relative to eq.M and eq.N.",
    )
    parser.add_argument(
        "--maxiter", type=int, default=40, help="Spline fit iterations."
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="DESC optimizer verbosity."
    )
    args = parser.parse_args()
    setup_matplotlib(9)
    eq = load_equilibrium(args.eq)
    eq_fixed, spline, result = optimize_spline(
        eq,
        n_control=args.n_control,
        grid_multiplier=args.grid_multiplier,
        maxiter=args.maxiter,
        verbose=args.verbose,
    )

    fig, (ax_map, ax_line) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    plot_spline_on_boozer(eq_fixed, spline, ax_map)
    ax_map.set_xlabel(r"$\zeta_{\mathrm{Boozer}}$")
    ax_map.set_ylabel(r"$\theta_{\mathrm{Boozer}}$")
    ax_map.set_title("(a) Fitted connectivity curves", loc="left")
    ax_map.legend(
        loc="upper center",
        ncols=2,
        handlelength=1.8,
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        edgecolor="0.75",
    )

    # Use the fitted spline, rather than sampled |B| extrema, to define the
    # decreasing and increasing portions of one complete max-to-max branch.
    alpha = 0.0
    bd = boozer_spectral_data(eq_fixed, rho=1.0, keys=("|B|_mn_B", "iota"))
    iota = float(bd["iota"][0])
    zeta_start, zeta_min, zeta_end = spline_branch_limits(spline, iota, alpha)
    branch_zeta = np.linspace(zeta_start, zeta_end, 2400)
    theta = alpha + iota * branch_zeta
    nodes = np.stack((np.ones_like(branch_zeta), theta, branch_zeta), axis=-1)
    branch_field = np.asarray(
        bd["basis"].evaluate(nodes) @ bd["coeff_B"][0], dtype=float
    )
    path = (branch_zeta - zeta_start) / (zeta_end - zeta_start)
    x_min = (zeta_min - zeta_start) / (zeta_end - zeta_start)
    field_min = float(np.interp(zeta_min, branch_zeta, branch_field))
    field_max = float(branch_field[0])
    ax_line.plot(path, branch_field, color=COLORS["reference"], lw=1.7)
    ax_line.axvline(x_min, color=COLORS["failure"], lw=1.2, ls="--")
    ax_line.axvspan(0.0, x_min, color=COLORS["compact"], alpha=0.08, lw=0)
    ax_line.axvspan(x_min, 1.0, color=COLORS["balanced"], alpha=0.08, lw=0)
    ax_line.text(
        0.23,
        0.96,
        "decreasing",
        transform=ax_line.transAxes,
        ha="center",
        va="top",
        color=COLORS["compact"],
    )
    ax_line.text(
        0.77,
        0.96,
        "increasing",
        transform=ax_line.transAxes,
        ha="center",
        va="top",
        color=COLORS["balanced"],
    )
    ax_line.text(
        x_min,
        field_min,
        r"  $\zeta_{\min}$",
        color=COLORS["failure"],
        va="bottom",
    )
    ax_line.text(0.01, field_max, r"$\zeta_{\max}$", va="bottom")
    ax_line.set_xlim(0.0, 1.0)
    ax_line.set_xlabel("normalized path coordinate")
    ax_line.set_ylabel(r"$|B|$")
    ax_line.set_title("(b) Monotonic branch ordering", loc="left")
    ax_line.grid(False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.94, wspace=0.30)

    output = save_matplotlib(fig, args.output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
