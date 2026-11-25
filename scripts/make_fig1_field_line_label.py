"""Boozer field-line labels over the compact configuration's field strength."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig_common import (
    COLORS,
    boozer_spectral_data,
    data_path,
    figure_path,
    load_equilibrium,
    save_matplotlib,
    setup_matplotlib,
)


DEFAULT_OUTPUT = figure_path("fig1_field_line_label.pdf")
EQUILIBRIUM_PATH = data_path("equilibria", "boozer_label_beta0.00.h5")


def boozer_field_map(eq, ntheta=181, nzeta=121):
    """Return the normalized boundary |B| map used by Figure 3."""
    data = boozer_spectral_data(eq, rho=1.0, keys=("|B|_mn_B", "iota"))
    zeta = np.linspace(0.0, 2.0 * np.pi / eq.NFP, nzeta)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    zz, tt = np.meshgrid(zeta, theta)
    nodes = np.stack((np.ones_like(tt), tt, zz), axis=-1).reshape(-1, 3)
    field = (
        np.asarray(data["basis"].evaluate(nodes)) @ np.asarray(data["coeff_B"])[0]
    ).reshape(tt.shape)
    field = (field - field.min()) / (field.max() - field.min())
    return zeta, theta, field, float(data["iota"][0])


def plot_periodic_fieldline(ax, zeta, theta, **kwargs):
    """Plot a field line without joining across the periodic poloidal seam."""
    breaks = np.where(np.abs(np.diff(theta)) > np.pi)[0] + 1
    for segment in np.split(np.arange(theta.size), breaks):
        if segment.size > 1:
            ax.plot(zeta[segment], theta[segment], **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_matplotlib(9)
    eq = load_equilibrium(EQUILIBRIUM_PATH)
    zeta, theta, field, iota = boozer_field_map(eq)

    fig, (ax_lines, ax_derivative) = plt.subplots(
        1,
        2,
        figsize=(6.5, 2.8),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    # Panel (a): field lines over the actual compact-configuration |B| map.
    ax_lines.contour(
        zeta,
        theta,
        field,
        levels=np.linspace(0.05, 0.95, 19),
        cmap="viridis",
        linewidths=0.75,
    )
    alphas = np.array([0.65, 2.35, 4.05])
    line_colors = [COLORS["compact"], COLORS["balanced"], COLORS["accent"]]
    line_styles = ["-", "--", ":"]
    zeta_fixed = 0.58 * zeta[-1]

    for idx, (alpha, linestyle) in enumerate(zip(alphas, line_styles), start=1):
        theta_line = np.mod(alpha + iota * zeta, 2.0 * np.pi)
        plot_periodic_fieldline(
            ax_lines,
            zeta,
            theta_line,
            color="0.15",
            lw=1.8,
            ls=linestyle,
            zorder=4,
        )
        ax_lines.scatter(
            [0.0], [alpha], s=23, color="white", edgecolor="0.15", lw=0.5,
            zorder=5, clip_on=False
        )
        ax_lines.text(
            0.035 * zeta[-1],
            alpha + 0.18,
            rf"$\alpha_{idx}$",
            color="0.15",
            ha="left",
            va="bottom",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.7},
            zorder=6,
        )

    ax_lines.axvline(zeta_fixed, color="0.55", lw=1.0, ls="--", zorder=0)
    theta_1 = np.mod(alphas[0] + iota * zeta_fixed, 2.0 * np.pi)
    theta_2 = np.mod(alphas[1] + iota * zeta_fixed, 2.0 * np.pi)
    ax_lines.scatter(
        [zeta_fixed, zeta_fixed],
        [theta_1, theta_2],
        s=25,
        color=[line_colors[0], line_colors[1]],
        edgecolor="0.15",
        linewidth=0.5,
        zorder=5,
    )
    ax_lines.annotate(
        "",
        xy=(zeta_fixed, theta_2 - 0.06),
        xytext=(zeta_fixed, theta_1 + 0.06),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.1},
    )
    # ax_lines.text(
    #     zeta_fixed + 0.12,
    #     0.5 * (theta_1 + theta_2) + 0.08,
    #     r"$\Delta\theta_B=\Delta\alpha$",
    #     ha="left",
    #     va="center",
    # )
    ax_lines.set_xlim(0.0, zeta[-1])
    ax_lines.set_ylim(0.0, 2.0 * np.pi)
    ax_lines.set_xlabel(r"Boozer toroidal angle $\zeta_B$")
    ax_lines.set_ylabel(r"Boozer poloidal angle $\theta_B$")
    ax_lines.set_xticks(
        [0.0, np.pi / eq.NFP, 2.0 * np.pi / eq.NFP],
        [r"$0$", r"$\pi/N_{\mathrm{FP}}$", r"$2\pi/N_{\mathrm{FP}}$"],
    )
    ax_lines.set_yticks([0.0, np.pi, 2.0 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
    ax_lines.set_title(r"(a)", loc="left")

    # Panel (b): local coordinate displacement underlying the chain rule.
    x_fixed = 0.46
    y1, y2 = 0.30, 0.72
    ax_derivative.axvline(x_fixed, color="0.55", lw=1.0, ls="--")
    ax_derivative.scatter(
        [x_fixed, x_fixed],
        [y1, y2],
        s=48,
        color=[line_colors[0], line_colors[1]],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax_derivative.annotate(
        "",
        xy=(x_fixed, y2 - 0.025),
        xytext=(x_fixed, y1 + 0.025),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.2},
    )
    ax_derivative.text(
        x_fixed + 0.07,
        y2,
        r"$Q(\rho,\theta_B(\alpha_2,\zeta_B),\zeta_B)$",
        ha="left",
        va="center",
    )
    ax_derivative.text(
        x_fixed + 0.07,
        y1,
        r"$Q(\rho,\theta_B(\alpha_1,\zeta_B),\zeta_B)$",
        ha="left",
        va="center",
    )
    ax_derivative.text(
        x_fixed - 0.08,
        0.5 * (y1 + y2),
        r"vary $\alpha$" "\n" r"at fixed $\rho,\zeta_B$",
        ha="right",
        va="center",
        color="0.25",
    )
    ax_derivative.text(
        0.18,
        0.08,
        r"$\left.\partial_{\alpha}Q\right|_{\rho,\zeta_B}"
        r"=\left.\partial_{\theta_B}Q\right|_{\rho,\zeta_B}$",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax_derivative.text(
        x_fixed-0.2,
        0.90,
        r"$\Delta\theta_B=\Delta\alpha$",
        ha="center",
        va="center",
    )
    ax_derivative.set_xlim(0.0, 1.0)
    ax_derivative.set_ylim(0.0, 1.0)
    ax_derivative.set_xticks([])
    ax_derivative.set_yticks([])
    ax_derivative.set_title(r"(b)", loc="left")
    for spine in ax_derivative.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.89, wspace=0.30)
    output = save_matplotlib(fig, args.output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
