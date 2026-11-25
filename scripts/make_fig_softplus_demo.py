"""Create fig_softplus_demo: softplus regularization of the bounce-action cutoff.

Three-panel publication figure illustrating:
  (a) Positive-part cutoff max(x, 0) vs softplus sp_eps(x) with zoomed inset near x=0.
  (b) Action integrand factor sqrt(max(x, 0)) vs sqrt(sp_eps(x)) with zoomed inset near x=0.
  (c) Derivative d/dx sqrt(sp_eps(x)) vs singular hard derivative 1 / (2 sqrt(x)).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from fig_common import COLORS, figure_path, save_matplotlib, setup_matplotlib

DEFAULT_OUTPUT = figure_path("fig_softplus_demo.pdf")


def softplus(x: np.ndarray, eps: float) -> np.ndarray:
    """Numerically stable softplus sp_eps(x) = eps * log(1 + exp(x / eps))."""
    return eps * np.logaddexp(0.0, x / eps)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid 1 / (1 + exp(-z))."""
    return np.where(
        z >= 0.0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def d_sqrt_softplus_dx(x: np.ndarray, eps: float) -> np.ndarray:
    """Exact derivative d/dx [sqrt(sp_eps(x))] = sigmoid(x/eps) / (2 sqrt(sp_eps(x)))."""
    sp = softplus(x, eps)
    sig = sigmoid(x / eps)
    return sig / (2.0 * np.sqrt(np.maximum(sp, 1e-300)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        type=Path,
        help="Output PDF path (PNG companion is also saved).",
    )
    args = parser.parse_args()

    setup_matplotlib(8.5)

    x_ab = np.linspace(-0.22, 0.32, 1600)
    x_zoom = np.linspace(-0.04, 0.04, 800)
    x_c = np.linspace(-0.14, 0.24, 1600)
    x_pos = np.linspace(1e-4, 0.24, 1200)

    eps_paper = 0.01
    eps_vis = 0.05

    c_hard = "#222222"
    c_paper = COLORS["compact"]
    c_vis = COLORS["failure"]

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65))
    ax_a, ax_b, ax_c = axes

    # ------------------------------------------------------------------
    # Panel (a): max(x, 0) vs sp_eps(x)
    # ------------------------------------------------------------------
    ax_a.axvline(0.0, color="0.78", lw=0.8, ls=":", zorder=1)
    l_hard, = ax_a.plot(
        x_ab,
        np.maximum(x_ab, 0.0),
        color=c_hard,
        lw=1.5,
        ls="--",
        label=r"Hard cutoff ($\epsilon \to 0$)",
        zorder=4,
    )
    l_paper, = ax_a.plot(
        x_ab,
        softplus(x_ab, eps_paper),
        color=c_paper,
        lw=1.7,
        label=r"$\epsilon = 0.01$ (paper)",
        zorder=3,
    )
    l_vis, = ax_a.plot(
        x_ab,
        softplus(x_ab, eps_vis),
        color=c_vis,
        lw=1.5,
        label=r"$\epsilon = 0.05$ (illustrative)",
        zorder=2,
    )
    ax_a.set_xlim(-0.22, 0.32)
    ax_a.set_ylim(-0.02, 0.35)
    ax_a.set_xlabel(r"$x = 1 - B/B^{*}$")
    ax_a.set_ylabel(r"$\operatorname{sp}_\epsilon(x)$")
    ax_a.set_title("(a) Smooth positive part", loc="left", pad=6)

    ax_a_in = inset_axes(
        ax_a,
        width="44%",
        height="44%",
        loc="upper left",
        bbox_to_anchor=(0.08, -0.02, 1.0, 1.0),
        bbox_transform=ax_a.transAxes,
    )
    ax_a_in.axvline(0.0, color="0.78", lw=0.6, ls=":")
    ax_a_in.plot(x_zoom, np.maximum(x_zoom, 0.0), color=c_hard, lw=1.1, ls="--", zorder=4)
    ax_a_in.plot(x_zoom, softplus(x_zoom, eps_paper), color=c_paper, lw=1.3, zorder=3)
    ax_a_in.plot(x_zoom, softplus(x_zoom, eps_vis), color=c_vis, lw=1.2, zorder=2)
    ax_a_in.set_xlim(-0.04, 0.04)
    ax_a_in.set_ylim(-0.004, 0.065)
    ax_a_in.set_xticks([-0.03, 0.0, 0.03])
    ax_a_in.set_yticks([0.0, 0.03, 0.06])
    ax_a_in.tick_params(labelsize=6.5, length=2.2, pad=1.5)
    mark_inset(ax_a, ax_a_in, loc1=3, loc2=4, fc="none", ec="0.55", lw=0.6)

    # ------------------------------------------------------------------
    # Panel (b): sqrt(max(x, 0)) vs sqrt(sp_eps(x))
    # ------------------------------------------------------------------
    ax_b.axvline(0.0, color="0.78", lw=0.8, ls=":", zorder=1)
    ax_b.plot(x_ab, np.sqrt(np.maximum(x_ab, 0.0)), color=c_hard, lw=1.5, ls="--", zorder=4)
    ax_b.plot(x_ab, np.sqrt(softplus(x_ab, eps_paper)), color=c_paper, lw=1.7, zorder=3)
    ax_b.plot(x_ab, np.sqrt(softplus(x_ab, eps_vis)), color=c_vis, lw=1.5, zorder=2)
    ax_b.set_xlim(-0.22, 0.32)
    ax_b.set_ylim(-0.03, 0.62)
    ax_b.set_xlabel(r"$x = 1 - B/B^{*}$")
    ax_b.set_ylabel(r"$\sqrt{\operatorname{sp}_\epsilon(x)}$")
    ax_b.set_title("(b) Action integrand factor", loc="left", pad=6)

    ax_b_in = inset_axes(
        ax_b,
        width="44%",
        height="44%",
        loc="upper left",
        bbox_to_anchor=(0.08, -0.02, 1.0, 1.0),
        bbox_transform=ax_b.transAxes,
    )
    ax_b_in.axvline(0.0, color="0.78", lw=0.6, ls=":")
    ax_b_in.plot(x_zoom, np.sqrt(np.maximum(x_zoom, 0.0)), color=c_hard, lw=1.1, ls="--", zorder=4)
    ax_b_in.plot(x_zoom, np.sqrt(softplus(x_zoom, eps_paper)), color=c_paper, lw=1.3, zorder=3)
    ax_b_in.plot(x_zoom, np.sqrt(softplus(x_zoom, eps_vis)), color=c_vis, lw=1.2, zorder=2)
    ax_b_in.set_xlim(-0.04, 0.04)
    ax_b_in.set_ylim(-0.01, 0.27)
    ax_b_in.set_xticks([-0.03, 0.0, 0.03])
    ax_b_in.set_yticks([0.0, 0.1, 0.2])
    ax_b_in.tick_params(labelsize=6.5, length=2.2, pad=1.5)
    mark_inset(ax_b, ax_b_in, loc1=3, loc2=4, fc="none", ec="0.55", lw=0.6)

    # ------------------------------------------------------------------
    # Panel (c): Derivative smoothness near turning point x=0
    # ------------------------------------------------------------------
    ax_c.axvline(0.0, color="0.78", lw=0.8, ls=":", zorder=1)
    # Hard derivative is 0 for x<0 and 1/(2 sqrt(x)) for x>0
    x_neg = np.linspace(-0.14, -1e-4, 200)
    ax_c.plot(x_neg, np.zeros_like(x_neg), color=c_hard, lw=1.5, ls="--", zorder=4)
    ax_c.plot(x_pos, 0.5 / np.sqrt(x_pos), color=c_hard, lw=1.5, ls="--", zorder=4)
    d_paper = d_sqrt_softplus_dx(x_c, eps_paper)
    d_vis = d_sqrt_softplus_dx(x_c, eps_vis)
    ax_c.plot(x_c, d_paper, color=c_paper, lw=1.7, zorder=3)
    ax_c.plot(x_c, d_vis, color=c_vis, lw=1.5, zorder=2)
    ax_c.set_xlim(-0.14, 0.24)
    ax_c.set_ylim(-0.2, 5.8)
    ax_c.set_xlabel(r"$x = 1 - B/B^{*}$")
    ax_c.set_ylabel(r"$\mathrm{d}\sqrt{\operatorname{sp}_\epsilon(x)}/\mathrm{d}x$")
    ax_c.set_title("(c) Turning-point derivative", loc="left", pad=6)
    ax_c.annotate(
        r"$(2\sqrt{x})^{-1} \to \infty$",
        xy=(0.012, 4.55),
        xytext=(0.075, 4.75),
        fontsize=7.5,
        color=c_hard,
        arrowprops={"arrowstyle": "->", "color": c_hard, "lw": 0.7},
    )

    fig.legend(
        handles=[l_hard, l_paper, l_vis],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.81, wspace=0.36)

    saved = save_matplotlib(fig, args.output)
    plt.close(fig)

    print(f"saved: {saved}")
    print(f"peak d/dx sqrt(sp_0.01): {float(np.max(d_paper)):.4f} at x={float(x_c[np.argmax(d_paper)]):.5f}")
    print(f"peak d/dx sqrt(sp_0.05): {float(np.max(d_vis)):.4f} at x={float(x_c[np.argmax(d_vis)]):.5f}")
    print(f"sp_0.01(0)={softplus(np.array(0.0), eps_paper):.6f}, sqrt(sp_0.01(0))={np.sqrt(softplus(np.array(0.0), eps_paper)):.6f}")


if __name__ == "__main__":
    main()
