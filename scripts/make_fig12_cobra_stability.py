"""Create fig_cobra: COBRA linear ideal-ballooning growth rate for Balanced at beta=4.12%.

Compares the baseline zero-current equilibrium (I_tor = 0) with the self-consistent
SFINCS bootstrap-current equilibrium (I_bs = -14.82 kA).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig_common import COLORS, data_path, figure_path, save_matplotlib, setup_matplotlib

DEFAULT_COBRA = data_path("cobra", "balanced_cobra_grate.beta4.12")
DEFAULT_COBRA_IBS = data_path("cobra", "balanced_cobra_grate.beta4.12_ibs")
DEFAULT_OUTPUT = figure_path("fig12_cobra_stability.pdf")
NZETA, NTHETA, NS = 8, 8, 16


def read_cobra_growth(path, nzeta=NZETA, ntheta=NTHETA, ns=NS):
    """Parse COBRA output: (nzeta, ntheta, ns) growth rate array."""
    data = np.loadtxt(path)
    growth = np.zeros((nzeta, ntheta, ns))
    for i in range(nzeta):
        for j in range(ntheta):
            start = (i * ntheta + j) * (ns + 1) + 1
            end = (i * ntheta + j + 1) * (ns + 1)
            growth[i, j, :] = data[start:end, 2]
    return growth


def summarize_profile(growth):
    """Return (min, mean, max) profiles including the s=0 origin."""
    g_max = np.insert(np.max(growth, axis=(0, 1)), 0, 0.0)
    g_min = np.insert(np.min(growth, axis=(0, 1)), 0, 0.0)
    g_mean = np.insert(np.mean(growth, axis=(0, 1)), 0, 0.0)
    return g_min, g_mean, g_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cobra", default=str(DEFAULT_COBRA), help="Baseline (I_tor=0) COBRA output."
    )
    parser.add_argument(
        "--cobra-ibs",
        default=str(DEFAULT_COBRA_IBS),
        help="Self-consistent bootstrap (I_bs) COBRA output.",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Output PDF path."
    )
    args = parser.parse_args()

    growth_base = read_cobra_growth(args.cobra)
    s_vals = np.linspace(0, 1, NS + 1)
    b_min, b_mean, b_max = summarize_profile(growth_base)

    cobra_ibs_path = Path(args.cobra_ibs)
    has_ibs = cobra_ibs_path.exists()
    if has_ibs:
        growth_ibs = read_cobra_growth(cobra_ibs_path)
        i_min, i_mean, i_max = summarize_profile(growth_ibs)

    setup_matplotlib(9)
    fig, ax = plt.subplots(figsize=(5.4, 3.25))

    # Baseline I_tor = 0
    ax.fill_between(
        s_vals,
        b_min,
        b_max,
        color=COLORS["compact"],
        alpha=0.14,
        lw=0,
        label=r"$I_{\mathrm{tor}}=0$: min–max",
    )
    ax.plot(
        s_vals,
        b_mean,
        color=COLORS["compact"],
        lw=1.6,
        label=r"$I_{\mathrm{tor}}=0$: mean",
    )
    ax.plot(
        s_vals,
        b_max,
        color=COLORS["compact"],
        lw=1.7,
        ls="--",
        label=r"$I_{\mathrm{tor}}=0$: max",
    )

    # Self-consistent SFINCS bootstrap current I_bs
    if has_ibs:
        ax.fill_between(
            s_vals,
            i_min,
            i_max,
            color=COLORS["balanced"],
            alpha=0.16,
            lw=0,
            label=r"self-consistent $I_{\mathrm{bs}}$: min–max",
        )
        ax.plot(
            s_vals,
            i_mean,
            color=COLORS["balanced"],
            lw=1.6,
            label=r"self-consistent $I_{\mathrm{bs}}$: mean",
        )
        ax.plot(
            s_vals,
            i_max,
            color=COLORS["balanced"],
            lw=1.7,
            ls="-.",
            label=r"self-consistent $I_{\mathrm{bs}}$: max",
        )

    ax.axhspan(0.0, 0.018, color=COLORS["failure"], alpha=0.08, lw=0)
    ax.axhline(0, color="0.25", linewidth=0.9)
    ax.text(
        0.99,
        0.009,
        "unstable",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=plt.rcParams["legend.fontsize"],
        color=COLORS["failure"],
        alpha=0.85,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"normalized toroidal flux  $s$")
    ax.set_ylabel(r"growth rate  $\gamma$")
    ax.set_ylim(-0.295, 0.018)
    ax.grid(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        ncol=2,
        fontsize=7.0,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.25,
    )

    fig.tight_layout()
    output = save_matplotlib(fig, args.output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
