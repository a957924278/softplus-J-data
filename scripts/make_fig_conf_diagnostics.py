"""Independent orbit and neoclassical validation, one figure per Results subsection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fig_common import (
    COLORS,
    data_path,
    figure_path,
    read_loss_curve,
    save_matplotlib,
    setup_matplotlib,
)


# Case name prefixes shared by the SIMPLE loss curves (data/simple/) and the
# NEO outputs (data/neo/), e.g. sc_off_beta0.00_s0.01.dat, sc_off_neo_out.beta0.00.
SC_OFF = ("SC-off", "sc_off", "beta0.00", COLORS["failure"])
SC_ON = ("SC-on", "sc_on", "beta0.00", COLORS["ablation"])
COMPACT = ("Compact", "compact", "beta0.00", COLORS["compact"])
BALANCED_VAC = (
    "Balanced, $\\beta=0$",
    "balanced_vac",
    "beta0.00",
    COLORS["balanced"],
)
BALANCED_BETA = (
    "Balanced, $\\beta=4.12\\%$",
    "balanced",
    "beta4.12",
    COLORS["accent"],
)

# Each Results subsection gets its own confinement figure. The loss panel shows
# every configuration discussed in that subsection; the ripple panel shows the
# vacuum equilibria, for which NEO was run.
GROUPS = (
    ("sc", (SC_OFF, SC_ON), (SC_OFF, SC_ON), "fig6_conf_sc.pdf"),
    ("compact", (COMPACT,), (COMPACT,), "fig8_conf_compact.pdf"),
    (
        "balanced",
        (BALANCED_VAC, BALANCED_BETA),
        (BALANCED_VAC,),
        "fig10_conf_balanced.pdf",
    ),
)

SIMPLE_DIR = data_path("simple")
NEO_DIR = data_path("neo")
W7X_NEO = NEO_DIR / "w7x_high_mirror_neo_out.beta0.00"
RADII = ((0.01, "-"), (0.25, "--"))


def _loss_path(case_name: str, beta_tag: str, radius: float) -> Path:
    """Resolve the SIMPLE loss curve, preferring the GPU-traced file when present."""
    gpu = SIMPLE_DIR / f"{case_name}_{beta_tag}_s{radius:.2f}_gpu.dat"
    return gpu if gpu.exists() else SIMPLE_DIR / f"{case_name}_{beta_tag}_s{radius:.2f}.dat"


def _read_neo_finite(path: Path, ns_max: int = 128):
    """Read NEO data and connect the remaining finite radial samples."""
    data = np.loadtxt(path, dtype=float)
    valid = np.isfinite(data[:, 1])
    return data[valid, 0] / ns_max, data[valid, 1]


def plot_group(loss_cases, neo_cases, output):
    """Plot the fast-ion and neoclassical panels for one subsection."""
    fig, (ax_loss, ax_neo) = plt.subplots(1, 2, figsize=(7.2, 3.05))

    # The legend lists each configuration/launch-surface pair explicitly so the
    # colour and line style can be read as one combination.
    loss_handles = []
    peak_loss = 0.0
    for label, case_name, beta_tag, color in loss_cases:
        for radius, linestyle in RADII:
            time, loss = read_loss_curve(_loss_path(case_name, beta_tag, radius))
            peak_loss = max(peak_loss, float(np.max(loss)))
            ax_loss.plot(time, loss, color=color, ls=linestyle, lw=2)
            loss_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    ls=linestyle,
                    lw=2,
                    label=rf"{label}, $s_0={radius:.2f}$",
                )
            )

    for label, case_name, _, color in neo_cases:
        neo_name = "balanced" if case_name == "balanced_vac" else case_name
        s, eps = _read_neo_finite(NEO_DIR / f"{neo_name}_neo_out.beta0.00")
        ax_neo.plot(s, eps, "o-", color=color, ms=3.5, lw=2, label=label)

    # W7-X High Mirror is kept in every subsection as the common reference.
    s_w7x, eps_w7x = _read_neo_finite(W7X_NEO)
    ax_neo.plot(
        s_w7x,
        eps_w7x,
        "s-",
        color=COLORS["reference"],
        ms=3.5,
        lw=1.7,
        label="W7-X High Mirror",
    )

    ax_loss.set_xscale("log")
    ax_loss.set_xlim(right=0.2)
    # A floor keeps the axis meaningful for the compact case, whose losses are
    # zero to numerical precision and would otherwise autoscale to ~1e-16.
    ax_loss.set_ylim(0.0, max(1.0, 1.06 * peak_loss))
    ax_loss.set_xlabel(r"trace time (s)")
    ax_loss.set_ylabel(r"$\alpha$-particle loss fraction (%)")
    ax_loss.set_title("(a) Fast-ion confinement", loc="left")
    ax_loss.legend(
        handles=loss_handles,
        loc="upper left",
        ncols=1,
        handlelength=2.6,
        fontsize=7.4,
    )

    ax_neo.set_yscale("log")
    ax_neo.set_xlim(0, 1)
    ax_neo.set_ylim(1e-5, 1e-2)
    ax_neo.set_xlabel(r"normalized toroidal flux $s$")
    ax_neo.set_ylabel(r"$\varepsilon_{\mathrm{eff}}$")
    ax_neo.set_title("(b) Neoclassical effective ripple", loc="left")
    ax_neo.legend(
        loc="lower center",
        ncols=1,
        columnspacing=0.9,
        handlelength=2.0,
        fontsize=7.0,
    )
    ax_loss.grid(False)
    ax_neo.grid(False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.93, wspace=0.30)
    print(f"saved: {save_matplotlib(fig, output)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default=str(figure_path()),
        help="Directory for the per-subsection outputs.",
    )
    args = parser.parse_args()
    setup_matplotlib(9)

    outdir = Path(args.outdir)
    for name, loss_cases, neo_cases, filename in GROUPS:
        print(f"plotting group: {name}")
        plot_group(loss_cases, neo_cases, outdir / filename)


if __name__ == "__main__":
    main()
