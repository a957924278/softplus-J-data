"""Generate publication figures for coil vacuum Poincare and free-boundary verification.

Produces:
  1. ``J_paper/figures/fig_coils_poincare.pdf`` (+ ``.png``)
     Vacuum Poincare cross-sections at phi = 0, pi/(2*NFP), pi/NFP with the
     fixed-boundary equilibrium LCFS overplotted (Referee 2 #12).
  2. ``J_paper/figures/fig_freeb_comparison.pdf`` (+ ``.png``)
     Free-boundary vs fixed-boundary LCFS cross-sections, rotational transform
     iota(rho) (including Poincare vacuum iota), and hard-cutoff action
     invariance sigma_J(t) (Referee 2 #13).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fig_common import COLORS, data_path, figure_path, save_matplotlib, setup_matplotlib

FREEB_DIR = data_path("freeb")
POINCARE_PATH = FREEB_DIR / "poincare_vac.npz"
ANALYSIS_NPZ = FREEB_DIR / "freeb_analysis.npz"

PLANE_TITLES = (
    r"(a) $\phi = 0$",
    r"(b) $\phi = \pi / (2 N_{\mathrm{FP}})$",
    r"(c) $\phi = \pi / N_{\mathrm{FP}}$",
)


def plot_poincare_figure(pc, output: Path) -> Path:
    """Plot 3-panel vacuum Poincare sections with fixed-boundary LCFS overlay."""
    setup_matplotlib(8.8)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.85),
        gridspec_kw={"wspace": 0.28, "left": 0.08, "right": 0.99, "bottom": 0.17, "top": 0.80},
    )

    fracs = pc["fracs"]
    confined = pc["confined"]
    planes_R = (pc["R_p0"], pc["R_p1"], pc["R_p2"])
    planes_Z = (pc["Z_p0"], pc["Z_p1"], pc["Z_p2"])
    eq_R = pc["eq_R"]
    eq_Z = pc["eq_Z"]
    eq_ax_R = pc["eq_ax_R"]
    eq_ax_Z = pc["eq_ax_Z"]
    vac_ax_R = pc["r_ax_vac_planes"]
    vac_ax_Z = pc["z_ax_vac_planes"]

    in_idx = np.where(fracs <= 1.0501)[0]
    out_idx = np.where(fracs > 1.0501)[0]
    has_isl = "isl_R_p0" in pc
    isl_R_planes = (pc["isl_R_p0"], pc["isl_R_p1"], pc["isl_R_p2"]) if has_isl else None
    isl_Z_planes = (pc["isl_Z_p0"], pc["isl_Z_p1"], pc["isl_Z_p2"]) if has_isl else None
    cmap_in = plt.get_cmap("viridis")

    for ip, ax in enumerate(axes):
        Rp, Zp = planes_R[ip], planes_Z[ip]

        # Exterior / stochastic edge field lines first (muted gray)
        for idx in out_idx:
            valid = np.isfinite(Rp[:, idx]) & np.isfinite(Zp[:, idx])
            if np.any(valid):
                ax.scatter(
                    Rp[valid, idx],
                    Zp[valid, idx],
                    s=0.35,
                    c="#B0B0B0",
                    alpha=0.45,
                    edgecolors="none",
                    rasterized=True,
                    zorder=1,
                )

        # Exterior n/m = 4/4 edge island chain
        if has_isl:
            Ri, Zi = isl_R_planes[ip], isl_Z_planes[ip]
            for k in range(Ri.shape[1]):
                valid = np.isfinite(Ri[:, k]) & np.isfinite(Zi[:, k])
                if np.any(valid):
                    ax.scatter(
                        Ri[valid, k],
                        Zi[valid, k],
                        s=0.50,
                        c=COLORS["accent"],
                        alpha=0.80,
                        edgecolors="none",
                        rasterized=True,
                        zorder=2,
                    )

        # Interior nested surfaces colored by radial launch fraction
        for k, idx in enumerate(in_idx):
            valid = np.isfinite(Rp[:, idx]) & np.isfinite(Zp[:, idx])
            if np.any(valid):
                color = cmap_in(0.10 + 0.78 * (k / max(len(in_idx) - 1, 1)))
                ax.scatter(
                    Rp[valid, idx],
                    Zp[valid, idx],
                    s=0.65,
                    color=color,
                    alpha=0.90,
                    edgecolors="none",
                    rasterized=True,
                    zorder=3,
                )

        # Fixed-boundary equilibrium LCFS overplot
        ax.plot(
            eq_R[ip, -1],
            eq_Z[ip, -1],
            color=COLORS["failure"],
            lw=1.55,
            zorder=4,
        )
        # Magnetic axes
        ax.plot(
            vac_ax_R[ip],
            vac_ax_Z[ip],
            marker="+",
            color="black",
            ms=5.5,
            mew=1.1,
            ls="none",
            zorder=5,
        )
        ax.plot(
            eq_ax_R[ip],
            eq_ax_Z[ip],
            marker="x",
            color=COLORS["failure"],
            ms=4.5,
            mew=1.0,
            ls="none",
            zorder=5,
        )

        # Tight, balanced viewport showing LCFS + full exterior 4/4 island chain
        r_min, r_max = float(eq_R[ip, -1].min()), float(eq_R[ip, -1].max())
        z_min, z_max = float(eq_Z[ip, -1].min()), float(eq_Z[ip, -1].max())
        if has_isl:
            Ri, Zi = isl_R_planes[ip], isl_Z_planes[ip]
            v_all = np.isfinite(Ri) & np.isfinite(Zi)
            if np.any(v_all):
                r_min = min(r_min, float(np.min(Ri[v_all])))
                r_max = max(r_max, float(np.max(Ri[v_all])))
                z_min = min(z_min, float(np.min(Zi[v_all])))
                z_max = max(z_max, float(np.max(Zi[v_all])))
        r_mid, z_mid = 0.5 * (r_min + r_max), 0.5 * (z_min + z_max)
        half = 0.5 * max(r_max - r_min, z_max - z_min) * 1.09
        ax.set_xlim(r_mid - half, r_mid + half)
        ax.set_ylim(z_mid - half, z_mid + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(PLANE_TITLES[ip], pad=5)
        ax.set_xlabel(r"$R\ (\mathrm{m})$")
        if ip == 0:
            ax.set_ylabel(r"$Z\ (\mathrm{m})$")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap_in(0.45),
            markersize=4.5,
            label="Nested core surfaces",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["accent"],
            markersize=4.5,
            label=r"Exterior $4/4$ island chain",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["failure"],
            lw=1.6,
            label="Fixed-boundary LCFS",
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            color="black",
            ms=5.5,
            mew=1.1,
            ls="none",
            label="Vacuum axis",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.45,
        frameon=False,
    )
    saved = save_matplotlib(fig, output)
    plt.close(fig)
    return saved


def plot_freeb_comparison(pc, an, output: Path, use_phiadj: bool = False) -> Path:
    """Plot free-boundary vs fixed-boundary LCFS, iota(rho), and hard-J sigma_J(t)."""
    setup_matplotlib(8.8)
    fig = plt.figure(figsize=(7.25, 5.60))
    gs_top = fig.add_gridspec(
        1, 3, left=0.08, right=0.99, top=0.89, bottom=0.56, wspace=0.28
    )
    gs_bot = fig.add_gridspec(
        1, 2, left=0.09, right=0.98, top=0.45, bottom=0.175, wspace=0.28
    )
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    ax_iota = fig.add_subplot(gs_bot[0, 0])
    ax_sig = fig.add_subplot(gs_bot[0, 1])

    eq_R = pc["eq_R"]
    eq_Z = pc["eq_Z"]
    b4_key = "freeb_beta4_phiadj" if (use_phiadj and "freeb_beta4_phiadj_R_lcfs" in an) else "freeb_beta4"
    has_vac = "freeb_vac_R_lcfs" in an
    has_b4 = f"{b4_key}_R_lcfs" in an

    for ip, ax in enumerate(axes_top):
        ax.plot(
            eq_R[ip, -1],
            eq_Z[ip, -1],
            color="black",
            lw=1.6,
            label="Fixed-boundary target",
            zorder=3,
        )
        if has_vac:
            ax.plot(
                an["freeb_vac_R_lcfs"][ip],
                an["freeb_vac_Z_lcfs"][ip],
                color=COLORS["compact"],
                lw=1.45,
                ls="--",
                label=r"Free-boundary, $\beta=0$",
                zorder=4,
            )
        if has_b4:
            ax.plot(
                an[f"{b4_key}_R_lcfs"][ip],
                an[f"{b4_key}_Z_lcfs"][ip],
                color=COLORS["balanced"],
                lw=1.45,
                ls="-.",
                label=r"Free-boundary, finite-$\beta$",
                zorder=5,
            )
        r_min, r_max = float(eq_R[ip, -1].min()), float(eq_R[ip, -1].max())
        z_min, z_max = float(eq_Z[ip, -1].min()), float(eq_Z[ip, -1].max())
        if has_b4:
            r_min = min(r_min, float(an[f"{b4_key}_R_lcfs"][ip].min()))
            r_max = max(r_max, float(an[f"{b4_key}_R_lcfs"][ip].max()))
            z_min = min(z_min, float(an[f"{b4_key}_Z_lcfs"][ip].min()))
            z_max = max(z_max, float(an[f"{b4_key}_Z_lcfs"][ip].max()))
        r_mid, z_mid = 0.5 * (r_min + r_max), 0.5 * (z_min + z_max)
        half = 0.5 * max(r_max - r_min, z_max - z_min) * 1.12
        ax.set_xlim(r_mid - half, r_mid + half)
        ax.set_ylim(z_mid - half, z_mid + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(PLANE_TITLES[ip], pad=4)
        ax.set_xlabel(r"$R\ (\mathrm{m})$")
        if ip == 0:
            ax.set_ylabel(r"$Z\ (\mathrm{m})$")

    h_top, l_top = axes_top[0].get_legend_handles_labels()
    fig.legend(
        h_top,
        l_top,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
    )

    # Panel (d): iota(rho)
    rho_g = an["rho_grid"]
    ax_iota.axhline(0.8, color="#B0B0B0", lw=0.8, ls=":", zorder=1)
    ax_iota.axhline(1.0, color="#B0B0B0", lw=0.8, ls=":", zorder=1)
    ax_iota.text(0.03, 0.804, r"$\iota=4/5$", fontsize=7.0, color="#666666")
    ax_iota.text(0.03, 0.985, r"$\iota=4/4$", fontsize=7.0, color="#666666", va="top")
    ax_iota.plot(
        rho_g,
        an["fixed_vac_iota"],
        color="black",
        lw=1.6,
        label=r"Fixed-b, $\beta=0$",
    )
    ax_iota.plot(
        rho_g,
        an["fixed_beta4_iota"],
        color=COLORS["reference"],
        lw=1.4,
        ls=":",
        label=r"Fixed-b, $\langle\beta\rangle=4.12\%$",
    )
    if has_vac:
        ax_iota.plot(
            an["freeb_vac_rho"],
            an["freeb_vac_iota"],
            color=COLORS["compact"],
            lw=1.5,
            ls="--",
            label=r"Free-b, $\beta=0$",
        )
    if has_b4:
        ax_iota.plot(
            an[f"{b4_key}_rho"],
            an[f"{b4_key}_iota"],
            color=COLORS["balanced"],
            lw=1.5,
            ls="-.",
            label=r"Free-b, $\langle\beta\rangle=4.43\%$",
        )
    # Overplot Poincare vacuum iota points
    mask_pc = pc["confined"] & (pc["fracs"] <= 1.0001)
    if np.any(mask_pc):
        ax_iota.scatter(
            pc["rho_eff"][mask_pc],
            pc["iota_vac"][mask_pc],
            s=18,
            facecolors="none",
            edgecolors=COLORS["failure"],
            linewidths=0.95,
            label="Poincaré (vacuum)",
            zorder=6,
        )
    ax_iota.set_xlim(0.0, 1.0)
    ax_iota.set_ylim(0.75, 1.015)
    ax_iota.set_xlabel(r"Normalized radius $\rho$")
    ax_iota.set_ylabel(r"Rotational transform $\iota$")
    ax_iota.set_title(r"(d) Rotational transform profile", pad=4)

    # Panel (e): Hard-cutoff sigma_J(t) at rho=1
    if "w7x_tJ" in an:
        ax_sig.plot(
            an["w7x_tJ"],
            np.maximum(an["w7x_sigmaJ"], 1e-4),
            color="#999999",
            lw=1.25,
            ls="-",
            label="W7-X (ref.)",
            zorder=1,
        )
    ax_sig.plot(
        an["fixed_vac_tJ"],
        np.maximum(an["fixed_vac_sigmaJ"], 1e-4),
        color="black",
        lw=1.6,
        label=r"Fixed-b, $\beta=0$",
    )
    ax_sig.plot(
        an["fixed_beta4_tJ"],
        np.maximum(an["fixed_beta4_sigmaJ"], 1e-4),
        color=COLORS["reference"],
        lw=1.4,
        ls=":",
        label=r"Fixed-b, $\langle\beta\rangle=4.12\%$",
    )
    if has_vac:
        ax_sig.plot(
            an["freeb_vac_tJ"],
            np.maximum(an["freeb_vac_sigmaJ"], 1e-4),
            color=COLORS["compact"],
            lw=1.5,
            ls="--",
            label=r"Free-b, $\beta=0$",
        )
    if has_b4:
        ax_sig.plot(
            an[f"{b4_key}_tJ"],
            np.maximum(an[f"{b4_key}_sigmaJ"], 1e-4),
            color=COLORS["balanced"],
            lw=1.5,
            ls="-.",
            label=r"Free-b, $\langle\beta\rangle=4.43\%$",
        )
    ax_sig.set_yscale("log")
    ax_sig.set_xlim(0.0, 1.0)
    ax_sig.set_ylim(5e-3, 1e1)
    ax_sig.set_xlabel(r"Normalized pitch $t$")
    ax_sig.set_ylabel(r"$\sigma_J(t)$ at $\rho = 1$")
    ax_sig.set_title(r"(e) Hard-cutoff action invariance", pad=4)

    # Single shared legend below panels (d) and (e) — zero curve overlap
    h_d, l_d = ax_iota.get_legend_handles_labels()
    h_e, l_e = ax_sig.get_legend_handles_labels()
    seen = {}
    for h, l in list(zip(h_d, l_d)) + list(zip(h_e, l_e)):
        if l not in seen:
            seen[l] = h
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="lower center",
        bbox_to_anchor=(0.53, 0.005),
        ncol=3,
        fontsize=7.6,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.1,
    )

    saved = save_matplotlib(fig, output)
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poincare-out",
        default=str(figure_path("fig_coils_poincare.pdf")),
        help="Output path for vacuum Poincare figure.",
    )
    parser.add_argument(
        "--freeb-out",
        default=str(figure_path("fig_freeb_comparison.pdf")),
        help="Output path for free-boundary comparison figure.",
    )
    parser.add_argument(
        "--use-phiadj",
        action="store_true",
        help="Plot flux-matched PHIEDGE finite-beta free-boundary run instead of fixed PHIEDGE.",
    )
    args = parser.parse_args()

    if not POINCARE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {POINCARE_PATH}. Run coils_freeb_compute.py --step poincare first."
        )
    pc = np.load(POINCARE_PATH)
    p_saved = plot_poincare_figure(pc, Path(args.poincare_out))
    print(f"saved: {p_saved}")

    if ANALYSIS_NPZ.is_file():
        an = np.load(ANALYSIS_NPZ)
        f_saved = plot_freeb_comparison(
            pc, an, Path(args.freeb_out), use_phiadj=args.use_phiadj
        )
        print(f"saved: {f_saved}")
    else:
        print(f"Skipping free-boundary figure: {ANALYSIS_NPZ} not found yet.")


if __name__ == "__main__":
    main()
