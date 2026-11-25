"""Create fig_balanced_profiles: pressure/kinetic profiles, iota, and bootstrap current.

Three-panel publication figure for the balanced NFP=4 configuration:
  (a) Normalized pressure p(rho)/p0 and thermodynamically consistent kinetic profiles
      T(rho)/T0 and n(rho)/n0 at <beta> = 4.12%.
  (b) Rotational transform iota(rho) comparing vacuum (beta=0), finite-beta zero-current
      (<beta>=4.12%, I=0), and finite-beta with self-consistent SFINCS bootstrap current,
      with low-order NFP=4 rational surfaces (4/5 and 4/4) marked.
  (c) SFINCS flux-surface-averaged bootstrap current density <J_bs . B> (with high-res
      harmonic-convergence checks at s=0.3, 0.5, 0.8) and Landreman-Catto enclosed
      toroidal bootstrap current I_bs(rho) at native equilibrium scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from desc.grid import LinearGrid
from desc.profiles import PowerSeriesProfile

from fig_common import (
    COLORS,
    data_path,
    figure_path,
    load_equilibrium,
    save_matplotlib,
    setup_matplotlib,
)

DEFAULT_VAC_EQ = data_path("equilibria", "balanced_vac_beta0.00.h5")
DEFAULT_FINITE_EQ = data_path("equilibria", "balanced_beta4.12.h5")
DEFAULT_IBS_EQ = data_path("equilibria", "balanced_beta4.12_ibs.h5")
DEFAULT_SFINCS_REACTOR = data_path("sfincs", "sfincs_reactor_results.json")
DEFAULT_SFINCS_CONV = data_path("sfincs", "sfincs_reactor_conv_results.json")
DEFAULT_OUTPUT = figure_path("fig_balanced_profiles.pdf")


def load_sfincs_reactor(reactor_path: Path, conv_path: Path):
    """Load SFINCS reactor-profile <J_bs . B> in DESC right-handed sign convention."""
    raw = json.loads(reactor_path.read_text())
    s_pts = np.array([r["s"] for r in raw["results"]], dtype=float)
    # VMEC wout has signgs = -1; convert to DESC right-handed convention (iota > 0).
    jb_pts = -np.array([r["JB_phys"] for r in raw["results"]], dtype=float)

    conv_s = []
    conv_jb = []
    if conv_path.exists():
        conv = json.loads(conv_path.read_text())
        for r in conv["results"]:
            if r.get("kind") == "bmn" and r.get("tag") == 1e-5 and r.get("status") == "ok":
                conv_s.append(float(r["s"]))
                conv_jb.append(-float(r["JB_phys_AT_m3"]))
    return s_pts, jb_pts, np.array(conv_s, dtype=float), np.array(conv_jb, dtype=float)


def solve_self_consistent_bootstrap(eq_finite, s_pts: np.ndarray, jb_pts: np.ndarray):
    """Integrate Landreman-Catto I_bs(s) and solve self-consistent finite-beta DESC eq."""
    psi_lcfs = float(eq_finite.Psi)
    s_full = np.concatenate(([0.0], s_pts, [1.0]))
    jb_full = np.concatenate(([0.0], jb_pts, [0.0]))
    rho_s = np.sqrt(np.clip(s_full, 1e-6, 1.0))

    grid_s = LinearGrid(
        rho=rho_s, M=eq_finite.M_grid, N=eq_finite.N_grid, NFP=eq_finite.NFP
    )
    B2 = grid_s.compress(eq_finite.compute("<|B|^2>", grid=grid_s)["<|B|^2>"])
    dI_ds = (psi_lcfs / B2) * jb_full
    I_ode = np.zeros_like(s_full)
    for k in range(1, len(s_full)):
        I_ode[k] = I_ode[k - 1] + 0.5 * (dI_ds[k - 1] + dI_ds[k]) * (
            s_full[k] - s_full[k - 1]
        )

    # Enforce vanishing parallel current density on axis (I(rho) = O(rho^4)).
    rho_fit = np.sqrt(s_full)
    powers = [4, 6, 8, 10]
    A = np.column_stack([rho_fit**p for p in powers])
    weights = np.ones_like(rho_fit)
    weights[-1] = 10.0
    coeffs, _, _, _ = np.linalg.lstsq(A * weights[:, None], I_ode * weights, rcond=None)

    eq_bs = eq_finite.copy()
    eq_bs.current = PowerSeriesProfile(params=coeffs, modes=powers, sym="even")
    eq_bs.solve(verbose=0)
    return eq_bs, s_full, I_ode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eq-vac", default=str(DEFAULT_VAC_EQ), type=Path)
    parser.add_argument("--eq-finite", default=str(DEFAULT_FINITE_EQ), type=Path)
    parser.add_argument("--eq-ibs", default=str(DEFAULT_IBS_EQ), type=Path)
    parser.add_argument("--recompute-ibs", action="store_true")
    parser.add_argument("--sfincs-reactor", default=str(DEFAULT_SFINCS_REACTOR), type=Path)
    parser.add_argument("--sfincs-conv", default=str(DEFAULT_SFINCS_CONV), type=Path)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=Path)
    args = parser.parse_args()

    setup_matplotlib(8.5)

    eq_vac = load_equilibrium(args.eq_vac)
    eq_fin = load_equilibrium(args.eq_finite)
    s_pts, jb_pts, conv_s, conv_jb = load_sfincs_reactor(
        args.sfincs_reactor, args.sfincs_conv
    )
    if args.eq_ibs.exists() and not args.recompute_ibs:
        eq_bs = load_equilibrium(args.eq_ibs)
        psi_lcfs = float(eq_fin.Psi)
        s_full = np.concatenate(([0.0], s_pts, [1.0]))
        jb_full = np.concatenate(([0.0], jb_pts, [0.0]))
        grid_s = LinearGrid(
            rho=np.sqrt(np.clip(s_full, 1e-6, 1.0)),
            M=eq_fin.M_grid,
            N=eq_fin.N_grid,
            NFP=eq_fin.NFP,
        )
        B2 = grid_s.compress(eq_fin.compute("<|B|^2>", grid=grid_s)["<|B|^2>"])
        dI_ds = (psi_lcfs / B2) * jb_full
        I_ode = np.zeros_like(s_full)
        for k in range(1, len(s_full)):
            I_ode[k] = I_ode[k - 1] + 0.5 * (dI_ds[k - 1] + dI_ds[k]) * (
                s_full[k] - s_full[k - 1]
            )
    else:
        eq_bs, s_full, I_ode = solve_self_consistent_bootstrap(eq_fin, s_pts, jb_pts)

    rho = np.linspace(0.0, 1.0, 101)
    s = rho**2
    grid_vac = LinearGrid(rho=rho, M=eq_vac.M_grid, N=eq_vac.N_grid, NFP=eq_vac.NFP)
    grid_fin = LinearGrid(rho=rho, M=eq_fin.M_grid, N=eq_fin.N_grid, NFP=eq_fin.NFP)

    iota_vac = grid_vac.compress(eq_vac.compute("iota", grid=grid_vac)["iota"])
    dat_fin = eq_fin.compute(["iota", "p", "magnetic well"], grid=grid_fin)
    iota_fin = grid_fin.compress(dat_fin["iota"])
    p_fin = grid_fin.compress(dat_fin["p"])
    well_fin = grid_fin.compress(dat_fin["magnetic well"])

    dat_bs = eq_bs.compute(["iota", "<J*B>", "current", "magnetic well"], grid=grid_fin)
    iota_bs = grid_fin.compress(dat_bs["iota"])
    jb_mhd = grid_fin.compress(dat_bs["<J*B>"])
    cur_mhd = grid_fin.compress(dat_bs["current"])

    # Normalized pressure and consistent kinetic profiles:
    # p(s) = p0 (1 - 1.58 s + 0.58 s^2), T(s) = T0 (1 - 0.98 s^1.5), n(s) = p(s)/(2 T(s)).
    p_norm = p_fin / p_fin[0]
    T_norm = 1.0 - 0.98 * np.power(s, 1.5)
    n_norm = p_norm / T_norm

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55))
    ax_a, ax_b, ax_c = axes

    # ------------------------------------------------------------------
    # Panel (a): Normalized pressure and kinetic profiles
    # ------------------------------------------------------------------
    ax_a.plot(rho, p_norm, color=COLORS["balanced"], lw=1.8, label=r"$p(\rho)/p_0$")
    ax_a.plot(
        rho, T_norm, color=COLORS["compact"], lw=1.5, ls="--", label=r"$T(\rho)/T_0$"
    )
    ax_a.plot(
        rho, n_norm, color=COLORS["failure"], lw=1.5, ls="-.", label=r"$n(\rho)/n_0$"
    )
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(-0.03, 1.08)
    ax_a.set_xlabel(r"normalized radius $\rho$")
    ax_a.set_ylabel("normalized profile")
    ax_a.set_title("(a) Pressure & kinetic profiles", loc="left", pad=6)
    ax_a.legend(loc="lower left", frameon=False, handlelength=2.1)

    # ------------------------------------------------------------------
    # Panel (b): Rotational transform iota(rho)
    # ------------------------------------------------------------------
    ax_b.axhline(0.80, color="0.55", lw=0.85, ls=":", zorder=1)
    ax_b.axhline(1.00, color="0.55", lw=0.85, ls=":", zorder=1)
    ax_b.text(0.97, 0.794, r"$\iota=4/5$", fontsize=7.2, color="0.40", ha="right", va="top")
    ax_b.text(0.97, 1.005, r"$\iota=4/4$", fontsize=7.2, color="0.40", ha="right", va="bottom")

    ax_b.plot(
        rho,
        iota_vac,
        color=COLORS["reference"],
        lw=1.5,
        ls="--",
        label=r"vacuum ($\beta=0$)",
    )
    ax_b.plot(
        rho,
        iota_fin,
        color=COLORS["balanced"],
        lw=1.8,
        label=r"$\langle\beta\rangle=4.12\%,\, I=0$",
    )
    ax_b.plot(
        rho,
        iota_bs,
        color=COLORS["compact"],
        lw=1.7,
        ls="-.",
        label=r"$\langle\beta\rangle=4.12\%,\, I_{\mathrm{bs}}$",
    )
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(0.76, 1.04)
    ax_b.set_xlabel(r"normalized radius $\rho$")
    ax_b.set_ylabel(r"rotational transform $\iota$")
    ax_b.set_title(r"(b) Rotational transform", loc="left", pad=6)
    ax_b.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.90),
        frameon=False,
        fontsize=7.2,
        handlelength=1.9,
        labelspacing=0.25,
    )

    # ------------------------------------------------------------------
    # Panel (c): Bootstrap current density & enclosed toroidal current
    # ------------------------------------------------------------------
    rho_pts = np.sqrt(s_pts)
    jb_scale = 1e5  # 10^5 A T / m^2

    ax_c.axhline(0.0, color="0.78", lw=0.7, ls=":")
    l_mhd, = ax_c.plot(
        rho,
        jb_mhd / jb_scale,
        color=COLORS["compact"],
        lw=1.6,
        label="DESC fit",
        zorder=3,
    )
    l_sf, = ax_c.plot(
        rho_pts,
        jb_pts / jb_scale,
        "o",
        color=COLORS["compact"],
        ms=3.6,
        label="SFINCS",
        zorder=4,
    )
    handles_c = [l_sf, l_mhd]
    if conv_s.size > 0:
        l_hi, = ax_c.plot(
            np.sqrt(conv_s),
            conv_jb / jb_scale,
            "s",
            mfc="white",
            mec="#111111",
            mew=0.9,
            ms=4.1,
            label="high-res",
            zorder=5,
        )
        handles_c.append(l_hi)

    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(-4.65, 0.35)
    ax_c.set_yticks([0.0, -1.0, -2.0, -3.0, -4.0])
    ax_c.set_xlabel(r"normalized radius $\rho$")
    ax_c.set_ylabel(
        r"$\langle\mathbf{J}_{\mathrm{bs}}\cdot\mathbf{B}\rangle$ [$10^5\,\mathrm{A\cdot T/m^2}$]",
        color=COLORS["compact"],
    )
    ax_c.tick_params(axis="y", colors=COLORS["compact"])
    ax_c.set_title("(c) Bootstrap current (native)", loc="left", pad=6)

    ax_c2 = ax_c.twinx()
    l_cur, = ax_c2.plot(
        rho,
        cur_mhd / 1e3,
        color=COLORS["failure"],
        lw=1.6,
        ls="--",
        label=r"$I_{\mathrm{bs}}(\rho)$",
        zorder=2,
    )
    ax_c2.set_ylim(-22.5, 1.7)
    ax_c2.set_yticks([0, -5, -10, -15, -20])
    ax_c2.set_ylabel(r"enclosed $I_{\mathrm{bs}}$ [kA]", color=COLORS["failure"])
    ax_c2.tick_params(axis="y", colors=COLORS["failure"])
    ax_c2.spines["right"].set_color(COLORS["failure"])
    ax_c2.spines["left"].set_color(COLORS["compact"])

    handles_c.append(l_cur)
    ax_c.legend(
        handles=handles_c,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.4,
        borderpad=0.25,
        labelspacing=0.22,
    )

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.19, top=0.88, wspace=0.40)
    saved = save_matplotlib(fig, args.output)
    plt.close(fig)

    diota = iota_bs - iota_fin
    print(f"saved: {saved}")
    print(f"p0 = {p_fin[0]:.1f} Pa, <beta>_vol = {float(eq_fin.compute('<beta>_vol')['<beta>_vol'])*100:.4f}%")
    print(f"iota_vac: [{iota_vac[0]:.5f}, {iota_vac[-1]:.5f}]")
    print(f"iota_fin (I=0): [{iota_fin[0]:.5f}, {iota_fin[-1]:.5f}]")
    print(
        f"iota_bs: axis={iota_bs[0]:.5f}, min={np.min(iota_bs):.5f} (rho={rho[np.argmin(iota_bs)]:.2f}), edge={iota_bs[-1]:.5f}"
    )
    print(
        f"Delta iota_bs: axis={diota[0]:+.5f}, max|diota|={np.max(np.abs(diota)):.5f} (rho={rho[np.argmax(np.abs(diota))]:.2f}), edge={diota[-1]:+.5f}"
    )
    print(f"I_bs(LCFS): ODE={I_ode[-1]/1e3:+.3f} kA, fitted={cur_mhd[-1]/1e3:+.3f} kA")
    print(f"well_fin(edge)={well_fin[-1]:+.5f}")


if __name__ == "__main__":
    main()
