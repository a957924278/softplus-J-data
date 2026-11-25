"""Create fig_maxJ_diagnostic: hard-cutoff radial gradient dJ/drho|_{B*} (maximum-J check).

Three-panel publication figure comparing the Balanced configuration in vacuum (beta=0),
at <beta>=4.12% (I_tor=0), and at <beta>=4.12% with self-consistent SFINCS bootstrap
current (I_bs):
  (a) Vacuum (beta=0): normalized radial gradient (1/<J>_alpha) d<J>_alpha/drho|_{B*}
      on the (rho, t) plane.
  (b) Finite-beta (<beta>=4.12%, I_tor=0): same quantity on the identical diverging
      color scale, showing global maximum-J (dJ/drho < 0 everywhere).
  (c) Radial slices at t = 0.25, 0.50, 0.75 comparing vacuum, finite-beta (I_tor=0),
      and finite-beta with self-consistent I_bs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from fig_common import (
    COLORS,
    boozer_spectral_data,
    data_path,
    figure_path,
    load_equilibrium,
    save_matplotlib,
    setup_matplotlib,
)

DEFAULT_VAC_EQ = data_path("equilibria", "balanced_vac_beta0.00.h5")
DEFAULT_FIN_EQ = data_path("equilibria", "balanced_beta4.12.h5")
DEFAULT_IBS_EQ = data_path("equilibria", "balanced_beta4.12_ibs.h5")
DEFAULT_CACHE = data_path("cobra", "balanced_maxJ_cache.json")
DEFAULT_OUTPUT = figure_path("fig_maxJ_diagnostic.pdf")


def compute_dJ_drho_verified(
    eq,
    rhos=np.linspace(0.15, 0.92, 16),
    drho: float = 0.015,
    num_alpha: int = 96,
    nt: int = 37,
    nzeta: int = 1200,
):
    """Compute exact hard-cutoff dJ/drho|_{B*} via centered finite differences."""
    from desc.grid import LinearGrid

    nfp = eq.NFP
    rhos = np.asarray(rhos, dtype=float)
    all_rhos = np.sort(np.concatenate([rhos - drho, rhos, rhos + drho]))
    assert np.all(np.diff(all_rhos) > 0), "all_rhos must be strictly increasing"

    bd = boozer_spectral_data(eq, rho=all_rhos)
    rho_bd = np.asarray(bd["rho"], dtype=float)
    assert np.allclose(rho_bd, all_rhos, atol=1e-12)

    basis = bd["basis"]
    grid_b = bd["grid"]
    iota_all = np.asarray(bd["iota"], dtype=float)
    coeff_B_all = np.asarray(bd["coeff_B"], dtype=float)

    grid_GI = LinearGrid(rho=all_rhos, M=eq.M_grid, N=eq.N_grid, NFP=nfp)
    dGI = eq.compute(["G", "I"], grid=grid_GI)
    G_all = np.asarray(grid_GI.compress(dGI["G"]), dtype=float)
    I_all = np.asarray(grid_GI.compress(dGI["I"]), dtype=float)
    GI_all = np.abs(G_all + iota_all * I_all)

    nodes_0 = grid_b.nodes[np.isclose(grid_b.nodes[:, 0], all_rhos[0])]
    A_surf = np.asarray(basis.evaluate(nodes_0))

    zeta_max = 2.0 * np.pi / nfp
    zeta = np.linspace(0.0, zeta_max, nzeta)
    t_grid = np.linspace(0.05, 0.95, nt)

    def eval_surface_B_dl(idx):
        iota = float(iota_all[idx])
        cB = coeff_B_all[idx]
        a0 = np.pi - iota * (np.pi / nfp)
        alpha_1d = np.linspace(-np.pi, 0.0, num_alpha, endpoint=False) + a0
        theta = alpha_1d[:, None] + iota * zeta[None, :]
        nodes = np.stack(
            (
                np.full_like(theta, all_rhos[idx]),
                theta,
                np.broadcast_to(zeta[None, :], theta.shape),
            ),
            axis=-1,
        ).reshape((-1, 3))
        mat = np.asarray(basis.evaluate(nodes))
        B = (mat @ cB).reshape((num_alpha, nzeta))
        dl = GI_all[idx] / B
        return B, dl

    rel_grad = np.zeros((len(rhos), nt), dtype=float)
    frac_neg = np.zeros(len(rhos), dtype=float)
    for k, rho in enumerate(rhos):
        i_m, i_0, i_p = 3 * k, 3 * k + 1, 3 * k + 2
        B_surf = coeff_B_all[i_0] @ A_surf.T
        B_min, B_max = float(np.min(B_surf)), float(np.max(B_surf))
        B_star = (1.0 - t_grid) * B_min + t_grid * B_max

        B_m, dl_m = eval_surface_B_dl(i_m)
        B_0, dl_0 = eval_surface_B_dl(i_0)
        B_p, dl_p = eval_surface_B_dl(i_p)

        def calc_J(B, dl):
            arg = np.maximum(1.0 - B[None, :, :] / B_star[:, None, None], 0.0)
            return np.trapezoid(
                np.sqrt(B_star[:, None, None]) * np.sqrt(arg) * dl[None, :, :],
                zeta,
                axis=-1,
            )

        J_m = calc_J(B_m, dl_m)
        J_0 = calc_J(B_0, dl_0)
        J_p = calc_J(B_p, dl_p)

        dJ_drho = (J_p - J_m) / (2.0 * drho)
        dJmean_drho = (np.mean(J_p, axis=1) - np.mean(J_m, axis=1)) / (2.0 * drho)
        Jmean = np.mean(J_0, axis=1)
        rel_grad[k] = dJmean_drho / np.maximum(Jmean, 1e-30)

        trapped = J_0 > 1e-12
        frac_neg[k] = float(np.mean(dJ_drho[trapped] < 0))

    return rhos, t_grid, rel_grad, frac_neg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eq-vac", default=str(DEFAULT_VAC_EQ), type=Path)
    parser.add_argument("--eq-fin", default=str(DEFAULT_FIN_EQ), type=Path)
    parser.add_argument("--eq-ibs", default=str(DEFAULT_IBS_EQ), type=Path)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=Path)
    args = parser.parse_args()

    if args.cache.exists() and not args.recompute:
        cached = json.loads(args.cache.read_text())
        rhos = np.asarray(cached["rhos"], dtype=float)
        t_grid = np.asarray(cached["t_grid"], dtype=float)
        g_vac = np.asarray(cached["rel_grad_vac"], dtype=float)
        g_fin = np.asarray(cached["rel_grad_fin"], dtype=float)
        g_ibs = np.asarray(cached["rel_grad_ibs"], dtype=float)
        f_vac = np.asarray(cached["frac_neg_vac"], dtype=float)
        f_fin = np.asarray(cached["frac_neg_fin"], dtype=float)
        f_ibs = np.asarray(cached["frac_neg_ibs"], dtype=float)
    else:
        from desc import set_device

        try:
            set_device("gpu")
        except Exception:
            pass
        eq_vac = load_equilibrium(args.eq_vac)
        eq_fin = load_equilibrium(args.eq_fin)
        eq_ibs = load_equilibrium(args.eq_ibs)
        rhos, t_grid, g_vac, f_vac = compute_dJ_drho_verified(eq_vac)
        _, _, g_fin, f_fin = compute_dJ_drho_verified(eq_fin)
        _, _, g_ibs, f_ibs = compute_dJ_drho_verified(eq_ibs)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(
            json.dumps(
                {
                    "rhos": rhos.tolist(),
                    "t_grid": t_grid.tolist(),
                    "rel_grad_vac": g_vac.tolist(),
                    "rel_grad_fin": g_fin.tolist(),
                    "rel_grad_ibs": g_ibs.tolist(),
                    "frac_neg_vac": f_vac.tolist(),
                    "frac_neg_fin": f_fin.tolist(),
                    "frac_neg_ibs": f_ibs.tolist(),
                },
                indent=2,
            )
        )

    print(
        f"max-J fraction across rho=[{rhos[0]:.2f},{rhos[-1]:.2f}]: "
        f"vac={f_vac.min()*100:.1f}%--{f_vac.max()*100:.1f}%, "
        f"beta=4.12%(I=0)={f_fin.min()*100:.1f}%--{f_fin.max()*100:.1f}%, "
        f"beta=4.12%(I_bs)={f_ibs.min()*100:.1f}%--{f_ibs.max()*100:.1f}%"
    )

    setup_matplotlib(8.5)
    fig = plt.figure(figsize=(7.35, 2.55))
    # Explicit axes layout so all three panels share identical height & baseline,
    # with a clean vertical colorbar between (b) and (c).
    ax_a = fig.add_axes([0.075, 0.20, 0.230, 0.68])
    ax_b = fig.add_axes([0.320, 0.20, 0.230, 0.68])
    cax = fig.add_axes([0.562, 0.20, 0.014, 0.68])
    ax_c = fig.add_axes([0.725, 0.20, 0.260, 0.68])

    # Diverging colormap centered at 0: blue = negative (maximum-J), red = positive
    vmin, vmax = -1.0, 0.25
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    R, T = np.meshgrid(rhos, t_grid, indexing="ij")

    pcm_a = ax_a.pcolormesh(R, T, g_vac, cmap="RdBu_r", norm=norm, shading="gouraud")
    cs = ax_a.contour(R, T, g_vac, levels=[0.0], colors="k", linewidths=0.9, linestyles="--")
    ax_a.clabel(cs, fmt="0", fontsize=7, inline=True)
    ax_a.set_xlabel(r"normalized radius $\rho$")
    ax_a.set_ylabel(r"normalized pitch $t$")
    ax_a.set_title(r"(a) Vacuum ($\beta=0$)", loc="left", pad=5)
    ax_a.text(
        0.52,
        0.24,
        r"$\partial_\rho J > 0$",
        fontsize=7.4,
        color="#7A1C1C",
        ha="center",
        bbox={"fc": "white", "ec": "none", "alpha": 0.72, "pad": 1.3},
    )

    pcm_b = ax_b.pcolormesh(R, T, g_fin, cmap="RdBu_r", norm=norm, shading="gouraud")
    ax_b.set_xlabel(r"normalized radius $\rho$")
    ax_b.set_ylabel("")
    ax_b.set_yticklabels([])
    ax_b.set_title(r"(b) $\langle\beta\rangle=4.12\%$ ($I_{\mathrm{tor}}=0$)", loc="left", pad=5)
    ax_b.text(
        0.535,
        0.52,
        "100% max-$J$\n" + r"($\partial_\rho J < 0$ everywhere)",
        fontsize=7.3,
        color="#08306B",
        weight="bold",
        ha="center",
        va="center",
        bbox={"fc": "white", "ec": "none", "alpha": 0.62, "pad": 2.0},
    )

    cbar = fig.colorbar(
        pcm_b,
        cax=cax,
        ticks=[-1.0, -0.5, 0.0, 0.25],
    )
    cbar.ax.tick_params(labelsize=7.3)

    # Panel (c): Radial slices at t = 0.25, 0.50, 0.75
    ax_c.axhline(0.0, color="0.35", lw=0.85, ls=":")
    pitches = [(0.25, COLORS["compact"], "-"), (0.50, COLORS["balanced"], "--"), (0.75, COLORS["accent"], "-.")]
    for t_target, col, ls in pitches:
        j = int(np.argmin(np.abs(t_grid - t_target)))
        # Vacuum (thin/muted)
        ax_c.plot(rhos, g_vac[:, j], color=col, lw=1.1, ls=ls, alpha=0.45)
        # Finite-beta I=0 (bold)
        ax_c.plot(rhos, g_fin[:, j], color=col, lw=1.75, ls=ls, label=rf"$t={t_target:.2f}$")
        # Finite-beta I_bs (open markers on every 3rd point to show overlap cleanly)
        ax_c.plot(
            rhos[::3],
            g_ibs[::3, j],
            "o",
            mfc="white",
            mec=col,
            mew=0.9,
            ms=3.3,
            zorder=5,
        )

    ax_c.annotate(
        r"$\beta=0$ (light)",
        xy=(0.42, 0.03),
        xytext=(0.17, 0.19),
        fontsize=6.8,
        color="0.35",
        arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 0.7},
    )
    ax_c.text(
        0.17,
        -0.83,
        r"$\langle\beta\rangle=4.12\%$: lines ($I=0$), $\circ$ ($I_{\mathrm{bs}}$)",
        fontsize=6.5,
        color="0.2",
    )
    ax_c.set_xlim(rhos[0], rhos[-1])
    ax_c.set_ylim(-0.88, 0.36)
    ax_c.set_xlabel(r"normalized radius $\rho$")
    ax_c.set_ylabel(r"$\langle J\rangle_\alpha^{-1}\,\partial\langle J\rangle_\alpha/\partial\rho|_{B^*}$", labelpad=3)
    ax_c.set_title("(c) Radial profiles", loc="left", pad=5)
    ax_c.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.8,
        handlelength=1.7,
        labelspacing=0.18,
        borderpad=0.15,
    )

    saved = save_matplotlib(fig, args.output)
    plt.close(fig)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
