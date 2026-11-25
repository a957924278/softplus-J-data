"""SC-off 2x2 diagnostic: field topology and hard/smooth action maps.

Layout:
    (a) Boozer |B| surface and selected field line
    (b) |B| on that line with disconnected trapped intervals
    (c) hard-cutoff J on single-well samples
    (d) fixed-domain smooth action proxy on the full grid

硬版 J：cutoff 用 jnp.maximum(x, 0)，B_min/B_max 用真实的 jnp.max/min；
软版：直接用 DESC 的 SecondAdiabaticInvariantAlphaDerivative 目标函数。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.compat import flip_helicity
from desc.grid import LinearGrid
from desc.io import load
from desc.plotting import plot_boozer_surface
from desc.vmec_utils import ptolemy_linear_transform  # noqa: F401
from softplus_j import (
    SecondAdiabaticInvariantAlphaDerivative,
    _boozer_B_star_from_t,
    _reshape_surface_coefficients,
    _smoothmax_logsumexp,
    boozer_second_adiabatic_invariant_alpha_derivative_from_data,
    boozer_second_adiabatic_invariant_from_data,
)

from fig_common import (
    COLORS,
    count_periodic_true_components,
    data_path,
    field_line_B,
    figure_path,
    save_matplotlib,
    setup_matplotlib,
    surface_B_range,
    true_intervals,
)


DEFAULT_OUTPUT = figure_path("fig2_action_diagnostics.pdf")
SC_OFF_EQUILIBRIUM = data_path("equilibria", "sc_off_beta0.00.h5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Output PNG path."
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # 平衡
    # ----------------------------------------------------------------------
    eq_sc_off = load(SC_OFF_EQUILIBRIUM)
    nfp = eq_sc_off.NFP

    # ----------------------------------------------------------------------
    # 公共 Boozer 网格 / 数据：rho = 1.0 边界
    # ----------------------------------------------------------------------
    RHO = 1.0
    M_BOOZ = 4 * eq_sc_off.M
    N_BOOZ = 4 * eq_sc_off.N

    boozer_grid = LinearGrid(rho=RHO, M=2 * M_BOOZ, N=2 * N_BOOZ, NFP=nfp, sym=False)
    data_keys = ["|B|_mn_B", "sqrt(g)_Boozer_mn", "iota"]
    profiles = get_profiles(data_keys, obj=eq_sc_off, grid=boozer_grid)
    transforms = get_transforms(
        data_keys, obj=eq_sc_off, grid=boozer_grid, M_booz=M_BOOZ, N_booz=N_BOOZ
    )
    data = compute_fun(
        "desc.equilibrium.equilibrium.Equilibrium",
        data_keys,
        params=eq_sc_off.params_dict,
        transforms=transforms,
        profiles=profiles,
    )

    basis = transforms["B"].basis
    grid_b = transforms["grid"]
    rho = jnp.asarray(grid_b.compress(grid_b.nodes[:, 0]))
    iota = jnp.asarray(grid_b.compress(data["iota"]))
    coeff_B = _reshape_surface_coefficients(grid_b, data["|B|_mn_B"])
    coeff_g = _reshape_surface_coefficients(grid_b, data["sqrt(g)_Boozer_mn"])

    # alpha 网格：DESC 的目标函数只采样对称轴一侧；绘图时再按对称性展开。
    NUM_ALPHA = 192
    alpha0 = jnp.pi - iota * (jnp.pi / nfp)
    alpha = jnp.linspace(-jnp.pi, 0.0, NUM_ALPHA, endpoint=False) + alpha0[:, None]
    alpha0_plot = float(np.asarray(alpha0)[0])
    iota_plot = float(np.asarray(iota)[0])

    # pitch 网格 t -> B* = (1-t) B_min + t B_max
    NUM_T = 201
    t = np.linspace(0.0, 1.0, NUM_T)

    NZETA_J = 4000

    # ----------------------------------------------------------------------
    # 硬离散版 J：jnp.maximum 截断 + 真实 max/min，无 softplus / smoothmax
    # ----------------------------------------------------------------------
    def hard_second_adiabatic_invariant(
        basis, rho, iota, coeff_B, coeff_g, num_alpha, t, nzeta=NZETA_J
    ):
        """真正硬截断的第二绝热不变量。

        cutoff = max(1 - B/B*, 0)，B_min/B_max 取每条 Boozer 场线的真实极值。
        alpha 网格按 DESC 约定采样 alpha0 - pi 到 alpha0，其中
        alpha0 = pi - iota*pi/nfp 是反射对称轴；另一侧由对称性展开。
        返回 shape (num_rho, num_alpha, num_t)。
        """
        zeta_max = 2 * np.pi / nfp
        zeta = jnp.linspace(0.0, zeta_max, nzeta)

        # 每条面线上 |B| 的真实极值（用于确定 B*）
        A_full = basis.evaluate(grid_b.nodes[:: grid_b.num_rho])
        B_full = coeff_B @ A_full.T
        B_max_surf = jnp.max(B_full, axis=1)
        B_min_surf = jnp.min(B_full, axis=1)
        B_star = _boozer_B_star_from_t(
            B_min_surf, B_max_surf, jnp.asarray(t)
        )  # (rho, t)

        def one_surface(r_it, cB, cg, B_star_s):
            """对单条磁通面 vmap：返回 (num_alpha, num_t)。"""
            r_, it_ = r_it
            a0 = jnp.pi - it_ * (jnp.pi / nfp)
            alpha_1d = jnp.linspace(-jnp.pi, 0.0, num_alpha, endpoint=False) + a0
            theta = alpha_1d[:, None] + it_ * (zeta[None, :] - 0.0)  # (alpha, zeta)
            rho2d = jnp.broadcast_to(r_, theta.shape)
            zeta2d = jnp.broadcast_to(zeta[None, :], theta.shape)
            nodes = jnp.stack((rho2d, theta, zeta2d), axis=-1).reshape((-1, 3))
            mat = basis.evaluate(nodes)

            B = (mat @ cB).reshape((num_alpha, nzeta))  # (alpha, zeta)
            g = (mat @ cg).reshape((num_alpha, nzeta))
            # cutoff(alpha, zeta, t) = max(1 - B/B*, 0)   硬截断
            arg = 1.0 - B[None, :, :] / B_star_s[:, None, None]  # (t, alpha, zeta)
            cutoff = jnp.maximum(arg, 0.0)
            integrand = jnp.sqrt(cutoff) * B[None, :, :] * jnp.abs(g[None, :, :])
            return trapezoid_(integrand, zeta, axis=-1)  # (t, alpha)

        from jax import vmap

        out = vmap(one_surface)(
            (rho, iota), coeff_B, coeff_g, B_star
        )  # (rho, t, alpha)
        return jnp.transpose(out, (0, 2, 1))  # (rho, alpha, t)

    # jax.numpy.trapezoid 在老版本里叫 trapezoid；做个兼容别名
    try:
        from jax.numpy import trapezoid as _trapezoid
    except ImportError:  # pragma: no cover
        from jax.numpy import trapz as _trapezoid

    def trapezoid_(y, x, axis=-1):
        return _trapezoid(y, x, axis=axis)

    # 软版 dJ/dalpha：直接调用 DESC 目标函数（含平滑 cutoff + 软极值）
    soft_obj = SecondAdiabaticInvariantAlphaDerivative(
        eq_sc_off,
        grid=boozer_grid,
        num_alpha=NUM_ALPHA,
        nzeta=NZETA_J,
        t=t,
        name="soft dJ/dalpha",
    )
    soft_obj.build(use_jit=False, verbose=0)

    J_hard = np.asarray(
        hard_second_adiabatic_invariant(
            basis, rho, iota, coeff_B, coeff_g, NUM_ALPHA, t
        )
    )[0]  # (alpha, t)

    # 软版 J（不是偏导）：用 boozer_second_adiabatic_invariant_from_data
    # B* 与 DESC 默认一致（软极值 smoothmax）；alpha 用同一套网格
    alpha_rho0 = np.asarray(alpha[0])  # (alpha,)
    A_full = np.asarray(basis.evaluate(grid_b.nodes[:: grid_b.num_rho]))
    B_full = np.asarray(coeff_B) @ A_full.T  # (rho, num_eval)
    B_range = B_full.max(axis=1) - B_full.min(axis=1)
    log_n = np.log(max(B_full.shape[1], 2))
    tau_eff = (0.1 * B_range / log_n)[:, None]  # soft_extrema_tau=0.1
    B_max_s = np.asarray(_smoothmax_logsumexp(B_full, axis=1, tau=tau_eff)).squeeze(-1)
    B_min_s = -np.asarray(_smoothmax_logsumexp(-B_full, axis=1, tau=tau_eff)).squeeze(
        -1
    )
    B_star_soft = _boozer_B_star_from_t(
        jnp.asarray(B_min_s), jnp.asarray(B_max_s), jnp.asarray(t)
    )  # (rho, t)
    # from_data 需要 alpha 为 1-D (num_alpha,)；原 alpha 是 (rho=1, num_alpha)
    alpha_1d = np.asarray(alpha)[0]  # (num_alpha,)
    J_soft = np.asarray(
        boozer_second_adiabatic_invariant_from_data(
            grid_b,
            basis,
            data,
            alpha_1d,
            B_star_soft,
            nzeta=NZETA_J,
        )
    )[0]  # (alpha, t)

    # 磁井连通数：对每个 (alpha,t)，{ζ: B(α,ζ) < B*(t)} 沿 ζ 有几个不相连分量
    # 用与硬 J 相同的 B*（面极值 max/min），连通数≥2 即"磁井分裂"
    NZETA_WELL = 4000
    zeta_w = np.linspace(0.0, 2 * np.pi / nfp, NZETA_WELL)
    iota0 = float(np.asarray(iota)[0])
    a0_w = np.pi - iota0 * (np.pi / nfp)
    alpha_w = np.linspace(-np.pi, 0.0, NUM_ALPHA, endpoint=False) + a0_w  # (alpha,)
    nodes_w = np.stack(
        np.broadcast_arrays(
            RHO,
            alpha_w[:, None] + iota0 * zeta_w[None, :],
            zeta_w[None, :],
        ),
        axis=-1,
    ).reshape(-1, 3)
    B_w = (
        np.asarray(np.asarray(basis.evaluate(nodes_w)) @ np.asarray(coeff_B[0]))
    ).reshape(NUM_ALPHA, NZETA_WELL)  # (alpha, zeta)
    # 硬版用的面极值 B*(t)
    B_max_h = float(B_full.max(axis=1)[0])
    B_min_h = float(B_full.min(axis=1)[0])
    B_star_hard = _boozer_B_star_from_t(
        jnp.asarray(B_min_h), jnp.asarray(B_max_h), jnp.asarray(t)
    )  # (t,)

    # 连通数：把 (alpha,zeta) 上 B<B* 的环(周期 ζ)分段计数
    def count_components(mask_row):
        """mask_row: 1D bool 长度 nzeta，ζ 周期。返回 True 段数。"""
        m = np.asarray(mask_row).astype(bool)
        if not m.any():
            return 0
        # 周期边界：首尾都 True 算同一段
        if m[0] and m[-1]:
            merged = np.concatenate([m, m])  # 拼接避免边界假断点
            diffs = np.diff(merged.astype(int))
            return int((diffs == 1).sum())
        diffs = np.diff(m.astype(int))
        return int((diffs == 1).sum())

    well_split = np.zeros((NUM_ALPHA, NUM_T), dtype=bool)  # (alpha, t) 分裂掩膜
    for it_t in range(NUM_T):
        bstar = float(np.asarray(B_star_hard[0, it_t]))
        for ia in range(NUM_ALPHA):
            mask = B_w[ia] < bstar
            if count_components(mask) >= 2:
                well_split[ia, it_t] = True

    # 以 alpha0 为中心显示完整周期。DESC 只算 delta<0 的一侧，
    # 因此用 J(alpha0 + delta) = J(alpha0 - delta) 显式补出 delta>0。
    delta_half = np.linspace(-np.pi, 0.0, NUM_ALPHA, endpoint=False)
    alpha_plot = np.concatenate((delta_half, -delta_half[::-1]))
    J_hard = np.concatenate((J_hard, J_hard[::-1]), axis=0)
    J_soft = np.concatenate((J_soft, J_soft[::-1]), axis=0)
    well_split = np.concatenate((well_split, well_split[::-1]), axis=0)
    t_plot = np.asarray(t)  # (t,) -> (0,1)

    # ======================================================================
    # 绘图
    # ======================================================================
    FONT = 9
    setup_matplotlib(FONT)

    fig1, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    ax1_0, ax1_3 = axes[0]
    ax1_1, ax1_2 = axes[1]

    # ---- ax1_0 : Boozer 面 -----------------------------------------------
    axes_before_boozer = set(fig1.axes)
    _, _ = plot_boozer_surface(
        eq_sc_off,
        ax=ax1_0,
        title_fontsize=FONT + 2,
        xlabel_fontsize=FONT,
        ylabel_fontsize=FONT,
        cmap="viridis",
    )
    for colorbar_ax in set(fig1.axes) - axes_before_boozer:
        colorbar_ax.remove()
    for text in list(ax1_0.texts):
        if "(T)" in text.get_text():
            text.remove()
    zeta_alpha0 = np.linspace(0.0, 2.0 * np.pi / nfp, 1000)
    theta_alpha0 = np.mod(iota_plot * zeta_alpha0, 2.0 * np.pi)
    ax1_0.plot(
        zeta_alpha0,
        theta_alpha0,
        color=COLORS["failure"],
        lw=1.6,
        ls="--",
    )
    ax1_0.set_title("", loc="center")
    ax1_0.set_title(r"(a) Boozer-field topology", loc="left")

    # 公共：横轴 pitch t∈(0,1)，纵轴 alpha-alpha0∈[-pi,pi)，同色系 viridis
    # 按 alpha-alpha0 单调排序，保证 contourf 网格有序
    CMAP = "cividis"
    order = np.argsort(alpha_plot)
    alpha_sorted = alpha_plot[order]
    J_hard_sorted = J_hard[order]
    J_soft_sorted = J_soft[order]
    well_split_sorted = well_split[order]  # (alpha, t) 分裂掩膜

    # 两图共用同一色标范围，便于硬/软对比
    VMAX = max(J_hard_sorted.max(), J_soft_sorted.max())
    VMIN = 0.0

    # ---- ax1_1 : 硬离散 J(alpha, t) 等高线 + 磁井分裂挖空 -----------------
    J_hard_plot = np.where(well_split_sorted, np.nan, J_hard_sorted)
    levels = np.linspace(VMIN, VMAX, 41)
    cf0 = ax1_1.contour(
        t_plot,
        alpha_sorted,
        J_hard_plot,
        levels=levels,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
    )
    with plt.rc_context({"hatch.color": "0.55", "hatch.linewidth": 0.6}):
        ax1_1.contourf(
            t_plot,
            alpha_sorted,
            well_split_sorted.astype(float),
            levels=[0.5, 1.5],
            colors=["#F0F0F0"],
            hatches=["///"],
            alpha=0.85,
        )
    ax1_1.set_xlabel(r"normalized pitch $t$")
    ax1_1.set_ylabel(r"$\alpha-\alpha_0$")
    ax1_1.set_xlim(0, 1)
    ax1_1.set_ylim(-np.pi, np.pi)
    ax1_1.axhline(0.0, color="white", lw=0.8, ls="--", alpha=0.8)
    ax1_1.set_title(r"(c) Single-branch hard $J(\alpha,t)$", loc="left")
    ax1_1.text(
        0.02,
        0.98,
        "hatched: multiple wells",
        transform=ax1_1.transAxes,
        ha="left",
        va="top",
        fontsize=FONT - 1,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    # ---- ax1_2 : 平滑 J(alpha, t) 等高线 --------------------------------
    cf1 = ax1_2.contour(
        t_plot,
        alpha_sorted,
        J_soft_sorted,
        levels=levels,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
    )
    ax1_2.set_xlabel(r"normalized pitch $t$")
    ax1_2.set_ylabel(r"$\alpha-\alpha_0$")
    ax1_2.set_xlim(0, 1)
    ax1_2.set_ylim(-np.pi, np.pi)
    ax1_2.axhline(0.0, color="white", lw=0.8, ls="--", alpha=0.8)
    ax1_2.set_title(r"(d) Global smooth proxy $\widetilde{J}(\alpha,t)$", loc="left")

    # ---- ax1_3 : 原 fig_02 的分裂磁井 -----------------------------------
    zeta_well, B_well, _, bd_well = field_line_B(
        eq_sc_off, rho=RHO, alpha=0.0, nzeta=2500
    )
    Bmin_well, Bmax_well = surface_B_range(bd_well)
    selected = None
    for t_candidate in np.linspace(0.05, 0.95, 181):
        B_star_candidate = Bmin_well + t_candidate * (Bmax_well - Bmin_well)
        mask_candidate = B_well < B_star_candidate
        components_candidate = count_periodic_true_components(mask_candidate)
        fill_fraction_candidate = float(np.mean(mask_candidate))
        if components_candidate < 2:
            continue
        score = (components_candidate, -abs(fill_fraction_candidate - 0.22))
        if selected is None or score > selected[0]:
            selected = (
                score,
                float(t_candidate),
                float(B_star_candidate),
                int(components_candidate),
                fill_fraction_candidate,
            )
    if selected is None:
        raise RuntimeError("No split-well pitch was found on the requested line.")
    _, t_well, B_star_well, components_well, fill_fraction_well = selected
    zeta_norm = zeta_well / (2.0 * np.pi / nfp)
    ax1_3.plot(zeta_norm, B_well, color="0.15", lw=2.0, label=r"$|B|(\zeta)$")
    ax1_3.axhline(
        B_star_well,
        color=COLORS["failure"],
        lw=1.6,
        ls="--",
        label=rf"$B^*(t_0={t_well:.2f})$",
    )
    for start, end in true_intervals(zeta_norm, B_well < B_star_well):
        ax1_3.axvspan(start, end, color=COLORS["failure"], alpha=0.12, lw=0)
    ax1_3.set_xlim(0.0, 1.0)
    ax1_3.set_xlabel(r"$\zeta /(2\pi/N_{\mathrm{FP}})$")
    ax1_3.set_ylabel(r"$|B|$")
    ax1_3.set_title(rf"(b) Split trapped region, $\rho={RHO:g}$", loc="left")
    ax1_3.grid(False)
    ax1_3.legend(loc="upper left")

    fig1.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.97, wspace=0.28, hspace=0.36)
    output = save_matplotlib(fig1, args.output)
    print(f"saved: {output}")
    fig1


if __name__ == "__main__":
    main()
