"""Analyze Poincare data and VMEC free-boundary equilibria against fixed-boundary.

Reads ``J_paper/data/freeb/poincare_vac.npz`` and VMEC ``wout_*.nc`` files,
computes boundary displacement, rotational transform comparison, magnetic well,
and hard-cutoff action invariance sigma_J(t), and saves a concise JSON summary
plus NPZ arrays for plotting.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.constants import mu_0
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from fig_common import boozer_spectral_data, data_path

FREEB_DIR = data_path("freeb")
EQ_VAC_PATH = data_path("equilibria", "coils_eq_tmp126.h5")
EQ_BETA4_PATH = data_path("equilibria", "balanced_beta4.12.h5")
COILS_PATH = data_path("coilsets", "tmp126_coils_n4_hard_bn_short19p2.h5")
MGRID_PATH = FREEB_DIR / "mgrid_tmp126_n4.nc"
POINCARE_PATH = FREEB_DIR / "poincare_vac.npz"
ANALYSIS_NPZ = FREEB_DIR / "freeb_analysis.npz"
SUMMARY_JSON = FREEB_DIR / "freeb_summary.json"


def eval_vmec_surface(wout_path: Path, phis: np.ndarray, thetas: np.ndarray, s_idx: int = -1):
    """Evaluate R, Z on VMEC flux surface ``s_idx`` at ``(phis, thetas)``."""
    with Dataset(wout_path) as ds:
        xm = np.asarray(ds["xm"][:], dtype=float)
        xn = np.asarray(ds["xn"][:], dtype=float)  # already includes nfp
        rmnc = np.asarray(ds["rmnc"][s_idx, :], dtype=float)
        zmns = np.asarray(ds["zmns"][s_idx, :], dtype=float)
    # VMEC convention: cos(m*theta - n*phi), sin(m*theta - n*phi)
    th = thetas[None, :, None]
    ph = phis[:, None, None]
    ang = xm[None, None, :] * th - xn[None, None, :] * ph
    R = np.sum(rmnc[None, None, :] * np.cos(ang), axis=-1)
    Z = np.sum(zmns[None, None, :] * np.sin(ang), axis=-1)
    return R, Z


def eval_vmec_axis(wout_path: Path, phis: np.ndarray):
    """Evaluate VMEC magnetic axis R(phi), Z(phi)."""
    with Dataset(wout_path) as ds:
        nfp = int(ds["nfp"][()])
        rac = np.asarray(ds["raxis_cc"][:], dtype=float)
        zas = np.asarray(ds["zaxis_cs"][:], dtype=float)
    n = np.arange(len(rac), dtype=float) * nfp
    ang = phis[:, None] * n[None, :]
    R = np.sum(rac[None, :] * np.cos(ang), axis=-1)
    Z = np.sum(-zas[None, :] * np.sin(ang), axis=-1)
    return R, Z


def read_vmec_scalars(wout_path: Path):
    """Extract convergence and global physics metrics from VMEC wout."""
    with Dataset(wout_path) as ds:
        signgs = int(ds["signgs"][()])
        ns = int(ds["ns"][()])
        s_full = np.linspace(0.0, 1.0, ns)
        rho_full = np.sqrt(s_full)
        iotaf = np.asarray(ds["iotaf"][:], dtype=float) * signgs
        vp = np.asarray(ds["vp"][:], dtype=float)
        return dict(
            ier_flag=int(ds["ier_flag"][()]),
            fsqr=float(ds["fsqr"][()]),
            fsqz=float(ds["fsqz"][()]),
            fsql=float(ds["fsql"][()]),
            niter=int(ds["niter"][()]),
            ns=ns,
            signgs=signgs,
            rbtor=float(ds["rbtor"][()]),
            wb=float(ds["wb"][()]),
            wp=float(ds["wp"][()]),
            betatotal=float(ds["betatotal"][()]),
            volume_p=float(ds["volume_p"][()]),
            Rmajor_p=float(ds["Rmajor_p"][()]),
            Aminor_p=float(ds["Aminor_p"][()]),
            aspect=float(ds["aspect"][()]),
            volavgB=float(ds["volavgB"][()]),
            phiedge=float(ds["phi"][-1]),
            rho=rho_full,
            iota=iotaf,
            vp=vp,
        )


def lcfs_displacement(eq_fixed, wout_path: Path, n_phi: int = 64, n_theta: int = 256):
    """Compute cross-sectional & 3D LCFS displacement between DESC fixed-b and VMEC free-b."""
    from desc.grid import LinearGrid

    nfp = int(eq_fixed.NFP)
    phis = np.linspace(0.0, 2.0 * np.pi / nfp, n_phi, endpoint=False)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    # Evaluate VMEC LCFS on identical phi planes
    R_v, Z_v = eval_vmec_surface(wout_path, phis, thetas, s_idx=-1)
    # Evaluate DESC LCFS on identical phi planes
    R_d = np.zeros_like(R_v)
    Z_d = np.zeros_like(Z_v)
    for ip, phi_val in enumerate(phis):
        g = LinearGrid(rho=np.array([1.0]), theta=thetas, zeta=np.array([phi_val]), NFP=nfp)
        d = eq_fixed.compute(["R", "Z"], grid=g)
        R_d[ip] = np.asarray(d["R"], dtype=float)
        Z_d[ip] = np.asarray(d["Z"], dtype=float)

    # 2D cross-section surface distance at each toroidal plane (exact normal-plane metric)
    d_plane_all = []
    for ip in range(n_phi):
        tree_v = cKDTree(np.column_stack([R_v[ip], Z_v[ip]]))
        d1, _ = tree_v.query(np.column_stack([R_d[ip], Z_d[ip]]))
        tree_d = cKDTree(np.column_stack([R_d[ip], Z_d[ip]]))
        d2, _ = tree_d.query(np.column_stack([R_v[ip], Z_v[ip]]))
        d_plane_all.append(0.5 * (d1 + d2))
    d_plane_all = np.concatenate(d_plane_all)

    # Also true 3D surface distance using full-torus VMEC point cloud
    phis_full = np.linspace(0.0, 2.0 * np.pi, n_phi * nfp, endpoint=False)
    R_vf, Z_vf = eval_vmec_surface(wout_path, phis_full, thetas, s_idx=-1)
    X_vf = R_vf * np.cos(phis_full[:, None])
    Y_vf = R_vf * np.sin(phis_full[:, None])
    tree_3d = cKDTree(np.column_stack([X_vf.ravel(), Y_vf.ravel(), Z_vf.ravel()]))
    X_d = R_d * np.cos(phis[:, None])
    Y_d = R_d * np.sin(phis[:, None])
    d_3d, _ = tree_3d.query(np.column_stack([X_d.ravel(), Y_d.ravel(), Z_d.ravel()]))

    a_fixed = float(eq_fixed.compute("a")["a"])
    return dict(
        plane_mean_mm=float(np.mean(d_plane_all) * 1e3),
        plane_rms_mm=float(np.sqrt(np.mean(d_plane_all**2)) * 1e3),
        plane_max_mm=float(np.max(d_plane_all) * 1e3),
        dist3d_mean_mm=float(np.mean(d_3d) * 1e3),
        dist3d_rms_mm=float(np.sqrt(np.mean(d_3d**2)) * 1e3),
        dist3d_max_mm=float(np.max(d_3d) * 1e3),
        dist3d_mean_over_a=float(np.mean(d_3d) / a_fixed),
        dist3d_max_over_a=float(np.max(d_3d) / a_fixed),
    )


def compute_hard_J_sigma_eq(eq, rho: float = 1.0, num_alpha: int = 96, nt: int = 61, nzeta: int = 2000):
    """Same hard-cutoff sigma_J(t) metric as make_fig11_hard_action.py."""
    from desc.backend import jnp
    from softplus_j import _boozer_B_star_from_t
    from jax import vmap

    try:
        from jax.numpy import trapezoid as _trapz
    except ImportError:
        from jax.numpy import trapz as _trapz

    nfp = eq.NFP
    bd = boozer_spectral_data(eq, rho=rho)
    basis = bd["basis"]
    grid_b = bd["grid"]
    rho_arr = jnp.asarray(bd["rho"])
    iota = jnp.asarray(bd["iota"])
    coeff_B = jnp.asarray(bd["coeff_B"])
    coeff_g = jnp.asarray(bd["coeff_g"])

    zeta_max = 2.0 * np.pi / nfp
    zeta = jnp.linspace(0.0, zeta_max, nzeta)
    t = np.linspace(0.01, 0.99, nt)

    A_full = basis.evaluate(grid_b.nodes[:: grid_b.num_rho])
    B_full = coeff_B @ A_full.T
    B_star = _boozer_B_star_from_t(jnp.min(B_full, axis=1), jnp.max(B_full, axis=1), jnp.asarray(t))

    def one_surface(r_it, cB, cg, B_star_s):
        r_, it_ = r_it
        a0 = jnp.pi - it_ * (jnp.pi / nfp)
        alpha_1d = jnp.linspace(-jnp.pi, 0.0, num_alpha, endpoint=False) + a0
        theta = alpha_1d[:, None] + it_ * zeta[None, :]
        nodes = jnp.stack(
            (jnp.broadcast_to(r_, theta.shape), theta, jnp.broadcast_to(zeta[None, :], theta.shape)),
            axis=-1,
        ).reshape((-1, 3))
        mat = basis.evaluate(nodes)
        B = (mat @ cB).reshape((num_alpha, nzeta))
        g = (mat @ cg).reshape((num_alpha, nzeta))
        cutoff = jnp.maximum(1.0 - B[None, :, :] / B_star_s[:, None, None], 0.0)
        integrand = jnp.sqrt(cutoff) * B[None, :, :] * jnp.abs(g[None, :, :])
        return _trapz(integrand, zeta, axis=-1)

    J_hard = np.asarray(vmap(one_surface)((rho_arr, iota), coeff_B, coeff_g, B_star))[0]
    sigma = (np.max(J_hard, axis=1) - np.min(J_hard, axis=1)) / (np.mean(J_hard, axis=1) + 1e-30)
    return t, sigma


def run_xbooz_xform(wout_path: Path, mboz: int = 24, nboz: int = 24) -> Path:
    """Run STELLOPT xbooz_xform directly on a converged VMEC wout file."""
    wdir = wout_path.parent
    suf = wout_path.stem.replace("wout_", "")
    booz_nc = wdir / f"boozmn_{suf}.nc"
    with Dataset(wout_path) as ds:
        ns = int(ds["ns"][()])
    surfs = [ns // 4, ns // 2, 3 * ns // 4, ns - 1, ns]
    in_booz = wdir / f"in_booz.{suf}"
    in_booz.write_text(
        f"{mboz} {nboz}\n{suf}\n" + " ".join(str(s) for s in surfs) + "\n"
    )
    proc = subprocess.run(
        ["/opt/bin/xbooz_xform", f"in_booz.{suf}", "F"],
        cwd=wdir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not booz_nc.is_file():
        raise RuntimeError(
            f"xbooz_xform failed on {wout_path}: {proc.stdout[-300:]} {proc.stderr[-300:]}"
        )
    return booz_nc


def compute_hard_J_sigma_boozmn(
    booz_nc: Path, s_idx: int = -1, num_alpha: int = 96, nt: int = 61, nzeta: int = 2000
):
    """Exact hard-cutoff sigma_J(t) directly from xbooz_xform Boozer spectrum.

    In Boozer coordinates, B * |sqrt(g)_B| = |G + iota*I| / B where |G + iota*I|
    is a flux-surface constant that cancels identically in sigma_J(t).
    """
    with Dataset(booz_nc) as ds:
        nfp = int(ds["nfp_b"][()])
        xm = np.asarray(ds["ixm_b"][:], dtype=float)
        xn = np.asarray(ds["ixn_b"][:], dtype=float)
        jlist = np.asarray(ds["jlist"][:], dtype=int)
        bmnc = np.asarray(ds["bmnc_b"][s_idx, :], dtype=float)
        j_s = int(jlist[s_idx]) - 1
        iota = float(ds["iota_b"][j_s])

    th_g = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    ze_g = np.linspace(0.0, 2.0 * np.pi / nfp, 128, endpoint=False)
    TT, ZZ = np.meshgrid(th_g, ze_g, indexing="ij")
    ang_g = xm[None, :] * TT.ravel()[:, None] - xn[None, :] * ZZ.ravel()[:, None]
    B_g = np.cos(ang_g) @ bmnc
    B_min, B_max = float(B_g.min()), float(B_g.max())

    t = np.linspace(0.01, 0.99, nt)
    B_star = B_min + t * (B_max - B_min)
    zeta = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta)
    alpha = np.linspace(0.0, 2.0 * np.pi, num_alpha, endpoint=False)

    J_hard = np.zeros((nt, num_alpha))
    for ia, a_val in enumerate(alpha):
        th = a_val + iota * zeta
        ang = xm[None, :] * th[:, None] - xn[None, :] * zeta[:, None]
        B = np.cos(ang) @ bmnc
        cutoff = np.maximum(1.0 - B[None, :] / B_star[:, None], 0.0)
        integrand = np.sqrt(cutoff) / B[None, :]
        J_hard[:, ia] = np.trapezoid(integrand, zeta, axis=-1)

    sigma = (np.max(J_hard, axis=1) - np.min(J_hard, axis=1)) / (
        np.mean(J_hard, axis=1) + 1e-30
    )
    return t, sigma


def compute_vmec_magnetic_well(wout_path: Path):
    """Landreman & Jorge (2020) magnetic well directly from VMEC wout."""
    with Dataset(wout_path) as ds:
        ns = int(ds["ns"][()])
        nfp = int(ds["nfp"][()])
        vp = np.asarray(ds["vp"][:], dtype=float)
        pres = np.asarray(ds["pres"][:], dtype=float)
        xm_nyq = np.asarray(ds["xm_nyq"][:], dtype=float)
        xn_nyq = np.asarray(ds["xn_nyq"][:], dtype=float)
        bmnc = np.asarray(ds["bmnc"][:], dtype=float)
        gmnc = np.asarray(ds["gmnc"][:], dtype=float)

    th = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    ze = np.linspace(0.0, 2.0 * np.pi / nfp, 32, endpoint=False)
    TT, ZZ = np.meshgrid(th, ze, indexing="ij")
    ang = xm_nyq[None, :] * TT.ravel()[:, None] - xn_nyq[None, :] * ZZ.ravel()[:, None]
    cos_a = np.cos(ang)
    B = bmnc[1:, :] @ cos_a.T
    g = np.abs(gmnc[1:, :] @ cos_a.T)
    B2_avg = np.sum(g * B**2, axis=1) / np.sum(g, axis=1)
    s_h = (np.arange(1, ns) - 0.5) / (ns - 1)
    ds_step = 1.0 / (ns - 1)
    V_h = np.cumsum(vp[1:]) * ds_step - 0.5 * vp[1:] * ds_step
    dp_ds = np.gradient(pres[1:], s_h, edge_order=2)
    dB2_ds = np.gradient(B2_avg, s_h, edge_order=2)
    W = V_h * (2.0 * mu_0 * dp_ds + dB2_ds) / (vp[1:] * B2_avg)
    return np.sqrt(s_h), W


def verify_mgrid_accuracy(eq_vac, coils):
    """Compare mgrid interpolation against direct Biot-Savart on the LCFS."""
    from desc.grid import LinearGrid

    with Dataset(MGRID_PATH) as ds:
        nfp = int(ds["nfp"][()])
        rmin, rmax = float(ds["rmin"][()]), float(ds["rmax"][()])
        zmin, zmax = float(ds["zmin"][()]), float(ds["zmax"][()])
        ir, jz, kp = int(ds["ir"][()]), int(ds["jz"][()]), int(ds["kp"][()])
        br = np.asarray(ds["br_001"][:], dtype=float)
        bp = np.asarray(ds["bp_001"][:], dtype=float)
        bz = np.asarray(ds["bz_001"][:], dtype=float)
    r_ax = np.linspace(rmin, rmax, ir)
    z_ax = np.linspace(zmin, zmax, jz)
    p_ax = np.linspace(0.0, 2.0 * np.pi / nfp, kp + 1)
    # Periodic pad in phi
    br_p = np.concatenate([br, br[:1]], axis=0)
    bp_p = np.concatenate([bp, bp[:1]], axis=0)
    bz_p = np.concatenate([bz, bz[:1]], axis=0)
    itp_r = RegularGridInterpolator((p_ax, z_ax, r_ax), br_p)
    itp_p = RegularGridInterpolator((p_ax, z_ax, r_ax), bp_p)
    itp_z = RegularGridInterpolator((p_ax, z_ax, r_ax), bz_p)

    lcfs = LinearGrid(rho=1.0, M=24, N=24, NFP=nfp)
    d = eq_vac.compute(["R", "phi", "Z"], grid=lcfs)
    pts_rpz = np.column_stack([d["R"], d["phi"], d["Z"]])
    pts_pzr = np.column_stack([np.mod(d["phi"], 2.0 * np.pi / nfp), d["Z"], d["R"]])
    B_mg = np.column_stack([itp_r(pts_pzr), itp_p(pts_pzr), itp_z(pts_pzr)])
    B_bs = np.asarray(
        coils.compute_magnetic_field(pts_rpz, basis="rpz", source_grid=LinearGrid(N=101)),
        dtype=float,
    )
    rel_err = np.linalg.norm(B_mg - B_bs, axis=1) / np.linalg.norm(B_bs, axis=1)
    return dict(
        mgrid_lcfs_rel_mean=float(np.mean(rel_err)),
        mgrid_lcfs_rel_max=float(np.max(rel_err)),
        has_nonfinite=bool(not (np.isfinite(br).all() and np.isfinite(bp).all() and np.isfinite(bz).all())),
    )


def main():
    from desc.compat import flip_helicity
    from desc.examples import get
    from desc.grid import LinearGrid
    from desc.io import load

    t0 = time.perf_counter()
    eq_vac = load(str(EQ_VAC_PATH))
    eq_b4 = load(str(EQ_BETA4_PATH))
    coils = load(str(COILS_PATH))
    nfp = int(eq_vac.NFP)

    summary = {}
    if MGRID_PATH.is_file():
        summary["mgrid_check"] = verify_mgrid_accuracy(eq_vac, coils)
        print("Mgrid check:", summary["mgrid_check"])

    planes = np.array([0.0, np.pi / (2.0 * nfp), np.pi / nfp])
    thetas_plot = np.linspace(0.0, 2.0 * np.pi, 361)
    npz_data = dict(planes=planes, thetas=thetas_plot)

    # W7-X reference hard-J curve (matching make_fig11_hard_action.py)
    eq_w7x = get("W7-X")
    flip_helicity(eq_w7x)
    t_w7x, sig_w7x = compute_hard_J_sigma_eq(eq_w7x, rho=1.0)
    npz_data["w7x_tJ"] = t_w7x
    npz_data["w7x_sigmaJ"] = sig_w7x

    # Fixed-boundary reference profiles & hard J
    g_rad = LinearGrid(rho=np.linspace(0.0, 1.0, 51), M=1, N=1, NFP=nfp)
    rho_grid = g_rad.compress(g_rad.nodes[:, 0])
    npz_data["rho_grid"] = rho_grid
    for tag, eq in [("fixed_vac", eq_vac), ("fixed_beta4", eq_b4)]:
        dr = eq.compute(["iota", "magnetic well"], grid=g_rad)
        iota_arr = np.asarray(g_rad.compress(dr["iota"]), dtype=float)
        well_arr = np.asarray(g_rad.compress(dr["magnetic well"]), dtype=float)
        t_j, sig_j = compute_hard_J_sigma_eq(eq, rho=1.0)
        npz_data[f"{tag}_iota"] = iota_arr
        npz_data[f"{tag}_well"] = well_arr
        npz_data[f"{tag}_tJ"] = t_j
        npz_data[f"{tag}_sigmaJ"] = sig_j
        summary[tag] = dict(
            R0=float(eq.compute("R0")["R0"]),
            a=float(eq.compute("a")["a"]),
            aspect=float(eq.compute("R0/a")["R0/a"]),
            V=float(eq.compute("V")["V"]),
            beta_pct=float(eq.compute("<beta>_vol")["<beta>_vol"]) * 100.0,
            iota_axis=float(iota_arr[0]),
            iota_edge=float(iota_arr[-1]),
            edge_well=float(well_arr[-1]),
            sigmaJ_median_mid=float(np.median(sig_j[(t_j >= 0.2) & (t_j <= 0.8)])),
            sigmaJ_t05=float(np.interp(0.5, t_j, sig_j)),
        )

    # Analyze VMEC free-boundary runs using xbooz_xform (exact, no Zernike refit error)
    vmec_cases = [
        ("freeb_vac", FREEB_DIR / "vmec_vac" / "wout_vac.nc", eq_vac),
        ("freeb_beta4", FREEB_DIR / "vmec_beta4.12" / "wout_beta4.12.nc", eq_b4),
        ("freeb_beta4_phiadj", FREEB_DIR / "vmec_beta4.12_phiadj" / "wout_beta4.12_phiadj.nc", eq_b4),
    ]
    for tag, wout, eq_ref in vmec_cases:
        if not wout.is_file():
            print(f"Skipping missing {wout}")
            continue
        sc = read_vmec_scalars(wout)
        disp = lcfs_displacement(eq_ref, wout)
        R_lcfs, Z_lcfs = eval_vmec_surface(wout, planes, thetas_plot, s_idx=-1)
        R_ax, Z_ax = eval_vmec_axis(wout, planes)
        rho_w, well_fb = compute_vmec_magnetic_well(wout)
        booz_nc = run_xbooz_xform(wout, mboz=24, nboz=24)
        t_j, sig_j = compute_hard_J_sigma_boozmn(booz_nc, s_idx=-1)
        _, sig_j_mid = compute_hard_J_sigma_boozmn(booz_nc, s_idx=1)

        # Iota diff vs fixed-b reference
        iota_ref = np.interp(sc["rho"], rho_grid, npz_data[f"{'fixed_vac' if 'vac' in tag else 'fixed_beta4'}_iota"])
        diota = sc["iota"] - iota_ref

        npz_data[f"{tag}_R_lcfs"] = R_lcfs
        npz_data[f"{tag}_Z_lcfs"] = Z_lcfs
        npz_data[f"{tag}_R_ax"] = R_ax
        npz_data[f"{tag}_Z_ax"] = Z_ax
        npz_data[f"{tag}_rho"] = sc["rho"]
        npz_data[f"{tag}_iota"] = sc["iota"]
        npz_data[f"{tag}_well_rho"] = rho_w
        npz_data[f"{tag}_well"] = well_fb
        npz_data[f"{tag}_tJ"] = t_j
        npz_data[f"{tag}_sigmaJ"] = sig_j
        npz_data[f"{tag}_sigmaJ_s05"] = sig_j_mid

        summary[tag] = dict(
            ier_flag=sc["ier_flag"],
            fsqr=sc["fsqr"],
            fsqz=sc["fsqz"],
            fsql=sc["fsql"],
            rbtor=sc["rbtor"],
            phiedge=sc["phiedge"],
            beta_pct=sc["betatotal"] * 100.0,
            R0=sc["Rmajor_p"],
            a=sc["Aminor_p"],
            aspect=sc["aspect"],
            V=sc["volume_p"],
            iota_axis=float(sc["iota"][0]),
            iota_edge=float(sc["iota"][-1]),
            diota_axis=float(diota[0]),
            diota_edge=float(diota[-1]),
            diota_max_abs=float(np.max(np.abs(diota))),
            well_s05=float(np.interp(np.sqrt(0.5), rho_w, well_fb)),
            well_s095=float(np.interp(np.sqrt(0.95), rho_w, well_fb)),
            edge_well=float(well_fb[-1]),
            sigmaJ_median_mid=float(np.median(sig_j[(t_j >= 0.2) & (t_j <= 0.8)])),
            sigmaJ_t05=float(np.interp(0.5, t_j, sig_j)),
            sigmaJ_s05_median_mid=float(np.median(sig_j_mid[(t_j >= 0.2) & (t_j <= 0.8)])),
            lcfs_displacement=disp,
        )
        print(f"=== {tag} ===")
        print(json.dumps(summary[tag], indent=2))

    # Include Poincare summary metrics if available
    if POINCARE_PATH.is_file():
        pc = np.load(POINCARE_PATH)
        confined = pc["confined"]
        fracs = pc["fracs"]
        iota_v = pc["iota_vac"]
        rho_eff = pc["rho_eff"]
        in_mask = fracs <= 1.0001
        in_conf = confined[in_mask]
        # Interpolate fixed-b iota at rho_eff of interior confined lines
        eq_iota_interp = np.interp(rho_eff[in_mask & confined], pc["eq_rho_iota"], pc["eq_iota"])
        diota_pc = iota_v[in_mask & confined] - eq_iota_interp
        summary["poincare_vac"] = dict(
            n_fp_trace=int(pc["n_fp_trace"]),
            source_n=int(pc["source_n"]),
            bs_conv_mm=float(pc["bs_conv_mm"]),
            r_ax_vac_phi0=float(pc["r_ax_vac"]),
            r_ax_eq_phi0=float(pc["eq_ax_R"][0]),
            axis_shift_phi0_mm=float((pc["r_ax_vac"] - pc["eq_ax_R"][0]) * 1e3),
            axis_closure_err_m=float(pc["axis_closure_err"]),
            interior_confined=f"{int(np.sum(in_conf))}/{len(in_conf)}",
            total_confined=f"{int(np.sum(confined))}/{len(confined)}",
            max_confined_frac=float(np.max(fracs[confined])) if np.any(confined) else 0.0,
            iota_vac_min=float(np.nanmin(iota_v[in_mask & confined])),
            iota_vac_max=float(np.nanmax(iota_v[in_mask & confined])),
            diota_vs_fixed_mean=float(np.mean(diota_pc)),
            diota_vs_fixed_max_abs=float(np.max(np.abs(diota_pc))),
        )
        print("=== poincare_vac ===")
        print(json.dumps(summary["poincare_vac"], indent=2))

    np.savez(ANALYSIS_NPZ, **npz_data)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Saved {ANALYSIS_NPZ} and {SUMMARY_JSON} in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
