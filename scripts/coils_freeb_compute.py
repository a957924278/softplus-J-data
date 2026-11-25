"""Compute vacuum Poincare sections, mgrid, and VMEC free-boundary equilibria.

Saves all computed artifacts under ``J_paper/data/freeb/`` so plotting scripts
can render publication figures cleanly without rerunning heavy GPU/MPI jobs.
"""

from __future__ import annotations

import os
import sys

# Ensure GPU device selection happens before any desc/jax submodule import when
# running GPU steps (poincare / mgrid).
_NEED_GPU = any(arg in ("poincare", "mgrid", "all") for arg in sys.argv[1:])
if _NEED_GPU and os.environ.get("JAX_PLATFORMS", "").lower() != "cpu":
    from desc import set_device

    set_device("gpu")

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from fig_common import data_path

FREEB_DIR = data_path("freeb")
EQ_VAC_PATH = data_path("equilibria", "coils_eq_tmp126.h5")
EQ_BETA4_PATH = data_path("equilibria", "balanced_beta4.12.h5")
COILS_PATH = data_path("coilsets", "tmp126_coils_n4_hard_bn_short19p2.h5")
MGRID_PATH = FREEB_DIR / "mgrid_tmp126_n4.nc"
POINCARE_PATH = FREEB_DIR / "poincare_vac.npz"
SUMMARY_PATH = FREEB_DIR / "freeb_summary.json"

SOURCE_N = 101
MGRID_BOX = dict(Rmin=1.42, Rmax=2.48, Zmin=-0.55, Zmax=0.55, nR=121, nZ=121, nphi=72)


def _trace_lines(coils, r0, z0, phis, source_n=SOURCE_N, rtol=1e-9, atol=1e-9):
    from desc.grid import LinearGrid
    from desc.magnetic_fields import field_line_integrate

    r, z = field_line_integrate(
        r0=np.asarray(r0, dtype=float),
        z0=np.asarray(z0, dtype=float),
        phis=np.asarray(phis, dtype=float),
        field=coils,
        source_grid=LinearGrid(N=source_n),
        rtol=rtol,
        atol=atol,
        bounds_R=(1.25, 2.65),
        bounds_Z=(-0.65, 0.65),
        use_precomputed_source=True,
    )
    return np.asarray(r, dtype=float), np.asarray(z, dtype=float)


def find_vacuum_axis(coils, eq_vac):
    """Find exact stellarator-symmetric magnetic axis R at phi=0 via Z(pi/NFP)=0."""
    from desc.grid import LinearGrid

    nfp = int(eq_vac.NFP)
    r_guess = float(
        eq_vac.compute("R", grid=LinearGrid(rho=0.0, theta=0.0, zeta=0.0, NFP=nfp))[
            "R"
        ][0]
    )
    phis_half = np.array([0.0, np.pi / nfp])

    def z_half(r_val):
        _, z_out = _trace_lines(coils, [r_val], [0.0], phis_half, rtol=1e-10, atol=1e-10)
        return float(z_out[-1, 0])

    r_lo, r_hi = r_guess - 0.04, r_guess + 0.04
    z_lo, z_hi = z_half(r_lo), z_half(r_hi)
    if z_lo * z_hi > 0:
        raise RuntimeError(
            f"Axis root bracket failed: Z({r_lo})={z_lo:.4e}, Z({r_hi})={z_hi:.4e}"
        )
    r_ax = float(brentq(z_half, r_lo, r_hi, xtol=1e-10, rtol=1e-10))

    # Trace 1 full field period at the 4 quarter-period planes to verify closure
    phis_fp = np.linspace(0.0, 2.0 * np.pi / nfp, 5)
    r_fp, z_fp = _trace_lines(coils, [r_ax], [0.0], phis_fp, rtol=1e-10, atol=1e-10)
    closure_err = float(np.hypot(r_fp[-1, 0] - r_ax, z_fp[-1, 0]))
    print(
        f"Vacuum axis phi=0: R={r_ax:.7f} m (eq={r_guess:.7f} m, "
        f"shift={(r_ax - r_guess)*1e3:+.3f} mm), 1-FP closure={closure_err:.2e} m"
    )
    return r_ax, r_fp[:4, 0], z_fp[:4, 0], closure_err


def compute_poincare(n_fp: int = 400):
    """Run direct Biot-Savart vacuum Poincare trace and compute surface metrics."""
    from desc.grid import LinearGrid
    from desc.io import load

    t0 = time.perf_counter()
    FREEB_DIR.mkdir(parents=True, exist_ok=True)
    eq_vac = load(str(EQ_VAC_PATH))
    coils = load(str(COILS_PATH))
    nfp = int(eq_vac.NFP)

    r_ax, r_ax_planes, z_ax_planes, closure_err = find_vacuum_axis(coils, eq_vac)

    # Fixed-boundary LCFS & interior contours at phi = 0, pi/(2*NFP), pi/NFP
    planes = np.array([0.0, np.pi / (2.0 * nfp), np.pi / nfp])
    rhos_eq = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    n_theta_plot = 361
    theta_plot = np.linspace(0.0, 2.0 * np.pi, n_theta_plot)
    eq_R = np.zeros((len(planes), len(rhos_eq), n_theta_plot))
    eq_Z = np.zeros((len(planes), len(rhos_eq), n_theta_plot))
    eq_ax_R = np.zeros(len(planes))
    eq_ax_Z = np.zeros(len(planes))
    for ip, phi_val in enumerate(planes):
        g0 = LinearGrid(rho=0.0, theta=0.0, zeta=np.array([phi_val]), NFP=nfp)
        d0 = eq_vac.compute(["R", "Z"], grid=g0)
        eq_ax_R[ip], eq_ax_Z[ip] = float(d0["R"][0]), float(d0["Z"][0])
        for ir, rho_val in enumerate(rhos_eq):
            g1 = LinearGrid(
                rho=np.array([rho_val]),
                theta=theta_plot,
                zeta=np.array([phi_val]),
                NFP=nfp,
            )
            d1 = eq_vac.compute(["R", "Z"], grid=g1)
            eq_R[ip, ir] = np.asarray(d1["R"], dtype=float)
            eq_Z[ip, ir] = np.asarray(d1["Z"], dtype=float)

    # Outboard LCFS radius at phi=0, Z=0
    r_lcfs_out = float(np.max(eq_R[0, -1]))
    dr_out = r_lcfs_out - r_ax
    fracs_in = np.array(
        [0.06, 0.14, 0.22, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.85, 0.91, 0.95, 0.98, 1.00]
    )
    fracs_out = np.array([1.02, 1.05, 1.08, 1.12, 1.16, 1.22, 1.30, 1.40])
    fracs = np.concatenate([fracs_in, fracs_out])
    r0 = r_ax + fracs * dr_out
    z0 = np.zeros_like(r0)

    print(
        f"Tracing {len(r0)} vacuum field lines for {n_fp} field periods "
        f"({n_fp / nfp:.0f} toroidal turns) with source_N={SOURCE_N}..."
    )
    # 4 quarter-period saves per field period -> idx%4==0,1,2 give phi=0, pi/8, pi/4
    phis = np.linspace(0.0, n_fp * 2.0 * np.pi / nfp, n_fp * 4 + 1)
    R_all, Z_all = _trace_lines(coils, r0, z0, phis, source_n=SOURCE_N)
    # Shape: (n_steps, n_lines)
    R_p0 = R_all[0::4]
    Z_p0 = Z_all[0::4]
    R_p1 = R_all[1::4]
    Z_p1 = Z_all[1::4]
    R_p2 = R_all[2::4]
    Z_p2 = Z_all[2::4]

    # Compute rotational transform and enclosed area for each confined line
    confined = np.all(np.isfinite(R_all), axis=0)
    iota_vac = np.full(len(r0), np.nan)
    area_p0 = np.full(len(r0), np.nan)
    # Fixed-boundary LCFS cross-sectional area at phi=0 (shoelace)
    x_lc, y_lc = eq_R[0, -1], eq_Z[0, -1]
    area_lcfs = 0.5 * abs(
        float(np.sum(x_lc[:-1] * y_lc[1:] - x_lc[1:] * y_lc[:-1]))
    )

    # Check DESC poloidal angle sign vs cylindrical atan2(Z-Zax, R-Rax)
    # At every quarter-period step, unwrap angle relative to vacuum axis at that plane
    ax_R_seq = np.tile(r_ax_planes, n_fp + 1)[: len(phis)]
    ax_Z_seq = np.tile(z_ax_planes, n_fp + 1)[: len(phis)]
    for i in range(len(r0)):
        if not confined[i]:
            continue
        ang_seq = np.arctan2(Z_all[:, i] - ax_Z_seq, R_all[:, i] - ax_R_seq)
        ang_unw = np.unwrap(ang_seq)
        # Linear least-squares fit of unwrapped poloidal angle vs phi
        slope, _ = np.polyfit(phis, ang_unw, 1)
        iota_vac[i] = abs(float(slope))
        # Enclosed area at phi=0 by sorting punctures in angle around (r_ax, 0)
        th0 = np.arctan2(Z_p0[:, i], R_p0[:, i] - r_ax)
        order = np.argsort(th0)
        xs = np.append(R_p0[order, i], R_p0[order[0], i])
        ys = np.append(Z_p0[order, i], Z_p0[order[0], i])
        area_p0[i] = 0.5 * abs(float(np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1])))

    rho_eff = np.sqrt(area_p0 / area_lcfs)

    # LCFS match metrics for the f=1.00 line (and best-matching confined line)
    idx_lcfs = int(np.argmin(np.abs(fracs - 1.0)))
    lcfs_dist_mm = {}
    for ip, (pname, Rp, Zp) in enumerate(
        [("phi0", R_p0, Z_p0), ("phi_pi8", R_p1, Z_p1), ("phi_pi4", R_p2, Z_p2)]
    ):
        tree = cKDTree(np.column_stack([eq_R[ip, -1], eq_Z[ip, -1]]))
        pts = np.column_stack([Rp[:, idx_lcfs], Zp[:, idx_lcfs]])
        valid = np.all(np.isfinite(pts), axis=1)
        if np.any(valid):
            d, _ = tree.query(pts[valid])
            lcfs_dist_mm[pname] = dict(
                mean=float(np.mean(d) * 1e3), max=float(np.max(d) * 1e3)
            )
        else:
            lcfs_dist_mm[pname] = dict(mean=float("nan"), max=float("nan"))

    # Resolution convergence check: SOURCE_N=101 vs 201 on 3 surfaces over 80 FPs
    test_idx = [4, 11, idx_lcfs]
    phis_chk = np.linspace(0.0, 80 * 2.0 * np.pi / nfp, 81)
    R101, Z101 = _trace_lines(coils, r0[test_idx], z0[test_idx], phis_chk, source_n=101)
    R201, Z201 = _trace_lines(coils, r0[test_idx], z0[test_idx], phis_chk, source_n=201)
    conv_mm = float(np.nanmax(np.hypot(R101 - R201, Z101 - Z201)) * 1e3)
    print(f"Biot-Savart convergence check (N=101 vs 201 over 80 FPs): max diff = {conv_mm:.4f} mm")

    # Trace 4 trajectories inside the exterior n/m = 4/4 = 1 edge island chain
    # (launched near its O-point at phi = pi/NFP, Z = 0) so the edge topology is
    # cleanly resolved across all three cross-sections.
    r0_isl = np.array([2.23, 2.26, 2.31, 2.34])
    z0_isl = np.zeros_like(r0_isl)
    phis_isl = np.pi / nfp + np.linspace(0.0, n_fp * 2.0 * np.pi / nfp, n_fp * 4 + 1)
    R_isl, Z_isl = _trace_lines(coils, r0_isl, z0_isl, phis_isl, source_n=SOURCE_N)
    # phis_isl starts at pi/4: 0::4 -> phi=pi/4, 2::4 -> phi=0 (mod 2pi/NFP), 3::4 -> phi=pi/8 (mod 2pi/NFP)
    isl_R_p0, isl_Z_p0 = R_isl[2::4], Z_isl[2::4]
    isl_R_p1, isl_Z_p1 = R_isl[3::4], Z_isl[3::4]
    isl_R_p2, isl_Z_p2 = R_isl[0::4], Z_isl[0::4]

    # Save fixed-boundary iota profile for direct comparison
    g_rad = LinearGrid(rho=np.linspace(0.0, 1.0, 41), M=1, N=1, NFP=nfp)
    d_rad = eq_vac.compute(["iota"], grid=g_rad)
    eq_rho_iota = g_rad.compress(g_rad.nodes[:, 0])
    eq_iota = g_rad.compress(d_rad["iota"])

    np.savez(
        POINCARE_PATH,
        nfp=nfp,
        n_fp_trace=n_fp,
        source_n=SOURCE_N,
        planes=planes,
        r_ax_vac=r_ax,
        r_ax_vac_planes=r_ax_planes,
        z_ax_vac_planes=z_ax_planes,
        axis_closure_err=closure_err,
        eq_ax_R=eq_ax_R,
        eq_ax_Z=eq_ax_Z,
        rhos_eq=rhos_eq,
        eq_R=eq_R,
        eq_Z=eq_Z,
        fracs=fracs,
        r0=r0,
        z0=z0,
        confined=confined,
        R_p0=R_p0,
        Z_p0=Z_p0,
        R_p1=R_p1,
        Z_p1=Z_p1,
        R_p2=R_p2,
        Z_p2=Z_p2,
        isl_R_p0=isl_R_p0,
        isl_Z_p0=isl_Z_p0,
        isl_R_p1=isl_R_p1,
        isl_Z_p1=isl_Z_p1,
        isl_R_p2=isl_R_p2,
        isl_Z_p2=isl_Z_p2,
        iota_vac=iota_vac,
        rho_eff=rho_eff,
        area_p0=area_p0,
        area_lcfs=area_lcfs,
        eq_rho_iota=eq_rho_iota,
        eq_iota=eq_iota,
        bs_conv_mm=conv_mm,
    )
    print(f"Saved {POINCARE_PATH} in {time.perf_counter() - t0:.1f} s")
    print(f"Confined lines: {int(np.sum(confined))}/{len(r0)}")
    for i in range(len(r0)):
        status = "CONFINED" if confined[i] else "LOST"
        print(
            f"  line {i:2d}: f={fracs[i]:.2f}, R0={r0[i]:.5f} -> "
            f"{status:8s}, rho_eff={rho_eff[i]:.4f}, iota_vac={iota_vac[i]:.5f}"
        )
    print("LCFS-start (f=1.00) distance to fixed-b LCFS (mm):", lcfs_dist_mm)


def build_mgrid():
    """Generate high-resolution raw mgrid NetCDF from the discrete coilset."""
    from desc.grid import LinearGrid
    from desc.io import load

    t0 = time.perf_counter()
    FREEB_DIR.mkdir(parents=True, exist_ok=True)
    eq_vac = load(str(EQ_VAC_PATH))
    coils = load(str(COILS_PATH))

    # Check clearance of coil filaments from the mgrid box corners/interior
    cg = LinearGrid(N=200)
    pts_c = []
    for c in coils:
        pts_c.append(np.asarray(c.compute("x", grid=cg, basis="rpz")["x"]))
    pts_c = np.vstack(pts_c)
    in_box = (
        (pts_c[:, 0] >= MGRID_BOX["Rmin"] - 0.01)
        & (pts_c[:, 0] <= MGRID_BOX["Rmax"] + 0.01)
        & (pts_c[:, 2] >= MGRID_BOX["Zmin"] - 0.01)
        & (pts_c[:, 2] <= MGRID_BOX["Zmax"] + 0.01)
    )
    print(f"Coil points inside/near mgrid box: {int(np.sum(in_box))} (should be 0)")

    print(f"Saving mgrid -> {MGRID_PATH} with {MGRID_BOX}...")
    coils.save_mgrid(
        str(MGRID_PATH),
        Rmin=MGRID_BOX["Rmin"],
        Rmax=MGRID_BOX["Rmax"],
        Zmin=MGRID_BOX["Zmin"],
        Zmax=MGRID_BOX["Zmax"],
        nR=MGRID_BOX["nR"],
        nZ=MGRID_BOX["nZ"],
        nphi=MGRID_BOX["nphi"],
        NFP=int(eq_vac.NFP),
        save_vector_potential=False,
        chunk_size=25000,
        source_grid=LinearGrid(N=SOURCE_N),
    )
    print(f"Saved {MGRID_PATH} ({MGRID_PATH.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t0:.1f} s")


def _write_freeb_vmec_input(eq, out_path: Path, mgrid_rel: str, phiedge_override: float | None = None):
    from desc.vmec import VMECIO

    VMECIO.write_vmec_input(
        eq,
        str(out_path),
        DELT=0.5,
        NITER=100,
        NS_ARRAY=[33, 65, 129, 257],
        NITER_ARRAY=[5000, 10000, 20000, 30000],
        FTOL_ARRAY=[1e-12, 1e-12, 1e-12, 1e-12],
        MPOL=11,
        NTOR=10,
    )
    txt = out_path.read_text()
    txt = txt.replace("  LFREEB = F\n", "  LFREEB = T\n")
    txt = txt.replace("  MGRID_FILE = 'none'\n", f"  MGRID_FILE = '{mgrid_rel}'\n  EXTCUR = 1.0\n")
    txt = txt.replace("  NTOR = 10\n", "  NTOR = 10\n  NZETA = 36\n")
    # VMEC2000 AM / AC arrays accept at most 21 coefficients (indices 0:20).
    cleaned_lines = []
    for line in txt.splitlines():
        if line.lstrip().startswith("AM ="):
            vals = [float(tok) for tok in line.split("=", 1)[1].split()]
            if len(vals) > 21 or all(v == 0.0 for v in vals):
                vals = [0.0] if all(v == 0.0 for v in vals) else vals[:21]
                line = "  AM = " + " ".join(f"{v:+14.8E}" for v in vals)
        elif line.lstrip().startswith("AC ="):
            vals = [float(tok) for tok in line.split("=", 1)[1].split()]
            if len(vals) > 21 or all(v == 0.0 for v in vals):
                vals = [0.0] if all(v == 0.0 for v in vals) else vals[:21]
                line = "  AC = " + " ".join(f"{v:+14.8E}" for v in vals)
        cleaned_lines.append(line)
    txt = "\n".join(cleaned_lines) + "\n"
    if phiedge_override is not None:
        old_line = f"  PHIEDGE = {eq.Psi:+14.8E}\n"
        new_line = f"  PHIEDGE = {phiedge_override:+14.8E}\n"
        assert old_line in txt, f"Could not find {old_line!r}"
        txt = txt.replace(old_line, new_line)
    out_path.write_text(txt)


def run_vmec_freeb(np_ranks: int = 32):
    """Run VMEC2000 free-boundary for vacuum and finite-beta (4.12%)."""
    from desc.io import load
    from netCDF4 import Dataset

    if not MGRID_PATH.is_file():
        raise FileNotFoundError(f"Missing {MGRID_PATH}; run --step mgrid first")

    import tempfile

    env = os.environ.copy()
    env.setdefault("TMPDIR", tempfile.gettempdir())
    env["PMIX_MCA_gds"] = "hash"
    env["HWLOC_COMPONENTS"] = "-gl"

    eq_vac = load(str(EQ_VAC_PATH))
    eq_b4 = load(str(EQ_BETA4_PATH))

    # First run vacuum and beta4.12 at fixed design PHIEDGE = eq.Psi
    cases = [
        ("vac", eq_vac, None),
        ("beta4.12", eq_b4, None),
    ]
    rbtor_vac = None
    for name, eq, phi_ovr in cases:
        cdir = FREEB_DIR / f"vmec_{name}"
        cdir.mkdir(parents=True, exist_ok=True)
        wout = cdir / f"wout_{name}.nc"
        if wout.is_file():
            with Dataset(wout) as ds:
                if int(ds["ier_flag"][()]) == 0 and float(ds["fsqr"][()]) < 1e-10:
                    rbtor = float(ds["rbtor"][()])
                    if name == "vac":
                        rbtor_vac = rbtor
                    print(f"  VMEC [{name}] already converged ({wout.name}, rbtor={rbtor:.5f}); skipping.")
                    continue
        inp = cdir / f"input.{name}"
        _write_freeb_vmec_input(eq, inp, f"../{MGRID_PATH.name}", phiedge_override=phi_ovr)
        cmd = ["mpirun", "-np", str(np_ranks), "/opt/bin/xvmec2000", f"input.{name}"]
        print(f"Running VMEC free-boundary [{name}]: {' '.join(cmd)}")
        t0 = time.perf_counter()
        log_path = cdir / f"run_{name}.log"
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, cwd=cdir, env=env, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        wout = cdir / f"wout_{name}.nc"
        if proc.returncode != 0 or not wout.is_file():
            print(f"  !! VMEC [{name}] failed (rc={proc.returncode}) in {dt:.1f}s; check {log_path}")
            continue
        with Dataset(wout) as ds:
            ier = int(ds["ier_flag"][()])
            fsqr = float(ds["fsqr"][()])
            rbtor = float(ds["rbtor"][()])
            beta = float(ds["betatotal"][()])
            vol = float(ds["volume_p"][()])
            rmaj = float(ds["Rmajor_p"][()])
            amin = float(ds["Aminor_p"][()])
            signgs = int(ds["signgs"][()])
            iotaf = np.asarray(ds["iotaf"][()], dtype=float) * signgs
        if name == "vac":
            rbtor_vac = rbtor
        print(
            f"  VMEC [{name}] done in {dt:.1f}s: ier={ier}, fsqr={fsqr:.2e}, "
            f"rbtor={rbtor:.5f}, beta={beta*100:.2f}%, R0={rmaj:.5f}, a={amin:.5f}, "
            f"V={vol:.4f}, iota=[{iotaf[0]:.5f} .. {iotaf[-1]:.5f}]"
        )

    # Also check flux-matched beta4.12 (adjust PHIEDGE by G_coil / G_fixed_b4)
    if rbtor_vac is not None:
        from desc.grid import LinearGrid

        g1 = LinearGrid(rho=1.0, M=8, N=8, NFP=int(eq_b4.NFP))
        G_b4_edge = float(np.mean(eq_b4.compute("G", grid=g1)["G"]))
        phi_adj = float(eq_b4.Psi) * (abs(rbtor_vac) / G_b4_edge)
        name = "beta4.12_phiadj"
        cdir = FREEB_DIR / f"vmec_{name}"
        cdir.mkdir(parents=True, exist_ok=True)
        inp = cdir / f"input.{name}"
        _write_freeb_vmec_input(eq_b4, inp, f"../{MGRID_PATH.name}", phiedge_override=phi_adj)
        cmd = ["mpirun", "-np", str(np_ranks), "/opt/bin/xvmec2000", f"input.{name}"]
        print(f"Running VMEC free-boundary [{name}] (PHIEDGE={phi_adj:.6f}): {' '.join(cmd)}")
        t0 = time.perf_counter()
        log_path = cdir / f"run_{name}.log"
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, cwd=cdir, env=env, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        wout = cdir / f"wout_{name}.nc"
        if proc.returncode == 0 and wout.is_file():
            with Dataset(wout) as ds:
                ier = int(ds["ier_flag"][()])
                fsqr = float(ds["fsqr"][()])
                rbtor = float(ds["rbtor"][()])
                beta = float(ds["betatotal"][()])
                vol = float(ds["volume_p"][()])
                rmaj = float(ds["Rmajor_p"][()])
                amin = float(ds["Aminor_p"][()])
                signgs = int(ds["signgs"][()])
                iotaf = np.asarray(ds["iotaf"][()], dtype=float) * signgs
            print(
                f"  VMEC [{name}] done in {dt:.1f}s: ier={ier}, fsqr={fsqr:.2e}, "
                f"rbtor={rbtor:.5f}, beta={beta*100:.2f}%, R0={rmaj:.5f}, a={amin:.5f}, "
                f"V={vol:.4f}, iota=[{iotaf[0]:.5f} .. {iotaf[-1]:.5f}]"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=("poincare", "mgrid", "vmec", "all"),
        default="all",
    )
    parser.add_argument("--n-fp", type=int, default=400, help="Poincare field periods.")
    parser.add_argument("--np-ranks", type=int, default=32, help="MPI ranks for xvmec2000.")
    args = parser.parse_args()

    if args.step in ("poincare", "all"):
        compute_poincare(n_fp=args.n_fp)
    if args.step in ("mgrid", "all"):
        build_mgrid()
    if args.step in ("vmec", "all"):
        run_vmec_freeb(np_ranks=args.np_ranks)


if __name__ == "__main__":
    main()
