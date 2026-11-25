"""Reference DESC optimization script reproducing the Table 1 objective setups.

Supports all four vacuum configurations reported in:
    H. Chen et al., "Direct Optimization of Stellarator Omnigenity from the
    Second Adiabatic Invariant" (Nuclear Fusion, NF-109458).

Usage
-----
    python scripts/run_optimization_examples.py --case sc_on --output out/sc_on.h5
    python scripts/run_optimization_examples.py --case sc_off --output out/sc_off.h5
    python scripts/run_optimization_examples.py --case compact --seed data/equilibria/seed_beta0.00.h5 --output out/compact.h5
    python scripts/run_optimization_examples.py --case balanced --seed data/equilibria/balanced_vac_beta0.00.h5 --output out/balanced.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from desc.grid import LinearGrid
from desc.io import load
from desc.objectives import (
    AspectRatio,
    CurrentDensity,
    Elongation,
    FixBoundaryR,
    FixCurrent,
    FixParameters,
    FixPressure,
    FixPsi,
    MagneticWell,
    ObjectiveFunction,
    RotationalTransform,
)
from desc.optimize import Optimizer

from softplus_j import (
    SecondAdiabaticInvariantAlphaDerivative,
    SoftConnectivity,
    SplineZetaMin,
)


def _well_lower_bound(rho, well_depth: float = 1e-2):
    return np.where(rho < 1.01, well_depth * np.power(rho, 2.2), 0.0)


# Exact Table 1 specifications from the manuscript.
CASE_SPECS = {
    "sc_off": dict(
        lm=8,
        w_mhd=1e-1,
        w_ar=5e-2,
        target_ar=10.0,
        w_el=1e-2,
        target_el=4.95,
        w_iota=1e-2,
        target_iota=1.0,
        w_J=3e-2,
        t_J=(0.05, 0.90),
        use_sc=False,
        use_well=False,
    ),
    "sc_on": dict(
        lm=8,
        w_mhd=1e-1,
        w_ar=5e-2,
        target_ar=10.0,
        w_el=1e-2,
        target_el=4.95,
        w_iota=1e-2,
        target_iota=1.0,
        w_J=3e-2,
        t_J=(0.05, 0.90),
        use_sc=True,
        w_sc=1e-4,
        n_knots=16,
        spline_sym=False,
        fix_zeta_max=False,
        sc_num_alpha=32,
        sc_xi=(0.0, 1.0),
        sc_nxi=32,
        use_well=False,
    ),
    "compact": dict(
        lm=8,
        w_mhd=1e-1,
        w_ar=5e-2,
        target_ar=4.295,
        w_el=3e-2,
        target_el=5.0,
        w_iota=1e-2,
        target_iota=0.72,
        w_J=1e-2,
        t_J=(0.05, 0.90),
        use_sc=True,
        w_sc=5e-5,
        n_knots=8,
        spline_sym=True,
        fix_zeta_max=True,
        sc_num_alpha=50,
        sc_xi=(0.2, 0.8),
        sc_nxi=32,
        use_well=False,
    ),
    "balanced": dict(
        lm=6,
        w_mhd=1.0,
        w_ar=5e-4,
        target_ar=8.5,
        w_el=5e-4,
        target_el=5.7,
        w_iota=1e-2,
        target_iota=0.96,
        w_J=3e-2,
        t_J=(0.00, 0.50),
        use_sc=True,
        w_sc=1e-4,
        n_knots=16,
        spline_sym=False,
        fix_zeta_max=False,
        sc_num_alpha=24,
        sc_xi=(0.0, 1.0),
        sc_nxi=24,
        use_well=True,
        w_well=1e-1,
    ),
}


def run_case(case_name: str, seed_path: Path, maxiter: int, stop_grad_sigma_B: bool):
    spec = CASE_SPECS[case_name]
    eq = load(str(seed_path))
    lm = spec["lm"]
    if (eq.L, eq.M, eq.N) != (lm, lm, lm):
        eq.change_resolution(L=lm, M=lm, N=lm, L_grid=2 * lm, M_grid=2 * lm, N_grid=2 * lm)

    grid_edge = LinearGrid(rho=1.0, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP, sym=True)
    omni_grid = LinearGrid(rho=1.0, M=4 * eq.M, N=4 * eq.N, NFP=eq.NFP, sym=False)

    n_J = 4 * eq.M
    objs = [
        CurrentDensity(eq, weight=spec["w_mhd"]),
        AspectRatio(eq, target=spec["target_ar"], weight=spec["w_ar"]),
        Elongation(eq, target=spec["target_el"], weight=spec["w_el"]),
        RotationalTransform(
            eq, target=spec["target_iota"], weight=spec["w_iota"], grid=grid_edge
        ),
        SecondAdiabaticInvariantAlphaDerivative(
            eq,
            grid=omni_grid,
            num_alpha=n_J,
            nzeta=n_J,
            t=np.linspace(spec["t_J"][0], spec["t_J"][1], n_J),
            weight=spec["w_J"],
            stop_grad_sigma_B=stop_grad_sigma_B,
        ),
    ]
    constraints = [
        FixBoundaryR(eq=eq, modes=[0, 0, 0]),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    ]
    things = [eq]
    spline = None
    if spec["use_sc"]:
        spline = SplineZetaMin(
            n_control=spec["n_knots"],
            NFP=eq.NFP,
            symmetry=spec["spline_sym"],
            zeta_max_knots=np.zeros(spec["n_knots"]),
        )
        objs.append(
            SoftConnectivity(
                eq,
                spline=spline,
                grid=omni_grid,
                weight=spec["w_sc"],
                num_alpha=spec["sc_num_alpha"],
                t=np.linspace(spec["sc_xi"][0], spec["sc_xi"][1], spec["sc_nxi"]),
                sigmoid_sharpness=50.0,
            )
        )
        things.append(spline)
        if spec["fix_zeta_max"]:
            constraints.append(
                FixParameters(spline, params={"zeta_max_knots": True})
            )

    if spec["use_well"]:
        objs.append(
            MagneticWell(
                eq,
                bounds=(lambda x: _well_lower_bound(x, 1e-2), np.inf),
                weight=spec["w_well"],
            )
        )

    optimizer = Optimizer("lsq-exact")
    out, result = optimizer.optimize(
        tuple(things) if len(things) > 1 else eq,
        ObjectiveFunction(tuple(objs)),
        tuple(constraints),
        maxiter=maxiter,
        ftol=1e-3,
        xtol=1e-10,
        verbose=2,
    )
    eq_opt = out[0] if isinstance(out, tuple) else out
    return eq_opt, spline, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_SPECS), default="sc_on")
    parser.add_argument(
        "--seed",
        type=Path,
        default=REPO_ROOT / "data" / "equilibria" / "seed_beta0.00.h5",
    )
    parser.add_argument("--maxiter", type=int, default=180)
    parser.add_argument(
        "--stop-grad-sigma-b",
        action="store_true",
        help="Apply jax.lax.stop_gradient to the adaptive log-sum-exp temperature sigma_B.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    eq_opt, _, result = run_case(
        args.case, args.seed, args.maxiter, args.stop_grad_sigma_b
    )
    print("Optimizer status:", result.get("message", ""))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        eq_opt.save(str(args.output))
        print(f"Saved equilibrium to {args.output}")


if __name__ == "__main__":
    main()
