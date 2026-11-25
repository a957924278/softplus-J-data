# Direct Optimization of Stellarator Omnigenity from the Second Adiabatic Invariant — Code & Data Package

Companion code, equilibria, modular coils, verification outputs, and figure-reproduction scripts for:

> **Hanlin Chen, Zhiyuan Lu, Guosheng Xu, Shuai Cao, Yang Han, Dehong Chen, and Baonian Wan**,  
> *Direct Optimization of Stellarator Omnigenity from the Second Adiabatic Invariant*,  
> submitted to *Nuclear Fusion* (manuscript NF-109458).

---

## Contents

```text
softplus-J-data/
├── softplus_j/                  # Standalone Python/JAX implementation for upstream DESC
│   ├── compute.py               # Softplus action proxy, analytical dJ/dalpha, log-sum-exp extrema, soft connectivity
│   ├── objectives.py            # SecondAdiabaticInvariantAlphaDerivative (f_J) & SoftConnectivity (f_SC)
│   └── spline.py                # SplineZetaMin periodic cubic-spline parameterization
├── data/
│   ├── equilibria/              # Final DESC equilibria (*.h5): seed, SC-off, SC-on, Compact, Balanced (vac, beta=4.12%, I_bs)
│   ├── coilsets/                # 32-coil modular coil set for the Balanced configuration (tmp126_coils_n4_hard_bn_short19p2.h5)
│   ├── cobra/                   # COBRA ideal-ballooning growth rates (I_tor=0 & self-consistent I_bs) + max-J diagnostic cache
│   ├── neo/                     # NEO effective-ripple outputs (SC-off, SC-on, Compact, Balanced, W7-X High Mirror)
│   ├── sfincs/                  # 4D drift-kinetic SFINCS bootstrap current scan & high-resolution convergence checks
│   ├── simple/                  # SIMPLE collisionless alpha-particle loss traces (ARIES-CS reactor scale)
│   └── freeb/                   # Vacuum Biot–Savart Poincare data and VMEC2000 free-boundary equilibria (wout_*.nc, boozmn_*.nc)
└── scripts/                     # Reproducible figure-generation, free-boundary analysis, and reference optimization scripts
```

*(Note: to keep the repository lightweight, large regenerable intermediate files—`mgrid_tmp126_n4.nc` and unused VMEC `jxbout_*.nc` files—are omitted and can be regenerated directly via `scripts/coils_freeb_compute.py`.)*

---

## Installation & Quick Start

The `softplus_j` package works directly with standard upstream [`desc-opt`](https://github.com/PlasmaControl/DESC) and JAX:

```bash
pip install -e .
```

Minimal example adding `f_J` and `f_SC` to a DESC `ObjectiveFunction`:

```python
import numpy as np
from desc.grid import LinearGrid
from desc.io import load
from desc.objectives import CurrentDensity, ObjectiveFunction
from softplus_j import (
    SecondAdiabaticInvariantAlphaDerivative,
    SoftConnectivity,
    SplineZetaMin,
)

eq = load("data/equilibria/seed_beta0.00.h5")
omni_grid = LinearGrid(rho=1.0, M=4 * eq.M, N=4 * eq.N, NFP=eq.NFP, sym=False)
spline = SplineZetaMin(n_control=16, NFP=eq.NFP, symmetry=False)

objective = ObjectiveFunction((
    CurrentDensity(eq, weight=0.1),
    SecondAdiabaticInvariantAlphaDerivative(
        eq, grid=omni_grid, num_alpha=32, nzeta=32,
        t=np.linspace(0.05, 0.90, 32), weight=3e-2,
    ),
    SoftConnectivity(
        eq, spline=spline, grid=omni_grid, num_alpha=32,
        t=np.linspace(0.0, 1.0, 32), sigmoid_sharpness=50.0, weight=1e-4,
    ),
))
```

To reproduce any of the four optimization configurations listed in **Table 1** of the manuscript (`sc_off`, `sc_on`, `compact`, `balanced`):

```bash
python scripts/run_optimization_examples.py --case sc_on --output out/sc_on.h5
```

---

## Reproducing Manuscript Figures

Run any script from the repository root (outputs are written into `figures/`):

| Manuscript Figure | Script | Primary Inputs |
|---|---|---|
| Fig. 1 (Softplus regularization) | `python scripts/make_fig_softplus_demo.py` | Analytical |
| Fig. 2 (Boozer field-line label) | `python scripts/make_fig1_field_line_label.py` | `data/equilibria/boozer_label_beta0.00.h5` |
| Fig. 3 (SC-off action diagnostics) | `python scripts/make_fig2_action_diagnostics.py` | `data/equilibria/sc_off_beta0.00.h5` |
| Fig. 4 (Soft-connectivity splines) | `python scripts/make_fig3_sc_construction.py` | `data/equilibria/compact_beta0.00.h5` |
| Figs. 5, 8, 10 (Config boundary, $\|B\|$, $\|\partial_\alpha\widetilde{J}\|$) | `python scripts/make_fig_config_maps.py` | `data/equilibria/{sc_off,sc_on,compact,balanced_vac}_beta0.00.h5` |
| Fig. 6 (SC-off branch splitting & $\sigma_J$) | `python scripts/make_fig5_sc_topology.py` | `data/equilibria/sc_off_beta0.00.h5` |
| Figs. 7, 9, 11 (SIMPLE $\alpha$ loss & NEO $\varepsilon_{\mathrm{eff}}$) | `python scripts/make_fig_conf_diagnostics.py` | `data/simple/*.dat`, `data/neo/*` |
| Fig. 12 (Hard-cutoff maximum-$J$ check $\partial J/\partial\rho|_{B^*}$) | `python scripts/make_fig_maxJ_diagnostic.py` | `data/equilibria/balanced_*.h5`, `data/cobra/balanced_maxJ_cache.json` |
| Fig. 13 (Balanced profiles, $\iota$, SFINCS $I_{\mathrm{bs}}$) | `python scripts/make_fig_profiles_bootstrap.py` | `data/equilibria/balanced_*.h5`, `data/sfincs/*.json` |
| Fig. 14 (Independent hard-cutoff $\sigma_J$ vs W7-X) | `python scripts/make_fig11_hard_action.py` | `data/equilibria/*.h5` |
| Fig. 15 (COBRA ideal-ballooning, $I_{\mathrm{tor}}=0$ & $I_{\mathrm{bs}}$) | `python scripts/make_fig12_cobra_stability.py` | `data/cobra/balanced_cobra_grate.beta4.12*` |
| Fig. 16 (32-coil realization) | `python scripts/make_fig13_coils.py` | `data/coilsets/tmp126_coils_n4_hard_bn_short19p2.h5` |
| Figs. 17–18 (Vacuum Poincaré & VMEC2000 free-boundary) | `python scripts/make_fig_coils_poincare_freeb.py` | `data/freeb/{poincare_vac,freeb_analysis}.npz` |

---

## License

Released under the [MIT License](LICENSE).
