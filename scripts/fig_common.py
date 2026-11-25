"""Shared helpers for J-paper figure scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COLORS = {
    "failure": "#D55E00",
    "ablation": "#56B4E9",
    "compact": "#0072B2",
    "balanced": "#009E73",
    "reference": "#4D4D4D",
    "accent": "#CC79A7",
}

# Submission-package layout:
#   J_paper/
#   ├── figures/   exported manuscript figures
#   ├── scripts/   figure/analysis scripts (this directory)
#   └── data/      input data consumed by the scripts
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
DATA_DIR = SCRIPT_DIR.parent / "data"


def data_path(*parts: str) -> Path:
    """Return an absolute path under the submission-package data directory."""
    return DATA_DIR.joinpath(*parts)


def figure_path(*parts: str) -> Path:
    """Return an absolute path under the submission-package figures directory."""
    return FIGURES_DIR.joinpath(*parts)


def setup_matplotlib(font_size: float = 9.0) -> None:
    """Use one journal-style plotting system across manuscript figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.fontsize": font_size - 0.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "lines.linewidth": 1.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_matplotlib(fig, output: str | Path) -> Path:
    """Save the requested figure and a PNG companion for vector outputs."""
    output = Path(output).expanduser()
    if not output.is_absolute():
        output = FIGURES_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300, pad_inches=0.03)
    if output.suffix.lower() in {".pdf", ".svg"}:
        fig.savefig(
            output.with_suffix(".png"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.03,
        )
    return output.resolve()


def load_equilibrium(path: str | Path):
    """Load a DESC equilibrium from a path resolved relative to the repo root."""
    from desc.io import load

    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return load(str(path))


def boozer_spectral_data(
    eq,
    *,
    rho: float = 1.0,
    keys: tuple[str, ...] = ("|B|_mn_B", "sqrt(g)_Boozer_mn", "iota"),
    m_booz: int | None = None,
    n_booz: int | None = None,
):
    """Compute Boozer spectral coefficients and return the pieces used by plots."""
    from desc.compute import get_profiles, get_transforms
    from softplus_j import _reshape_surface_coefficients
    from desc.compute.utils import _compute as compute_fun
    from desc.grid import LinearGrid

    m_booz = 2 * eq.M if m_booz is None else m_booz
    n_booz = 2 * eq.N if n_booz is None else n_booz
    grid = LinearGrid(rho=rho, M=2 * m_booz, N=2 * n_booz, NFP=eq.NFP, sym=False)
    profiles = get_profiles(keys, obj=eq, grid=grid)
    transforms = get_transforms(keys, obj=eq, grid=grid, M_booz=m_booz, N_booz=n_booz)
    data = compute_fun(
        "desc.equilibrium.equilibrium.Equilibrium",
        keys,
        params=eq.params_dict,
        transforms=transforms,
        profiles=profiles,
    )

    basis = transforms["B"].basis
    grid_b = transforms["grid"]
    out = {
        "basis": basis,
        "grid": grid_b,
        "data": data,
        "rho": np.asarray(grid_b.compress(grid_b.nodes[:, 0]), dtype=float),
        "iota": np.asarray(grid_b.compress(data["iota"]), dtype=float),
    }
    if "|B|_mn_B" in keys:
        out["coeff_B"] = np.asarray(
            _reshape_surface_coefficients(grid_b, data["|B|_mn_B"])
        )
    if "sqrt(g)_Boozer_mn" in keys:
        out["coeff_g"] = np.asarray(
            _reshape_surface_coefficients(grid_b, data["sqrt(g)_Boozer_mn"])
        )
    return out


def field_line_B(
    eq,
    *,
    rho: float = 1.0,
    alpha: float = 0.0,
    nzeta: int = 2000,
    zeta_periods: float = 1.0,
):
    """Evaluate |B| along one Boozer field line over ``zeta_periods``."""
    bd = boozer_spectral_data(eq, rho=rho, keys=("|B|_mn_B", "iota"))
    zeta = np.linspace(
        0.0, zeta_periods * 2.0 * np.pi / eq.NFP, nzeta
    )
    iota = float(bd["iota"][0])
    theta = alpha + iota * zeta
    nodes = np.stack((np.full_like(zeta, rho), theta, zeta), axis=-1)
    B = np.asarray(bd["basis"].evaluate(nodes) @ bd["coeff_B"][0], dtype=float)
    return zeta, B, iota, bd


def surface_B_range(bd) -> tuple[float, float]:
    """Return min and max |B| on the sampled Boozer surface."""
    grid = bd["grid"]
    B = np.asarray(bd["basis"].evaluate(grid.nodes[:: grid.num_rho]) @ bd["coeff_B"][0])
    return float(np.min(B)), float(np.max(B))


def count_periodic_true_components(mask: np.ndarray) -> int:
    """Count true-valued connected components on a periodic 1D mask."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0
    return int(np.sum((~np.roll(mask, 1)) & mask))


def true_intervals(x: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Return non-periodic intervals where mask is true, for visual shading."""
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)
    intervals = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            intervals.append((float(x[start]), float(x[i - 1])))
            start = None
    if start is not None:
        intervals.append((float(x[start]), float(x[-1])))
    return intervals


def smooth_J_map(
    eq,
    *,
    rho: float = 1.0,
    num_alpha: int = 48,
    nt: int = 48,
    nzeta: int = 800,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    wrap_alpha: bool = True,
):
    """Compute the smoothed second adiabatic invariant J(alpha, t).

    By default this retains the historical half-domain alpha grid.  Supplying
    ``alpha_min`` and ``alpha_max`` computes that interval directly, which is
    useful when a full alpha map is required without symmetry reconstruction.
    """
    from desc.backend import jnp
    from softplus_j import (
        _boozer_B_star_from_t,
        _smoothmax_logsumexp,
        boozer_second_adiabatic_invariant_from_data,
    )

    bd = boozer_spectral_data(eq, rho=rho)
    basis = bd["basis"]
    grid = bd["grid"]
    B_full = np.asarray(basis.evaluate(grid.nodes[:: grid.num_rho]) @ bd["coeff_B"].T)
    B_full = B_full.T
    B_range = B_full.max(axis=1) - B_full.min(axis=1)
    tau_eff = (0.1 * B_range / np.log(max(B_full.shape[1], 2)))[:, None]
    B_max = np.asarray(_smoothmax_logsumexp(B_full, axis=1, tau=tau_eff)).squeeze(-1)
    B_min = -np.asarray(_smoothmax_logsumexp(-B_full, axis=1, tau=tau_eff)).squeeze(-1)

    t = np.linspace(0.05, 0.95, nt)
    B_star = _boozer_B_star_from_t(
        jnp.asarray(B_min), jnp.asarray(B_max), jnp.asarray(t)
    )

    iota = float(bd["iota"][0])
    if alpha_min is None and alpha_max is None:
        alpha0 = np.pi - iota * (np.pi / eq.NFP)
        alpha = np.linspace(-np.pi, 0.0, num_alpha, endpoint=False) + alpha0
    elif alpha_min is not None and alpha_max is not None:
        alpha = np.linspace(alpha_min, alpha_max, num_alpha, endpoint=False)
    else:
        raise ValueError("alpha_min and alpha_max must be supplied together")
    J = np.asarray(
        boozer_second_adiabatic_invariant_from_data(
            grid, basis, bd["data"], alpha, B_star, nzeta=nzeta
        )
    )[0]

    if wrap_alpha:
        alpha_plot = np.mod(alpha, 2.0 * np.pi)
        order = np.argsort(alpha_plot)
        return t, alpha_plot[order], J[order]
    return t, alpha, J


def full_spline_knots(spline, iota: float):
    """Expand SplineZetaMin reduced knots using the same symmetry as DESC."""
    alpha = np.asarray(spline._alpha_knots, dtype=float)
    zeta_span = 2.0 * np.pi / spline._NFP

    zeta_min_half = np.clip(
        np.asarray(spline.zeta_min_knots, dtype=float), 0.0, zeta_span
    )
    alpha_min = np.concatenate(
        [alpha, np.mod(2.0 * np.pi - alpha - iota * zeta_span, 2.0 * np.pi)]
    )
    zeta_min = np.concatenate([zeta_min_half, zeta_span - zeta_min_half])
    min_order = np.argsort(alpha_min)

    zeta_max_half = (
        np.mod(
            np.asarray(spline.zeta_max_knots, dtype=float) + 0.5 * zeta_span, zeta_span
        )
        - 0.5 * zeta_span
    )
    alpha_max = np.concatenate([alpha, np.mod(2.0 * np.pi - alpha, 2.0 * np.pi)])
    zeta_max = np.concatenate([zeta_max_half, -zeta_max_half])
    max_order = np.argsort(alpha_max)

    return (
        alpha_min[min_order],
        zeta_min[min_order],
        alpha_max[max_order],
        zeta_max[max_order],
    )


def plot_spline_on_boozer(eq, spline, ax):
    """Plot Boozer surface with zeta_min and zeta_max spline overlays."""
    from desc.plotting import plot_boozer_surface
    from interpax import interp1d

    existing_axes = set(ax.figure.axes)
    plot_boozer_surface(eq, ax=ax, cmap="viridis")
    # The surface plot always creates a colorbar; this diagnostic only needs
    # the geometry and spline overlays.
    for extra_ax in set(ax.figure.axes) - existing_axes:
        extra_ax.remove()
    ax.set_title("")
    iota = float(eq.compute("iota")["iota"][-1])
    zeta_span = 2.0 * np.pi / spline._NFP
    (
        alpha_min_knots,
        zeta_min_knots,
        alpha_max_knots,
        zeta_max_knots,
    ) = full_spline_knots(spline, iota)

    alpha = np.linspace(0.0, 2.0 * np.pi, 600, endpoint=False)
    zeta_min = np.asarray(
        interp1d(
            alpha, alpha_min_knots, zeta_min_knots, method="cubic", period=2.0 * np.pi
        )
    )
    zeta_max = np.asarray(
        interp1d(
            alpha, alpha_max_knots, zeta_max_knots, method="cubic", period=2.0 * np.pi
        )
    )

    def draw_periodic_line(zeta_line, theta_line, *args, **kwargs):
        zeta_plot = np.mod(np.asarray(zeta_line), zeta_span)
        theta_plot = np.mod(np.asarray(theta_line), 2.0 * np.pi)
        breaks = (
            np.where(
                (np.abs(np.diff(zeta_plot)) > 0.5 * zeta_span)
                | (np.abs(np.diff(theta_plot)) > np.pi)
            )[0]
            + 1
        )
        labeled = False
        for segment in np.split(np.arange(zeta_plot.size), breaks):
            if segment.size < 2:
                continue
            label = kwargs.get("label") if not labeled else None
            ax.plot(
                zeta_plot[segment],
                theta_plot[segment],
                *args,
                **{**kwargs, "label": label},
            )
            labeled = True

    draw_periodic_line(
        zeta_min,
        alpha + iota * zeta_min,
        color=COLORS["failure"],
        lw=1.6,
        label=r"$\zeta_{\min}(\alpha)$",
    )
    draw_periodic_line(
        zeta_max,
        alpha + iota * zeta_max,
        color=COLORS["compact"],
        lw=1.6,
        label=r"$\zeta_{\max}(\alpha)$",
    )
    return ax


def read_loss_curve(path: str | Path):
    """Read SIMPLE confined-fraction output and return time and loss percent."""
    data = np.loadtxt(path, comments="#", dtype=float)
    return data[:, 0], (1.0 - data[:, 1] - data[:, 2]) * 100.0
