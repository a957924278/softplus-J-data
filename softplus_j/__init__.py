"""Differentiable second-adiabatic-invariant omnigenity objectives for DESC.

Companion implementation for:
    H. Chen, Z. Lu, G. Xu, S. Cao, Y. Han, D. Chen, and B. Wan,
    "Direct Optimization of Stellarator Omnigenity from the Second Adiabatic
    Invariant", submitted to Nuclear Fusion (NF-109458).
"""

from .compute import (
    SOFTPLUS_SHARPNESS,
    _boozer_B_star_from_t,
    _reshape_surface_coefficients,
    _smoothmax_logsumexp,
    _softplus_relu,
    _softplus_relu_sigmoid,
    boozer_second_adiabatic_invariant_alpha_derivative_analytical,
    boozer_second_adiabatic_invariant_alpha_derivative_from_data,
    boozer_second_adiabatic_invariant_from_data,
    boozer_soft_connectivity_penalty,
    boozer_soft_connectivity_penalty_from_data,
)
from .objectives import (
    SecondAdiabaticInvariantAlphaDerivative,
    SoftConnectivity,
)
from .spline import SplineZetaMin

__all__ = [
    "SOFTPLUS_SHARPNESS",
    "SecondAdiabaticInvariantAlphaDerivative",
    "SoftConnectivity",
    "SplineZetaMin",
    "boozer_second_adiabatic_invariant_alpha_derivative_analytical",
    "boozer_second_adiabatic_invariant_alpha_derivative_from_data",
    "boozer_second_adiabatic_invariant_from_data",
    "boozer_soft_connectivity_penalty",
    "boozer_soft_connectivity_penalty_from_data",
    "_boozer_B_star_from_t",
    "_reshape_surface_coefficients",
    "_smoothmax_logsumexp",
    "_softplus_relu",
    "_softplus_relu_sigmoid",
]
