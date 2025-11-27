"""Optimizers for Collaborative Nested Learning.

This module provides:
- DeepMomentumOptimizer: Learned momentum with neural preconditioning
- NestedOptimizer: Multi-timescale optimization
- CollaborativeCMSOptimizer: Full implementation with CMS and bridges
- MultiScaleNestedOptimizer: Generalized N-level nested optimizer

Reference: Nested Learning (Behrouz et al., NeurIPS 2025)
https://abehrouz.github.io/files/NL.pdf
"""

from src.optimizers.deep_momentum import DeepMomentumOptimizer
from src.optimizers.nested_optimizer import NestedOptimizer
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer
from src.optimizers.multi_scale import (
    MultiScaleNestedOptimizer,
    MultiScaleConfig,
    MultiScaleBridges,
)

__all__ = [
    "DeepMomentumOptimizer",
    "NestedOptimizer",
    "CollaborativeCMSOptimizer",
    "MultiScaleNestedOptimizer",
    "MultiScaleConfig",
    "MultiScaleBridges",
]