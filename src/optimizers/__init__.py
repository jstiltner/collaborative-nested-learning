"""Optimizers for Collaborative Nested Learning.

This module provides:
- DeepMomentumOptimizer: Learned momentum with neural preconditioning
- NestedOptimizer: Multi-timescale optimization
- CollaborativeCMSOptimizer: Full implementation with CMS and bridges
- MultiScaleNestedOptimizer: Generalized N-level nested optimizer

Reference: Nested Learning (Behrouz et al., NeurIPS 2025)
https://abehrouz.github.io/files/NL.pdf
"""

__all__ = [
    "DeepMomentumOptimizer",
    "NestedOptimizer",
    "CollaborativeCMSOptimizer",
    "MultiScaleNestedOptimizer",
    "MultiScaleConfig",
    "MultiScaleBridges",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "DeepMomentumOptimizer":
        from src.optimizers.deep_momentum import DeepMomentumOptimizer

        return DeepMomentumOptimizer
    elif name == "NestedOptimizer":
        from src.optimizers.nested_optimizer import NestedOptimizer

        return NestedOptimizer
    elif name == "CollaborativeCMSOptimizer":
        from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer

        return CollaborativeCMSOptimizer
    elif name == "MultiScaleNestedOptimizer":
        from src.optimizers.multi_scale import MultiScaleNestedOptimizer

        return MultiScaleNestedOptimizer
    elif name == "MultiScaleConfig":
        from src.optimizers.multi_scale import MultiScaleConfig

        return MultiScaleConfig
    elif name == "MultiScaleBridges":
        from src.optimizers.multi_scale import MultiScaleBridges

        return MultiScaleBridges
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
