"""Collaborative Nested Learning: Production implementation of Google's Nested Learning.

This package implements:
- DeepMomentumOptimizer: Neural network that learns to optimize
- NestedOptimizer: Multi-timescale optimization
- CollaborativeNestedOptimizer: Our novel bidirectional knowledge bridges
- KnowledgeBridge: Attention-gated knowledge transfer
- ContinuumMemorySystem: Multi-frequency MLP architecture (coming soon)

Reference: Nested Learning (Behrouz et al., NeurIPS 2025)
https://abehrouz.github.io/files/NL.pdf

Our Novel Contribution:
- Bidirectional knowledge bridges between timescales
- Attention-based gating to decide when to share knowledge
- Fast memory can teach medium, slow can guide fast
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
# Users should import directly from submodules:
#   from src.optimizers.deep_momentum import DeepMomentumOptimizer
#   from src.optimizers.nested_optimizer import NestedOptimizer
#   from src.bridges.knowledge_bridges import KnowledgeBridge, CollaborativeNestedOptimizer

__all__ = [
    "DeepMomentumOptimizer",
    "NestedOptimizer",
    "KnowledgeBridge",
    "CollaborativeNestedOptimizer",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "DeepMomentumOptimizer":
        from src.optimizers.deep_momentum import DeepMomentumOptimizer

        return DeepMomentumOptimizer
    elif name == "NestedOptimizer":
        from src.optimizers.nested_optimizer import NestedOptimizer

        return NestedOptimizer
    elif name == "KnowledgeBridge":
        from src.bridges.knowledge_bridges import KnowledgeBridge

        return KnowledgeBridge
    elif name == "CollaborativeNestedOptimizer":
        from src.bridges.knowledge_bridges import CollaborativeNestedOptimizer

        return CollaborativeNestedOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
