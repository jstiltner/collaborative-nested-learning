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

from src.optimizers import DeepMomentumOptimizer, NestedOptimizer
from src.bridges import KnowledgeBridge, CollaborativeNestedOptimizer

__version__ = "0.1.0"
__all__ = [
    "DeepMomentumOptimizer",
    "NestedOptimizer",
    "KnowledgeBridge",
    "CollaborativeNestedOptimizer",
]