"""Knowledge Bridges for Collaborative Nested Learning.

This module provides our NOVEL contribution:
- KnowledgeBridge: Attention-gated knowledge transfer between timescales
- CollaborativeNestedOptimizer: NestedOptimizer with bidirectional bridges

The paper's information flow is unidirectional (fast → medium → slow).
We add explicit bidirectional bridges with learned gating.
"""

from src.bridges.knowledge_bridges import (
    KnowledgeBridge,
    CollaborativeNestedOptimizer,
)

__all__ = ["KnowledgeBridge", "CollaborativeNestedOptimizer"]