"""Knowledge Bridges for Collaborative Nested Learning.

This module provides our NOVEL contribution:
- KnowledgeBridge: Attention-gated knowledge transfer between timescales
- CollaborativeNestedOptimizer: NestedOptimizer with bidirectional bridges

The paper's information flow is unidirectional (fast → medium → slow).
We add explicit bidirectional bridges with learned gating.
"""

__all__ = ["KnowledgeBridge", "CollaborativeNestedOptimizer"]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "KnowledgeBridge":
        from src.bridges.knowledge_bridges import KnowledgeBridge

        return KnowledgeBridge
    elif name == "CollaborativeNestedOptimizer":
        from src.bridges.knowledge_bridges import CollaborativeNestedOptimizer

        return CollaborativeNestedOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
