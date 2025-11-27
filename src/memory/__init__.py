"""Continuum Memory System for Nested Learning.

This module implements the multi-timescale memory system from the Nested Learning paper.
Memory banks at different timescales naturally preserve knowledge through temporal abstraction.

Reference: Nested Learning (Behrouz et al., NeurIPS 2025)
https://abehrouz.github.io/files/NL.pdf
"""

from src.memory.continuum import ContinuumMemorySystem
from src.memory.memory_bank import MemoryBank
from src.memory.multi_scale_cms import MultiScaleCMS, MultiScaleCMSConfig

__all__ = [
    "MemoryBank",
    "ContinuumMemorySystem",
    "MultiScaleCMS",
    "MultiScaleCMSConfig",
]
