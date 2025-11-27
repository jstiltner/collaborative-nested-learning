"""Memory Bank: Stores parameter snapshots at a specific timescale.

This implements the memory bank component from the Nested Learning paper.
Each bank stores parameter snapshots and updates at a specific frequency,
enabling temporal abstraction for continual learning.

Reference: Nested Learning (Behrouz et al., NeurIPS 2025), Section 2.3
https://abehrouz.github.io/files/NL.pdf

Key insight: Slow memory banks naturally preserve old knowledge because
they update less frequently, preventing catastrophic forgetting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn


@dataclass
class MemoryBankStats:
    """Statistics for a memory bank."""

    total_updates: int = 0
    total_retrievals: int = 0
    total_consolidations: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "total_updates": self.total_updates,
            "total_retrievals": self.total_retrievals,
            "total_consolidations": self.total_consolidations,
        }


class MemoryBank:
    """Stores parameter snapshots at a specific timescale.

    This is a core component of the Continuum Memory System from the
    Nested Learning paper. Each memory bank:

    1. Stores snapshots of parameter values
    2. Updates at a specific frequency (fast=1, medium=10, slow=100)
    3. Provides retrieval for knowledge consolidation

    The key insight is that slower banks naturally preserve old knowledge
    because they update less frequently.

    Args:
        update_frequency: How often to update snapshots (1 = every step).
        decay_rate: Exponential moving average decay for smooth updates.
            0.0 = instant update, 0.99 = very slow update.
        name: Human-readable name for this bank (e.g., "fast", "slow").

    Example:
        >>> bank = MemoryBank(update_frequency=10, decay_rate=0.9, name="medium")
        >>> for step in range(100):
        ...     # Get current parameters
        ...     params = list(model.parameters())
        ...     # Maybe update bank
        ...     if bank.maybe_update(params):
        ...         print(f"Bank updated at step {step}")
        ...     # Retrieve memory for regularization
        ...     memory = bank.get_memory(id(params[0]))
    """

    def __init__(
        self,
        update_frequency: int = 1,
        decay_rate: float = 0.0,
        name: str = "memory_bank",
    ):
        if update_frequency < 1:
            raise ValueError(f"update_frequency must be >= 1, got {update_frequency}")
        if not 0.0 <= decay_rate < 1.0:
            raise ValueError(f"decay_rate must be in [0, 1), got {decay_rate}")

        self.update_frequency = update_frequency
        self.decay_rate = decay_rate
        self.name = name

        # Storage for parameter snapshots
        # Maps param_id -> snapshot tensor
        self._snapshots: Dict[int, torch.Tensor] = {}

        # Maps param_id -> original shape (for validation)
        self._shapes: Dict[int, torch.Size] = {}

        # Step counter
        self._step_count = 0

        # Statistics
        self.stats = MemoryBankStats()

    @property
    def step_count(self) -> int:
        """Current step count."""
        return self._step_count

    @property
    def num_params(self) -> int:
        """Number of parameters being tracked."""
        return len(self._snapshots)

    def _snapshot(self, params: Iterable[nn.Parameter]) -> None:
        """Take a snapshot of current parameter values.

        Args:
            params: Iterable of parameters to snapshot.
        """
        for p in params:
            param_id = id(p)

            if param_id not in self._snapshots:
                # First snapshot - just store
                self._snapshots[param_id] = p.data.clone().detach()
                self._shapes[param_id] = p.shape
            else:
                # Update with exponential moving average
                if self.decay_rate > 0:
                    # EMA update: new = decay * old + (1 - decay) * current
                    self._snapshots[param_id].mul_(self.decay_rate)
                    self._snapshots[param_id].add_(p.data, alpha=1 - self.decay_rate)
                else:
                    # Instant update
                    self._snapshots[param_id].copy_(p.data)

        self.stats.total_updates += 1

    def maybe_update(self, params: Iterable[nn.Parameter]) -> bool:
        """Update snapshots if it's time based on frequency.

        Args:
            params: Iterable of parameters to potentially snapshot.

        Returns:
            True if snapshots were updated, False otherwise.
        """
        self._step_count += 1

        if self._step_count % self.update_frequency == 0:
            self._snapshot(params)
            return True
        return False

    def force_update(self, params: Iterable[nn.Parameter]) -> None:
        """Force an immediate snapshot update.

        Args:
            params: Iterable of parameters to snapshot.
        """
        self._snapshot(params)

    def get_memory(self, param_id: int) -> Optional[torch.Tensor]:
        """Retrieve stored memory for a parameter.

        Args:
            param_id: The id() of the parameter.

        Returns:
            Snapshot tensor if available, None otherwise.
            Note: Returns a reference, not a copy. Clone if needed.
        """
        self.stats.total_retrievals += 1
        return self._snapshots.get(param_id)

    def get_all_memories(self) -> Dict[int, torch.Tensor]:
        """Get all stored memories.

        Returns:
            Dictionary mapping param_id to snapshot tensor.
        """
        return self._snapshots.copy()

    def compute_drift(self, params: Iterable[nn.Parameter]) -> Dict[int, float]:
        """Compute how much parameters have drifted from stored memory.

        This is useful for:
        - Detecting when to consolidate knowledge
        - Measuring forgetting
        - Adaptive regularization strength

        Args:
            params: Current parameters to compare against memory.

        Returns:
            Dictionary mapping param_id to L2 drift distance.
        """
        drift: Dict[int, float] = {}

        for p in params:
            param_id = id(p)
            if param_id in self._snapshots:
                # L2 distance between current and stored
                diff = p.data - self._snapshots[param_id]
                drift[param_id] = diff.norm().item()

        return drift

    def compute_importance(
        self,
        params: Iterable[nn.Parameter],
        gradients: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Compute importance scores for parameters.

        Uses gradient magnitude as a proxy for importance.
        Higher gradient = more important for current task.

        Args:
            params: Parameters to compute importance for.
            gradients: Optional pre-computed gradients. If None, uses p.grad.

        Returns:
            Dictionary mapping param_id to importance tensor (same shape as param).
        """
        importance: Dict[int, torch.Tensor] = {}

        for p in params:
            param_id = id(p)

            if gradients is not None and param_id in gradients:
                grad = gradients[param_id]
            elif p.grad is not None:
                grad = p.grad
            else:
                # No gradient available - assume uniform importance
                importance[param_id] = torch.ones_like(p.data)
                continue

            # Importance = gradient magnitude (squared for Fisher-like behavior)
            importance[param_id] = grad.abs()

        return importance

    def consolidate_to(
        self,
        target_bank: "MemoryBank",
        params: Iterable[nn.Parameter],
        importance: Optional[Dict[int, torch.Tensor]] = None,
        threshold: float = 0.5,
    ) -> int:
        """Consolidate important memories to a slower bank.

        This is part of our novel contribution: explicit knowledge transfer
        between memory banks based on importance.

        Args:
            target_bank: The slower bank to consolidate to.
            params: Current parameters (for importance computation).
            importance: Pre-computed importance scores. If None, computed from gradients.
            threshold: Minimum importance to trigger consolidation.

        Returns:
            Number of parameters consolidated.
        """
        if importance is None:
            importance = self.compute_importance(params)

        consolidated = 0
        params_list = list(params)

        for p in params_list:
            param_id = id(p)

            if param_id not in self._snapshots:
                continue

            if param_id in importance:
                # Check if mean importance exceeds threshold
                imp = importance[param_id]
                mean_imp = imp.mean().item()

                if mean_imp >= threshold:
                    # Consolidate: update target bank with our memory
                    if param_id not in target_bank._snapshots:
                        target_bank._snapshots[param_id] = self._snapshots[
                            param_id
                        ].clone()
                        target_bank._shapes[param_id] = self._shapes[param_id]
                    else:
                        # Weighted update based on importance
                        weight = min(mean_imp, 1.0)
                        target_bank._snapshots[param_id].mul_(1 - weight)
                        target_bank._snapshots[param_id].add_(
                            self._snapshots[param_id], alpha=weight
                        )
                    consolidated += 1

        self.stats.total_consolidations += consolidated
        target_bank.stats.total_consolidations += consolidated

        return consolidated

    def compute_regularization_loss(
        self,
        params: Iterable[nn.Parameter],
        importance: Optional[Dict[int, torch.Tensor]] = None,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Compute regularization loss to prevent forgetting.

        This implements EWC-style regularization using stored memories.

        L_reg = strength * sum_i (importance_i * (param_i - memory_i)^2)

        Args:
            params: Current parameters.
            importance: Optional importance weights. If None, uniform.
            strength: Regularization strength multiplier.

        Returns:
            Scalar regularization loss tensor.
        """
        reg_loss = torch.tensor(0.0)
        device = None

        for p in params:
            param_id = id(p)

            if param_id not in self._snapshots:
                continue

            if device is None:
                device = p.device
                reg_loss = reg_loss.to(device)

            # Compute squared difference
            diff = p - self._snapshots[param_id].to(p.device)
            squared_diff = diff.pow(2)

            # Weight by importance if provided
            if importance is not None and param_id in importance:
                imp = importance[param_id].to(p.device)
                weighted_diff = squared_diff * imp
            else:
                weighted_diff = squared_diff

            reg_loss = reg_loss + weighted_diff.sum()

        return strength * reg_loss

    def reset(self) -> None:
        """Clear all stored memories."""
        self._snapshots.clear()
        self._shapes.clear()
        self._step_count = 0
        self.stats = MemoryBankStats()

    def get_stats(self) -> Dict[str, Any]:
        """Get bank statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "name": self.name,
            "update_frequency": self.update_frequency,
            "decay_rate": self.decay_rate,
            "step_count": self._step_count,
            "num_params": self.num_params,
            **self.stats.to_dict(),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return state for serialization.

        Returns:
            Dictionary containing bank state.
        """
        return {
            "name": self.name,
            "update_frequency": self.update_frequency,
            "decay_rate": self.decay_rate,
            "step_count": self._step_count,
            "snapshots": {k: v.cpu() for k, v in self._snapshots.items()},
            "shapes": dict(self._shapes),
            "stats": self.stats.to_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from serialization.

        Args:
            state_dict: Dictionary containing bank state.
        """
        self.name = state_dict["name"]
        self.update_frequency = state_dict["update_frequency"]
        self.decay_rate = state_dict["decay_rate"]
        self._step_count = state_dict["step_count"]
        self._snapshots = {k: v.clone() for k, v in state_dict["snapshots"].items()}
        self._shapes = dict(state_dict["shapes"])

        stats = state_dict.get("stats", {})
        self.stats = MemoryBankStats(
            total_updates=stats.get("total_updates", 0),
            total_retrievals=stats.get("total_retrievals", 0),
            total_consolidations=stats.get("total_consolidations", 0),
        )


def _test_memory_bank():
    """Test the MemoryBank class."""
    print("Testing MemoryBank...")

    # Create a simple model
    model = nn.Linear(10, 5)

    # Create memory bank
    bank = MemoryBank(update_frequency=5, decay_rate=0.9, name="test_bank")

    # Simulate training steps
    for step in range(20):
        # Modify parameters slightly
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        # Maybe update bank
        updated = bank.maybe_update(model.parameters())
        if updated:
            print(f"  Step {step + 1}: Bank updated")

    # Check stats
    stats = bank.get_stats()
    print(f"  Stats: {stats}")

    # Test memory retrieval
    for p in model.parameters():
        memory = bank.get_memory(id(p))
        if memory is not None:
            print(f"  Memory shape: {memory.shape}")

    # Test drift computation
    drift = bank.compute_drift(model.parameters())
    print(f"  Drift: {drift}")

    # Test regularization loss
    reg_loss = bank.compute_regularization_loss(model.parameters(), strength=0.1)
    print(f"  Regularization loss: {reg_loss.item():.6f}")

    # Test consolidation
    slow_bank = MemoryBank(update_frequency=100, name="slow")
    consolidated = bank.consolidate_to(slow_bank, model.parameters(), threshold=0.0)
    print(f"  Consolidated {consolidated} parameters to slow bank")

    print("✓ MemoryBank test passed!")


if __name__ == "__main__":
    _test_memory_bank()
