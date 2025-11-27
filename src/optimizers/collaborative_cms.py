"""Collaborative Nested Optimizer with Continuum Memory System.

This is the FULL implementation combining:
1. Multi-timescale optimization (NestedOptimizer)
2. Bidirectional knowledge bridges (our novel contribution)
3. Continuum Memory System for forgetting prevention

This optimizer implements the complete Nested Learning architecture with
our novel extension of bidirectional knowledge bridges between memory banks.

Reference: Nested Learning (Behrouz et al., NeurIPS 2025)
https://abehrouz.github.io/files/NL.pdf

Our extension: Bidirectional bridges enable explicit cross-timescale learning.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

from src.bridges.knowledge_bridges import CollaborativeNestedOptimizer
from src.memory.continuum import CMSConfig, ContinuumMemorySystem


class CollaborativeCMSOptimizer(CollaborativeNestedOptimizer):
    """Full Nested Learning optimizer with CMS and bidirectional bridges.

    This is the complete implementation combining:
    - Multi-timescale optimization (fast/medium/slow)
    - Continuum Memory System for forgetting prevention
    - Bidirectional knowledge bridges (our novel contribution)

    The CMS stores parameter snapshots at different timescales:
    - Fast memory: Updates every step, captures immediate patterns
    - Medium memory: Updates every N steps, captures short-term patterns
    - Slow memory: Updates every M steps, captures long-term principles

    The bridges enable explicit knowledge transfer:
    - Forward: fast → medium → slow (consolidation)
    - Reverse: slow → fast, slow → medium (guidance) [NOVEL]

    Args:
        params: Model parameters to optimize.
        cms_config: Configuration for Continuum Memory System.
        use_cms_regularization: Whether to add CMS regularization loss.
        consolidate_on_task_switch: Whether to consolidate at task boundaries.
        **kwargs: Arguments passed to CollaborativeNestedOptimizer.

    Example:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = CollaborativeCMSOptimizer(model.parameters())
        >>> for task_id, task_data in enumerate(tasks):
        ...     for x, y in task_data:
        ...         optimizer.zero_grad()
        ...         loss = criterion(model(x), y)
        ...
        ...         # Get regularization loss from CMS
        ...         reg_loss = optimizer.get_regularization_loss()
        ...         total_loss = loss + reg_loss
        ...
        ...         total_loss.backward()
        ...         result = optimizer.step()
        ...
        ...     # Consolidate at task boundary
        ...     optimizer.consolidate_memory()
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        cms_config: Optional[CMSConfig] = None,
        use_cms_regularization: bool = True,
        consolidate_on_task_switch: bool = True,
        adjacent_only: bool = False,
        **kwargs: Any,
    ):
        # Convert to list for multiple iterations
        params_list = list(params)

        # Initialize parent CollaborativeNestedOptimizer
        super().__init__(iter(params_list), adjacent_only=adjacent_only, **kwargs)

        # Store params reference for CMS
        self._params_list = params_list

        # Configuration
        self.use_cms_regularization = use_cms_regularization
        self.consolidate_on_task_switch = consolidate_on_task_switch

        # Create CMS with matching frequencies
        if cms_config is None:
            cms_config = CMSConfig(
                fast_frequency=self.fast_freq,
                medium_frequency=self.medium_freq,
                slow_frequency=self.slow_freq,
            )

        self.cms = ContinuumMemorySystem(iter(params_list), cms_config)

        # Track current task for task-switch detection
        self._current_task_id: Optional[int] = None

        # Memory bridge connections (connect optimizer states to CMS)
        self._memory_bridge_stats: Dict[str, Dict[str, float]] = {}

    def step(
        self, closure: Optional[Callable[[], torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Perform optimization step with CMS updates.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Extended dict including CMS update info.
        """
        # Call parent step (handles optimizer updates and bridges)
        result = super().step(closure)

        # Update CMS memory banks
        cms_update = self.cms.update(self._params_list)
        result["cms_update"] = cms_update

        # Accumulate importance from gradients
        self.cms.accumulate_importance(self._params_list)

        return result

    def get_regularization_loss(
        self,
        importance: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get CMS regularization loss to prevent forgetting.

        Call this before backward() and add to your loss.

        Args:
            importance: Optional importance weights.

        Returns:
            Scalar regularization loss tensor.
        """
        if not self.use_cms_regularization:
            return torch.tensor(0.0)

        return self.cms.compute_regularization_loss(
            self._params_list,
            importance,
        )

    def consolidate_memory(
        self,
        importance: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, int]:
        """Consolidate knowledge from fast → medium → slow.

        Call this at task boundaries to preserve important knowledge.

        Args:
            importance: Optional pre-computed importance.

        Returns:
            Dictionary with consolidation counts.
        """
        result = self.cms.consolidate(self._params_list, importance)

        # Reset importance accumulator after consolidation
        self.cms.reset_importance()

        return result

    def set_task(self, task_id: int) -> Dict[str, Any]:
        """Signal a task switch.

        Call this when switching to a new task. If consolidate_on_task_switch
        is True, will automatically consolidate memory.

        Args:
            task_id: ID of the new task.

        Returns:
            Dictionary with task switch info.
        """
        result: Dict[str, Any] = {
            "previous_task": self._current_task_id,
            "new_task": task_id,
            "consolidated": False,
        }

        if self._current_task_id is not None and self.consolidate_on_task_switch:
            consolidation = self.consolidate_memory()
            result["consolidation"] = consolidation
            result["consolidated"] = True

        self._current_task_id = task_id

        return result

    def get_memory_drift(self) -> Dict[str, float]:
        """Get mean parameter drift from each memory bank.

        Useful for monitoring forgetting.

        Returns:
            Dictionary with mean drift per bank.
        """
        drift = self.cms.compute_drift(self._params_list)

        result: Dict[str, float] = {}
        for bank_name, bank_drift in drift.items():
            if bank_drift:
                result[bank_name] = sum(bank_drift.values()) / len(bank_drift)
            else:
                result[bank_name] = 0.0

        return result

    def get_cms_stats(self) -> Dict[str, Any]:
        """Get CMS statistics.

        Returns:
            Dictionary with CMS stats.
        """
        return self.cms.get_stats()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get full diagnostics including CMS.

        Returns:
            Extended diagnostics dictionary.
        """
        diag = super().get_diagnostics()
        diag["cms"] = self.get_cms_stats()
        diag["memory_drift"] = self.get_memory_drift()
        diag["current_task"] = self._current_task_id
        return diag

    def state_dict(self) -> Dict[str, Any]:
        """Return state for serialization.

        Returns:
            Dictionary containing optimizer state.
        """
        state = super().state_dict()
        state["cms"] = self.cms.state_dict()
        state["current_task_id"] = self._current_task_id
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from serialization.

        Args:
            state_dict: Dictionary containing optimizer state.
        """
        super().load_state_dict(state_dict)
        if "cms" in state_dict:
            self.cms.load_state_dict(state_dict["cms"])
        self._current_task_id = state_dict.get("current_task_id")


def _test_collaborative_cms_optimizer():
    """Test the CollaborativeCMSOptimizer."""
    import torch.nn.functional as F

    print("Testing CollaborativeCMSOptimizer...")

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )

    # Create optimizer
    optimizer = CollaborativeCMSOptimizer(
        model.parameters(),
        fast_lr=0.01,
        medium_lr=0.005,
        slow_lr=0.001,
        fast_freq=1,
        medium_freq=5,
        slow_freq=20,
        bridge_threshold=0.3,
        bridge_frequency=5,
        hidden_dim=32,
        use_cms_regularization=True,
    )

    # Simulate multi-task learning
    torch.manual_seed(42)

    print("\n  Simulating multi-task learning...")

    for task_id in range(3):
        # Signal task switch
        task_info = optimizer.set_task(task_id)
        print(f"\n  Task {task_id}: {task_info}")

        # Generate task-specific data
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        # Train on task
        for step in range(50):
            optimizer.zero_grad()

            # Forward pass
            output = model(x)
            loss = F.cross_entropy(output, y)

            # Add CMS regularization
            reg_loss = optimizer.get_regularization_loss()
            total_loss = loss + reg_loss

            # Backward
            total_loss.backward()

            # Step
            optimizer.step()

            if step % 20 == 0:
                drift = optimizer.get_memory_drift()
                print(
                    f"    Step {step}: loss={loss.item():.4f}, "
                    f"reg={reg_loss.item():.6f}, "
                    f"drift_slow={drift.get('slow', 0):.6f}"
                )

    # Final diagnostics
    print("\n  Final diagnostics:")
    diag = optimizer.get_diagnostics()
    print(f"    Step count: {diag['step_count']}")
    print(f"    CMS fast updates: {diag['cms']['fast']['total_updates']}")
    print(f"    CMS slow updates: {diag['cms']['slow']['total_updates']}")

    # Bridge stats
    print("\n  Bridge statistics:")
    bridge_stats = optimizer.get_bridge_stats()
    for bridge_name, stats in bridge_stats.items():
        print(f"    {bridge_name}: transfer_rate={stats['transfer_rate']:.2f}")

    print("\n✓ CollaborativeCMSOptimizer test passed!")


if __name__ == "__main__":
    _test_collaborative_cms_optimizer()
