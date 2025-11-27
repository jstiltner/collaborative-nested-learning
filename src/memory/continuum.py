"""Continuum Memory System: Multi-timescale memory for continual learning.

This implements the Continuum Memory System from the Nested Learning paper.
Three memory banks at different timescales work together:

- Fast bank: Updates every step, captures immediate patterns
- Medium bank: Updates every N steps, captures short-term patterns  
- Slow bank: Updates every M steps, captures long-term principles

The key insight is that slower banks naturally preserve old knowledge
because they update less frequently, preventing catastrophic forgetting.

Reference: Nested Learning (Behrouz et al., NeurIPS 2025), Section 2.3
https://abehrouz.github.io/files/NL.pdf

Our extension: Bidirectional knowledge bridges between banks enable
explicit cross-timescale learning (see src/bridges/knowledge_bridges.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from src.memory.memory_bank import MemoryBank


@dataclass
class CMSConfig:
    """Configuration for Continuum Memory System.
    
    Attributes:
        fast_frequency: Update frequency for fast bank (default: 1).
        medium_frequency: Update frequency for medium bank (default: 10).
        slow_frequency: Update frequency for slow bank (default: 100).
        fast_decay: EMA decay for fast bank (default: 0.0 = instant).
        medium_decay: EMA decay for medium bank (default: 0.5).
        slow_decay: EMA decay for slow bank (default: 0.9).
        consolidation_threshold: Importance threshold for consolidation.
        regularization_strength: Base strength for memory regularization.
    """
    fast_frequency: int = 1
    medium_frequency: int = 10
    slow_frequency: int = 100
    fast_decay: float = 0.0
    medium_decay: float = 0.5
    slow_decay: float = 0.9
    consolidation_threshold: float = 0.1
    regularization_strength: float = 0.01


class ContinuumMemorySystem:
    """Multi-timescale memory system from Nested Learning.
    
    This system manages three memory banks at different timescales:
    - Fast: Captures immediate patterns, updates frequently
    - Medium: Captures short-term patterns, updates moderately
    - Slow: Captures long-term principles, updates rarely
    
    The system provides:
    1. Automatic memory updates at appropriate frequencies
    2. Knowledge consolidation from fast → medium → slow
    3. Regularization loss to prevent forgetting
    4. Memory retrieval for knowledge injection
    
    Reference: Nested Learning (Behrouz et al., NeurIPS 2025)
    
    Args:
        params: Model parameters to track.
        config: CMS configuration (uses defaults if None).
    
    Example:
        >>> model = nn.Linear(10, 5)
        >>> cms = ContinuumMemorySystem(model.parameters())
        >>> for step, (x, y) in enumerate(dataloader):
        ...     # Forward pass
        ...     loss = criterion(model(x), y)
        ...     
        ...     # Add memory regularization
        ...     reg_loss = cms.compute_regularization_loss(model.parameters())
        ...     total_loss = loss + reg_loss
        ...     
        ...     # Backward and optimize
        ...     total_loss.backward()
        ...     optimizer.step()
        ...     
        ...     # Update memory banks
        ...     cms.update(model.parameters())
    """
    
    def __init__(
        self,
        params: Optional[Iterable[nn.Parameter]] = None,
        config: Optional[CMSConfig] = None,
    ):
        self.config = config or CMSConfig()
        
        # Validate config
        if self.config.fast_frequency > self.config.medium_frequency:
            raise ValueError(
                f"fast_frequency ({self.config.fast_frequency}) must be <= "
                f"medium_frequency ({self.config.medium_frequency})"
            )
        if self.config.medium_frequency > self.config.slow_frequency:
            raise ValueError(
                f"medium_frequency ({self.config.medium_frequency}) must be <= "
                f"slow_frequency ({self.config.slow_frequency})"
            )
        
        # Create memory banks
        self.fast = MemoryBank(
            update_frequency=self.config.fast_frequency,
            decay_rate=self.config.fast_decay,
            name="fast",
        )
        self.medium = MemoryBank(
            update_frequency=self.config.medium_frequency,
            decay_rate=self.config.medium_decay,
            name="medium",
        )
        self.slow = MemoryBank(
            update_frequency=self.config.slow_frequency,
            decay_rate=self.config.slow_decay,
            name="slow",
        )
        
        # Initialize with parameters if provided
        if params is not None:
            params_list = list(params)
            self.fast.force_update(params_list)
            self.medium.force_update(params_list)
            self.slow.force_update(params_list)
        
        # Global step counter
        self._step_count = 0
        
        # Track which banks updated in last step
        self._last_update_info: Dict[str, bool] = {
            'fast': False,
            'medium': False,
            'slow': False,
        }
        
        # Accumulated importance for consolidation
        self._importance_accum: Dict[int, torch.Tensor] = {}
        self._importance_count = 0
    
    @property
    def step_count(self) -> int:
        """Current global step count."""
        return self._step_count
    
    def update(self, params: Iterable[nn.Parameter]) -> Dict[str, bool]:
        """Update all memory banks based on their frequencies.
        
        Args:
            params: Current model parameters.
        
        Returns:
            Dictionary indicating which banks were updated.
        """
        self._step_count += 1
        params_list = list(params)
        
        # Update each bank
        fast_updated = self.fast.maybe_update(params_list)
        medium_updated = self.medium.maybe_update(params_list)
        slow_updated = self.slow.maybe_update(params_list)
        
        self._last_update_info = {
            'fast': fast_updated,
            'medium': medium_updated,
            'slow': slow_updated,
        }
        
        return self._last_update_info
    
    def accumulate_importance(self, params: Iterable[nn.Parameter]) -> None:
        """Accumulate importance scores from gradients.
        
        Call this after backward() to track which parameters are important
        for the current task. Used for consolidation decisions.
        
        Args:
            params: Parameters with gradients.
        """
        for p in params:
            if p.grad is None:
                continue
            
            param_id = id(p)
            importance = p.grad.abs()
            
            if param_id not in self._importance_accum:
                self._importance_accum[param_id] = importance.clone()
            else:
                self._importance_accum[param_id].add_(importance)
        
        self._importance_count += 1
    
    def get_importance(self) -> Dict[int, torch.Tensor]:
        """Get averaged importance scores.
        
        Returns:
            Dictionary mapping param_id to importance tensor.
        """
        if self._importance_count == 0:
            return {}
        
        return {
            k: v / self._importance_count 
            for k, v in self._importance_accum.items()
        }
    
    def reset_importance(self) -> None:
        """Reset accumulated importance scores."""
        self._importance_accum.clear()
        self._importance_count = 0
    
    def consolidate(
        self,
        params: Iterable[nn.Parameter],
        importance: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, int]:
        """Consolidate knowledge from fast → medium → slow.
        
        This is called periodically (e.g., at task boundaries) to
        transfer important patterns to slower memory banks.
        
        Args:
            params: Current model parameters.
            importance: Optional pre-computed importance. Uses accumulated if None.
        
        Returns:
            Dictionary with consolidation counts per direction.
        """
        if importance is None:
            importance = self.get_importance()
        
        params_list = list(params)
        threshold = self.config.consolidation_threshold
        
        # Fast → Medium
        fast_to_medium = self.fast.consolidate_to(
            self.medium, params_list, importance, threshold
        )
        
        # Medium → Slow
        medium_to_slow = self.medium.consolidate_to(
            self.slow, params_list, importance, threshold
        )
        
        return {
            'fast_to_medium': fast_to_medium,
            'medium_to_slow': medium_to_slow,
        }
    
    def compute_regularization_loss(
        self,
        params: Iterable[nn.Parameter],
        importance: Optional[Dict[int, torch.Tensor]] = None,
        use_slow_only: bool = False,
    ) -> torch.Tensor:
        """Compute regularization loss to prevent forgetting.
        
        Uses stored memories to penalize parameter drift. Slower banks
        contribute more to regularization (they represent more stable knowledge).
        
        Args:
            params: Current model parameters.
            importance: Optional importance weights.
            use_slow_only: If True, only use slow bank for regularization.
        
        Returns:
            Scalar regularization loss tensor.
        """
        params_list = list(params)
        strength = self.config.regularization_strength
        
        if use_slow_only:
            # Only slow bank (most stable knowledge)
            return self.slow.compute_regularization_loss(
                params_list, importance, strength
            )
        
        # Weighted combination: slow > medium > fast
        # Slow bank has most stable knowledge, so weight it highest
        slow_loss = self.slow.compute_regularization_loss(
            params_list, importance, strength * 1.0
        )
        medium_loss = self.medium.compute_regularization_loss(
            params_list, importance, strength * 0.5
        )
        fast_loss = self.fast.compute_regularization_loss(
            params_list, importance, strength * 0.1
        )
        
        return slow_loss + medium_loss + fast_loss
    
    def get_memory(
        self,
        param_id: int,
        bank: str = "slow",
    ) -> Optional[torch.Tensor]:
        """Retrieve memory from a specific bank.
        
        Args:
            param_id: The id() of the parameter.
            bank: Which bank to retrieve from ("fast", "medium", "slow").
        
        Returns:
            Memory tensor if available, None otherwise.
        """
        if bank == "fast":
            return self.fast.get_memory(param_id)
        elif bank == "medium":
            return self.medium.get_memory(param_id)
        elif bank == "slow":
            return self.slow.get_memory(param_id)
        else:
            raise ValueError(f"Unknown bank: {bank}")
    
    def get_consolidated_memory(
        self,
        param_id: int,
        weights: Tuple[float, float, float] = (0.1, 0.3, 0.6),
    ) -> Optional[torch.Tensor]:
        """Get weighted combination of memories from all banks.
        
        Args:
            param_id: The id() of the parameter.
            weights: Weights for (fast, medium, slow) banks.
        
        Returns:
            Weighted memory tensor if available, None otherwise.
        """
        fast_mem = self.fast.get_memory(param_id)
        medium_mem = self.medium.get_memory(param_id)
        slow_mem = self.slow.get_memory(param_id)
        
        if slow_mem is None:
            return None
        
        # Start with slow (always available if any are)
        result = slow_mem.clone() * weights[2]
        
        if medium_mem is not None:
            result.add_(medium_mem, alpha=weights[1])
        
        if fast_mem is not None:
            result.add_(fast_mem, alpha=weights[0])
        
        # Normalize
        total_weight = sum(weights)
        result.div_(total_weight)
        
        return result
    
    def compute_drift(
        self,
        params: Iterable[nn.Parameter],
    ) -> Dict[str, Dict[int, float]]:
        """Compute parameter drift from each memory bank.
        
        Args:
            params: Current model parameters.
        
        Returns:
            Nested dict: bank_name -> param_id -> drift_distance.
        """
        params_list = list(params)
        
        return {
            'fast': self.fast.compute_drift(params_list),
            'medium': self.medium.compute_drift(params_list),
            'slow': self.slow.compute_drift(params_list),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all banks.
        
        Returns:
            Dictionary with stats for each bank.
        """
        return {
            'step_count': self._step_count,
            'importance_count': self._importance_count,
            'fast': self.fast.get_stats(),
            'medium': self.medium.get_stats(),
            'slow': self.slow.get_stats(),
            'last_update': self._last_update_info,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state for serialization.
        
        Returns:
            Dictionary containing CMS state.
        """
        return {
            'config': {
                'fast_frequency': self.config.fast_frequency,
                'medium_frequency': self.config.medium_frequency,
                'slow_frequency': self.config.slow_frequency,
                'fast_decay': self.config.fast_decay,
                'medium_decay': self.config.medium_decay,
                'slow_decay': self.config.slow_decay,
                'consolidation_threshold': self.config.consolidation_threshold,
                'regularization_strength': self.config.regularization_strength,
            },
            'step_count': self._step_count,
            'fast': self.fast.state_dict(),
            'medium': self.medium.state_dict(),
            'slow': self.slow.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from serialization.
        
        Args:
            state_dict: Dictionary containing CMS state.
        """
        config = state_dict.get('config', {})
        self.config = CMSConfig(
            fast_frequency=config.get('fast_frequency', 1),
            medium_frequency=config.get('medium_frequency', 10),
            slow_frequency=config.get('slow_frequency', 100),
            fast_decay=config.get('fast_decay', 0.0),
            medium_decay=config.get('medium_decay', 0.5),
            slow_decay=config.get('slow_decay', 0.9),
            consolidation_threshold=config.get('consolidation_threshold', 0.1),
            regularization_strength=config.get('regularization_strength', 0.01),
        )
        
        self._step_count = state_dict['step_count']
        self.fast.load_state_dict(state_dict['fast'])
        self.medium.load_state_dict(state_dict['medium'])
        self.slow.load_state_dict(state_dict['slow'])
    
    def reset(self) -> None:
        """Reset all memory banks."""
        self.fast.reset()
        self.medium.reset()
        self.slow.reset()
        self._step_count = 0
        self._importance_accum.clear()
        self._importance_count = 0
        self._last_update_info = {'fast': False, 'medium': False, 'slow': False}


def _test_continuum_memory_system():
    """Test the ContinuumMemorySystem class."""
    import torch.nn.functional as F
    
    print("Testing ContinuumMemorySystem...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )
    
    # Create CMS with custom config
    config = CMSConfig(
        fast_frequency=1,
        medium_frequency=5,
        slow_frequency=20,
        regularization_strength=0.01,
    )
    cms = ContinuumMemorySystem(model.parameters(), config)
    
    # Dummy data
    torch.manual_seed(42)
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    
    # Simulate training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print("  Simulating training...")
    for step in range(50):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        # Add regularization
        reg_loss = cms.compute_regularization_loss(model.parameters())
        total_loss = loss + reg_loss
        
        # Backward
        total_loss.backward()
        
        # Accumulate importance
        cms.accumulate_importance(model.parameters())
        
        # Optimize
        optimizer.step()
        
        # Update memory banks
        update_info = cms.update(model.parameters())
        
        if step % 10 == 0:
            print(f"    Step {step}: loss={loss.item():.4f}, "
                  f"reg={reg_loss.item():.6f}, "
                  f"updates={update_info}")
    
    # Test consolidation
    print("\n  Testing consolidation...")
    consolidation = cms.consolidate(model.parameters())
    print(f"    Consolidation: {consolidation}")
    
    # Test drift computation
    print("\n  Testing drift computation...")
    drift = cms.compute_drift(model.parameters())
    for bank_name, bank_drift in drift.items():
        if bank_drift:
            mean_drift = sum(bank_drift.values()) / len(bank_drift)
            print(f"    {bank_name} mean drift: {mean_drift:.6f}")
    
    # Test memory retrieval
    print("\n  Testing memory retrieval...")
    for p in model.parameters():
        param_id = id(p)
        consolidated = cms.get_consolidated_memory(param_id)
        if consolidated is not None:
            print(f"    Consolidated memory shape: {consolidated.shape}")
            break
    
    # Get stats
    stats = cms.get_stats()
    print(f"\n  Final stats:")
    print(f"    Step count: {stats['step_count']}")
    print(f"    Fast updates: {stats['fast']['total_updates']}")
    print(f"    Medium updates: {stats['medium']['total_updates']}")
    print(f"    Slow updates: {stats['slow']['total_updates']}")
    
    print("\n✓ ContinuumMemorySystem test passed!")


if __name__ == "__main__":
    _test_continuum_memory_system()