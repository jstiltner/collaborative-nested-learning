"""Multi-Scale Nested Learning Optimizer.

Generalizes nested learning to N timescale levels with full bidirectional bridges.
Inspired by brain wave hierarchies (gamma → delta).

The human brain processes information at multiple timescales:
- Gamma (30-100 Hz): Immediate patterns, working memory binding
- Beta (12-30 Hz): Active thinking, focus
- Alpha (8-12 Hz): Relaxed awareness, memory consolidation
- Theta (4-8 Hz): Deep memory, episodic encoding
- Delta (0.5-4 Hz): Deep sleep, long-term consolidation

We use geometric frequency progression to mirror this natural hierarchy.

Reference: Nested Learning (Behrouz et al., NeurIPS 2025)
Our extension: Arbitrary depth with full bridge connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from src.optimizers.deep_momentum import DeepMomentumOptimizer
from src.bridges.knowledge_bridges import KnowledgeBridge


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale nested learning.
    
    Args:
        num_levels: Number of timescale levels (3, 5, 7, etc.)
        base_lr: Learning rate for fastest level.
        lr_decay: Multiplicative decay per level (slower = lower lr).
            Brain analogy: slower processes have lower "learning rates".
        freq_ratio: Frequency ratio between adjacent levels.
            Uses geometric progression like brain waves.
        hidden_dim: Hidden dimension for memory networks.
        bridge_threshold: Gate threshold for knowledge bridges.
        bridge_frequency: How often to attempt bridge transfers.
    
    Example:
        >>> config = MultiScaleConfig(num_levels=5, freq_ratio=5.0)
        >>> print(config.get_frequencies())
        [1, 5, 25, 125, 625]
    """
    
    num_levels: int = 3
    base_lr: float = 0.01
    lr_decay: float = 0.7
    freq_ratio: float = 5.0
    hidden_dim: int = 64
    bridge_threshold: float = 0.3
    bridge_frequency: int = 10
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {self.num_levels}")
        if self.freq_ratio <= 1.0:
            raise ValueError(f"freq_ratio must be > 1.0, got {self.freq_ratio}")
        if self.base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {self.base_lr}")
        if not 0 < self.lr_decay <= 1:
            raise ValueError(f"lr_decay must be in (0, 1], got {self.lr_decay}")
    
    def get_learning_rates(self) -> List[float]:
        """Get learning rate for each level (fastest to slowest).
        
        Returns:
            List of learning rates, decreasing geometrically.
        """
        return [self.base_lr * (self.lr_decay ** i) for i in range(self.num_levels)]
    
    def get_frequencies(self) -> List[int]:
        """Get update frequency for each level (fastest to slowest).
        
        Returns:
            List of frequencies (steps between updates).
        """
        return [max(1, int(self.freq_ratio ** i)) for i in range(self.num_levels)]
    
    def get_level_names(self) -> List[str]:
        """Get human-readable names for each level.
        
        Returns:
            List of level names inspired by brain waves.
        """
        names = ['gamma', 'beta', 'alpha', 'theta', 'delta', 'infra_slow', 'ultra_slow']
        if self.num_levels <= len(names):
            return names[:self.num_levels]
        return [f'level_{i}' for i in range(self.num_levels)]


class MultiScaleBridges(nn.Module):
    """Full bidirectional bridges between all timescale levels.
    
    For N levels, creates N*(N-1) bridges (all pairs, both directions).
    This enables knowledge to flow freely between any two timescales.
    
    Args:
        num_levels: Number of timescale levels.
        hidden_dim: Hidden dimension for bridge networks.
        gate_threshold: Minimum gate value to allow transfer.
    
    Example:
        >>> bridges = MultiScaleBridges(num_levels=5, hidden_dim=64)
        >>> print(len(bridges.bridges))  # 5*4 = 20 bridges
        20
    """
    
    def __init__(
        self,
        num_levels: int,
        hidden_dim: int,
        gate_threshold: float = 0.3,
    ):
        super().__init__()
        
        if num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {num_levels}")
        
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.gate_threshold = gate_threshold
        
        # Create bridge for each ordered pair of levels
        self.bridges = nn.ModuleDict()
        for i in range(num_levels):
            for j in range(num_levels):
                if i != j:
                    key = f"{i}_to_{j}"
                    self.bridges[key] = KnowledgeBridge(
                        hidden_dim=hidden_dim,
                        gate_threshold=gate_threshold,
                    )
    
    def forward(
        self,
        states: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, Dict[str, Any]]]:
        """Compute knowledge transfers between all levels.
        
        Args:
            states: List of state tensors, one per level.
                Each tensor has shape (hidden_dim,).
        
        Returns:
            Tuple of:
                - List of update tensors to add to each level's state.
                - Dictionary with transfer info for each bridge.
        
        Raises:
            ValueError: If number of states doesn't match num_levels.
        """
        if len(states) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} states, got {len(states)}"
            )
        
        updates = [torch.zeros_like(s) for s in states]
        transfer_info: Dict[str, Dict[str, Any]] = {}
        
        for i in range(self.num_levels):
            for j in range(self.num_levels):
                if i != j:
                    key = f"{i}_to_{j}"
                    bridge = self.bridges[key]
                    
                    knowledge, gate = bridge(states[i], states[j])
                    
                    if gate >= self.gate_threshold:
                        updates[j] = updates[j] + knowledge
                    
                    transfer_info[key] = {
                        'gate': gate,
                        'transferred': gate >= self.gate_threshold,
                    }
        
        return updates, transfer_info
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all bridges.
        
        Returns:
            Dictionary mapping bridge name to its statistics.
        """
        return {key: bridge.get_stats() for key, bridge in self.bridges.items()}
    
    def reset_stats(self) -> None:
        """Reset statistics for all bridges."""
        for bridge in self.bridges.values():
            bridge.reset_stats()
    
    def get_bridge_count(self) -> int:
        """Get total number of bridges.
        
        Returns:
            Number of bridges (N*(N-1) for N levels).
        """
        return len(self.bridges)


class MultiScaleNestedOptimizer(torch.optim.Optimizer):
    """Multi-scale nested optimizer with N timescale levels.
    
    Generalizes the 3-level NestedOptimizer to arbitrary depth.
    Each level has its own DeepMomentumOptimizer with different
    learning rate and update frequency.
    
    Brain-inspired design:
    - Faster levels (gamma, beta) capture immediate patterns
    - Slower levels (theta, delta) consolidate long-term knowledge
    - Bridges enable bidirectional knowledge flow
    
    Args:
        params: Model parameters to optimize.
        config: MultiScaleConfig with all hyperparameters.
    
    Example:
        >>> config = MultiScaleConfig(num_levels=5, freq_ratio=5.0)
        >>> optimizer = MultiScaleNestedOptimizer(model.parameters(), config)
        >>> for x, y in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(x), y)
        ...     loss.backward()
        ...     result = optimizer.step()
        ...     print(f"Updated levels: {result['levels_updated']}")
    """
    
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        config: Optional[MultiScaleConfig] = None,
    ):
        if config is None:
            config = MultiScaleConfig()
        
        self.config = config
        self.num_levels = config.num_levels
        
        # Convert params to list for multiple iterations
        params_list = list(params)
        
        # Initialize base optimizer (required by PyTorch)
        defaults = {'lr': config.base_lr}
        super().__init__(params_list, defaults)
        
        # Get learning rates and frequencies
        learning_rates = config.get_learning_rates()
        self.frequencies = config.get_frequencies()
        self.level_names = config.get_level_names()
        
        # Create optimizer for each level
        self.optimizers: List[DeepMomentumOptimizer] = []
        for i in range(self.num_levels):
            opt = DeepMomentumOptimizer(
                iter(params_list),
                lr=learning_rates[i],
                hidden_dim=config.hidden_dim,
            )
            self.optimizers.append(opt)
        
        # Create bridges between all levels
        self.bridges = MultiScaleBridges(
            num_levels=self.num_levels,
            hidden_dim=config.hidden_dim,
            gate_threshold=config.bridge_threshold,
        )
        
        # Gradient accumulators for slower levels
        self._grad_accumulators: List[Dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_levels)
        ]
        
        # Step counter
        self._step_count = 0
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients for all optimizers.
        
        Args:
            set_to_none: If True, set gradients to None instead of zero.
        """
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Dict[str, Any]:
        """Perform optimization step across all timescales.
        
        Each level updates at its own frequency:
        - Level 0 (gamma): Every step
        - Level 1 (beta): Every freq_ratio steps
        - Level 2 (alpha): Every freq_ratio^2 steps
        - etc.
        
        Args:
            closure: Optional closure for loss computation.
        
        Returns:
            Dictionary with:
                - step: Current step count
                - levels_updated: List of level names that updated
                - bridges: Bridge transfer information (if checked)
        """
        self._step_count += 1
        result: Dict[str, Any] = {
            'step': self._step_count,
            'levels_updated': [],
            'bridges': {},
        }
        
        # Accumulate gradients for all levels
        for param_group in self.param_groups:
            for i, p in enumerate(param_group['params']):
                if p.grad is not None:
                    for level in range(self.num_levels):
                        if i not in self._grad_accumulators[level]:
                            self._grad_accumulators[level][i] = torch.zeros_like(p.grad)
                        self._grad_accumulators[level][i].add_(p.grad)
        
        # Update each level at its frequency
        for level in range(self.num_levels):
            freq = self.frequencies[level]
            
            if self._step_count % freq == 0:
                # Apply accumulated gradients
                for param_group in self.param_groups:
                    for i, p in enumerate(param_group['params']):
                        if i in self._grad_accumulators[level]:
                            # Average the accumulated gradients
                            p.grad = self._grad_accumulators[level][i] / freq
                
                # Step this level's optimizer
                self.optimizers[level].step()
                result['levels_updated'].append(self.level_names[level])
                
                # Clear accumulator
                self._grad_accumulators[level] = {}
        
        # Attempt bridge transfers at specified frequency
        if self._step_count % self.config.bridge_frequency == 0:
            states = self.get_memory_states()
            updates, transfer_info = self.bridges(states)
            
            # Inject knowledge into each level
            for level, update in enumerate(updates):
                if update.abs().sum() > 0:
                    self.optimizers[level].inject_knowledge(update, gate=1.0)
            
            result['bridges'] = transfer_info
        
        return result
    
    def get_memory_states(self) -> List[torch.Tensor]:
        """Get knowledge states from all levels.
        
        Returns:
            List of state tensors, one per level.
        """
        return [opt.get_knowledge_state() for opt in self.optimizers]
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics for all levels.
        
        Returns:
            Dictionary with configuration and statistics.
        """
        return {
            'step_count': self._step_count,
            'num_levels': self.num_levels,
            'level_names': self.level_names,
            'frequencies': self.frequencies,
            'learning_rates': self.config.get_learning_rates(),
            'bridge_count': self.bridges.get_bridge_count(),
            'bridge_stats': self.bridges.get_stats(),
        }
    
    def get_bridge_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all bridges.
        
        Returns:
            Dictionary mapping bridge name to its statistics.
        """
        return self.bridges.get_stats()
    
    def reset_bridge_stats(self) -> None:
        """Reset statistics for all bridges."""
        self.bridges.reset_stats()


def _test_multi_scale():
    """Test the multi-scale optimizer."""
    import torch.nn.functional as F
    
    print("Testing MultiScaleNestedOptimizer...")
    
    # Test different configurations
    for num_levels in [3, 5, 7]:
        print(f"\n  Testing {num_levels} levels...")
        
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        
        # Create config
        config = MultiScaleConfig(
            num_levels=num_levels,
            freq_ratio=3.0,  # 3x between levels
            base_lr=0.01,
            hidden_dim=32,
        )
        
        print(f"    Frequencies: {config.get_frequencies()}")
        print(f"    Learning rates: {[f'{lr:.4f}' for lr in config.get_learning_rates()]}")
        print(f"    Level names: {config.get_level_names()}")
        
        # Create optimizer
        optimizer = MultiScaleNestedOptimizer(model.parameters(), config)
        
        print(f"    Bridge count: {optimizer.bridges.get_bridge_count()}")
        
        # Training loop
        torch.manual_seed(42)
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        
        initial_loss = F.cross_entropy(model(x), y).item()
        
        for step in range(100):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            result = optimizer.step()
        
        final_loss = F.cross_entropy(model(x), y).item()
        
        print(f"    Initial loss: {initial_loss:.4f}")
        print(f"    Final loss: {final_loss:.4f}")
        print(f"    Improvement: {initial_loss - final_loss:.4f}")
        
        assert final_loss < initial_loss, f"Loss should decrease for {num_levels} levels"
    
    print("\n✓ MultiScaleNestedOptimizer test passed!")


if __name__ == "__main__":
    _test_multi_scale()