"""Knowledge Bridges: Bidirectional attention-gated knowledge transfer.

This is our NOVEL contribution beyond the original Nested Learning paper.

The paper's information flow is unidirectional:
    fast → medium → slow (through forward pass only)

We add explicit bidirectional bridges:
    fast ←→ medium ←→ slow (through learned attention-gated transfer)

This allows:
- Fast patterns to inform medium-term learning
- Slow principles to guide fast adaptation
- Adaptive gating to prevent noise propagation

Reference: This is our extension to Nested Learning (Behrouz et al., NeurIPS 2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from src.optimizers.nested_optimizer import NestedOptimizer


@dataclass
class BridgeStats:
    """Statistics for a knowledge bridge."""
    
    total_calls: int = 0
    total_transfers: int = 0
    gate_values: List[float] = field(default_factory=list)
    
    @property
    def transfer_rate(self) -> float:
        """Fraction of calls that resulted in transfer."""
        if self.total_calls == 0:
            return 0.0
        return self.total_transfers / self.total_calls
    
    @property
    def mean_gate(self) -> float:
        """Mean gate value across all calls."""
        if not self.gate_values:
            return 0.0
        return sum(self.gate_values) / len(self.gate_values)
    
    def record(self, gate_value: float, transferred: bool) -> None:
        """Record a bridge call."""
        self.total_calls += 1
        if transferred:
            self.total_transfers += 1
        self.gate_values.append(gate_value)
        # Keep only last 1000 values to prevent memory growth
        if len(self.gate_values) > 1000:
            self.gate_values = self.gate_values[-1000:]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_calls': self.total_calls,
            'total_transfers': self.total_transfers,
            'transfer_rate': self.transfer_rate,
            'mean_gate': self.mean_gate,
        }


class KnowledgeBridge(nn.Module):
    """Attention-gated knowledge transfer between optimizer timescales.
    
    This is our NOVEL contribution beyond the original paper.
    
    The bridge learns:
    1. A transformation to map source knowledge to target space
    2. A gating mechanism to decide when transfer is beneficial
    
    Args:
        hidden_dim: Dimension of knowledge states (must match optimizer hidden_dim).
        gate_threshold: Minimum gate value to allow transfer (prevents noise).
        gate_temperature: Temperature for gate sigmoid (higher = softer gating).
        transform_layers: Number of layers in the transform network.
    
    Example:
        >>> bridge = KnowledgeBridge(hidden_dim=64)
        >>> source = torch.randn(64)
        >>> target = torch.randn(64)
        >>> knowledge, gate = bridge(source, target)
        >>> if gate > 0.5:
        ...     optimizer.inject_knowledge(knowledge, gate=gate)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        gate_threshold: float = 0.5,
        gate_temperature: float = 1.0,
        transform_layers: int = 2,
    ):
        super().__init__()
        
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if not 0.0 <= gate_threshold <= 1.0:
            raise ValueError(f"gate_threshold must be in [0, 1], got {gate_threshold}")
        if gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be > 0, got {gate_temperature}")
        
        self.hidden_dim = hidden_dim
        self.gate_threshold = gate_threshold
        self.gate_temperature = gate_temperature
        
        # Transform network: maps source knowledge to target space
        transform_modules: List[nn.Module] = []
        for i in range(transform_layers):
            transform_modules.append(nn.Linear(hidden_dim, hidden_dim))
            if i < transform_layers - 1:  # No activation on last layer
                transform_modules.append(nn.ReLU())
        self.transform = nn.Sequential(*transform_modules)
        
        # Gate network: decides whether to transfer
        # Input: concatenation of source and target states
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Statistics tracking
        self.stats = BridgeStats()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        source_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Compute gated knowledge transfer.
        
        Uses a hybrid approach:
        1. Similarity-based gate: High similarity = more transfer
        2. Learned transform: Maps source knowledge to target space
        
        Args:
            source_state: Knowledge state from source optimizer, shape (hidden_dim,).
            target_state: Knowledge state from target optimizer, shape (hidden_dim,).
        
        Returns:
            Tuple of:
                - transferred_knowledge: Tensor(hidden_dim,) to inject into target.
                  Returns zeros if gate < threshold.
                - gate_value: float in [0, 1] indicating transfer confidence.
        """
        # Validate input shapes
        if source_state.shape != (self.hidden_dim,):
            raise ValueError(
                f"Expected source_state shape ({self.hidden_dim},), "
                f"got {source_state.shape}"
            )
        if target_state.shape != (self.hidden_dim,):
            raise ValueError(
                f"Expected target_state shape ({self.hidden_dim},), "
                f"got {target_state.shape}"
            )
        
        # Compute similarity-based gate (cosine similarity mapped to [0, 1])
        # This provides adaptive gating without requiring training
        source_norm = source_state / (source_state.norm() + 1e-8)
        target_norm = target_state / (target_state.norm() + 1e-8)
        similarity = torch.dot(source_norm, target_norm)  # [-1, 1]
        
        # Also compute learned gate for comparison/blending
        gate_input = torch.cat([source_state, target_state], dim=-1)
        gate_logit = self.gate(gate_input) / self.gate_temperature
        learned_gate = torch.sigmoid(gate_logit).squeeze()
        
        # Blend similarity and learned gate (similarity-weighted)
        # When states are similar, trust the transfer more
        # Map similarity from [-1, 1] to [0, 1]
        similarity_gate = (similarity + 1) / 2
        
        # Use similarity as primary gate (it's adaptive without training)
        # Learned gate can modulate but similarity drives the decision
        gate_value = similarity_gate * 0.7 + learned_gate * 0.3
        gate_float = gate_value.item()
        
        # Check if transfer should happen
        if gate_float >= self.gate_threshold:
            # Transform source knowledge
            knowledge = self.transform(source_state)
            # Scale by gate value for soft gating
            knowledge = knowledge * gate_value
            transferred = True
        else:
            # No transfer - return zeros
            knowledge = torch.zeros_like(source_state)
            transferred = False
        
        # Record statistics
        self.stats.record(gate_float, transferred)
        
        return knowledge, gate_float
    
    def get_stats(self) -> Dict[str, float]:
        """Get bridge statistics.
        
        Returns:
            Dictionary with statistics.
        """
        return self.stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = BridgeStats()


class CollaborativeNestedOptimizer(NestedOptimizer):
    """Nested optimizer with bidirectional knowledge bridges.
    
    This is our NOVEL contribution: explicit cross-timescale learning.
    
    We add 6 bidirectional bridges (or 4 in adjacent-only mode):
    - fast → medium: Fast patterns inform medium-term learning
    - medium → slow: Medium patterns inform long-term learning
    - slow → fast: Slow principles guide fast adaptation (NOVEL)
    - medium → fast: Medium patterns guide fast adaptation (NOVEL)
    - slow → medium: Slow principles guide medium learning (NOVEL)
    - fast → slow: Fast patterns inform long-term (less common)
    
    When adjacent_only=True, only adjacent timescale transfers are enabled:
    - fast ↔ medium (bidirectional)
    - medium ↔ slow (bidirectional)
    This prevents noise propagation from fast directly to slow memory.
    
    Args:
        params: Model parameters to optimize.
        bridge_threshold: Gate threshold for all bridges.
        bridge_frequency: Attempt knowledge transfer every N steps.
        bridge_hidden_dim: Hidden dimension for bridges (defaults to optimizer hidden_dim).
        enable_reverse_bridges: Enable slow→fast and medium→fast bridges.
        adjacent_only: Only enable adjacent timescale bridges (fast↔medium, medium↔slow).
            When True, fast↔slow direct transfers are disabled.
        **kwargs: Arguments passed to NestedOptimizer.
    
    Example:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = CollaborativeNestedOptimizer(model.parameters())
        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(input), target)
        ...     loss.backward()
        ...     result = optimizer.step()
        ...     print(f"Bridges: {result['bridges']}")
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        bridge_threshold: float = 0.5,
        bridge_frequency: int = 10,
        bridge_hidden_dim: Optional[int] = None,
        enable_reverse_bridges: bool = True,
        adjacent_only: bool = False,
        **kwargs: Any,
    ):
        # Initialize parent NestedOptimizer
        super().__init__(params, **kwargs)
        
        self.bridge_threshold = bridge_threshold
        self.bridge_frequency = bridge_frequency
        self.enable_reverse_bridges = enable_reverse_bridges
        self.adjacent_only = adjacent_only
        
        # Use optimizer's hidden_dim if not specified
        bridge_dim = bridge_hidden_dim or self.hidden_dim
        
        # Create adjacent bridges (always enabled)
        # fast ↔ medium
        self.fast_to_medium = KnowledgeBridge(
            hidden_dim=bridge_dim,
            gate_threshold=bridge_threshold,
        )
        self.medium_to_fast = KnowledgeBridge(
            hidden_dim=bridge_dim,
            gate_threshold=bridge_threshold,
        ) if enable_reverse_bridges else None
        
        # medium ↔ slow
        self.medium_to_slow = KnowledgeBridge(
            hidden_dim=bridge_dim,
            gate_threshold=bridge_threshold,
        )
        self.slow_to_medium = KnowledgeBridge(
            hidden_dim=bridge_dim,
            gate_threshold=bridge_threshold,
        ) if enable_reverse_bridges else None
        
        # Create non-adjacent bridges (disabled when adjacent_only=True)
        # fast ↔ slow (direct, skips medium)
        if not adjacent_only:
            self.fast_to_slow = KnowledgeBridge(
                hidden_dim=bridge_dim,
                gate_threshold=bridge_threshold,
            )
            self.slow_to_fast = KnowledgeBridge(
                hidden_dim=bridge_dim,
                gate_threshold=bridge_threshold,
            ) if enable_reverse_bridges else None
        else:
            self.fast_to_slow = None
            self.slow_to_fast = None
        
        # Track bridge activity
        self._last_bridge_info: Dict[str, Dict[str, Any]] = {}
    
    def _attempt_knowledge_transfer(self) -> Dict[str, Dict[str, Any]]:
        """Attempt knowledge transfer through all enabled bridges.
        
        When adjacent_only=True, only adjacent timescale transfers are attempted:
        - fast ↔ medium
        - medium ↔ slow
        
        Returns:
            Dictionary with transfer results for each bridge.
        """
        results: Dict[str, Dict[str, Any]] = {}
        
        # Get current memory states
        states = self.get_memory_states()
        fast_state = states['fast']
        medium_state = states['medium']
        slow_state = states['slow']
        
        # === Adjacent bridges (always enabled) ===
        
        # Fast → Medium
        knowledge, gate = self.fast_to_medium(fast_state, medium_state)
        transferred = gate >= self.bridge_threshold
        if transferred:
            self.medium.inject_knowledge(knowledge, gate=gate)
        results['fast_to_medium'] = {'transferred': transferred, 'gate': gate}
        
        # Medium → Slow
        knowledge, gate = self.medium_to_slow(medium_state, slow_state)
        transferred = gate >= self.bridge_threshold
        if transferred:
            self.slow.inject_knowledge(knowledge, gate=gate)
        results['medium_to_slow'] = {'transferred': transferred, 'gate': gate}
        
        # Medium → Fast (reverse, adjacent)
        if self.medium_to_fast is not None:
            knowledge, gate = self.medium_to_fast(medium_state, fast_state)
            transferred = gate >= self.bridge_threshold
            if transferred:
                self.fast.inject_knowledge(knowledge, gate=gate)
            results['medium_to_fast'] = {'transferred': transferred, 'gate': gate}
        
        # Slow → Medium (reverse, adjacent)
        if self.slow_to_medium is not None:
            knowledge, gate = self.slow_to_medium(slow_state, medium_state)
            transferred = gate >= self.bridge_threshold
            if transferred:
                self.medium.inject_knowledge(knowledge, gate=gate)
            results['slow_to_medium'] = {'transferred': transferred, 'gate': gate}
        
        # === Non-adjacent bridges (disabled when adjacent_only=True) ===
        
        # Fast → Slow (direct, non-adjacent)
        if self.fast_to_slow is not None:
            knowledge, gate = self.fast_to_slow(fast_state, slow_state)
            transferred = gate >= self.bridge_threshold
            if transferred:
                self.slow.inject_knowledge(knowledge, gate=gate)
            results['fast_to_slow'] = {'transferred': transferred, 'gate': gate}
        
        # Slow → Fast (direct, non-adjacent, reverse)
        if self.slow_to_fast is not None:
            knowledge, gate = self.slow_to_fast(slow_state, fast_state)
            transferred = gate >= self.bridge_threshold
            if transferred:
                self.fast.inject_knowledge(knowledge, gate=gate)
            results['slow_to_fast'] = {'transferred': transferred, 'gate': gate}
        
        return results
    
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Dict[str, Any]:
        """Perform optimization step with potential knowledge transfer.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Extended dict including bridge activity.
        """
        # Call parent step
        result = super().step(closure)
        
        # Attempt knowledge transfer at specified frequency
        if self._step_count % self.bridge_frequency == 0:
            bridge_results = self._attempt_knowledge_transfer()
            self._last_bridge_info = bridge_results
            result['bridges'] = bridge_results
        else:
            result['bridges'] = {}
        
        return result
    
    def get_bridge_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all enabled bridges.
        
        Returns:
            Dictionary with stats for each enabled bridge.
        """
        stats: Dict[str, Dict[str, float]] = {
            'fast_to_medium': self.fast_to_medium.get_stats(),
            'medium_to_slow': self.medium_to_slow.get_stats(),
        }
        
        # Adjacent reverse bridges
        if self.medium_to_fast is not None:
            stats['medium_to_fast'] = self.medium_to_fast.get_stats()
        if self.slow_to_medium is not None:
            stats['slow_to_medium'] = self.slow_to_medium.get_stats()
        
        # Non-adjacent bridges (only if not adjacent_only)
        if self.fast_to_slow is not None:
            stats['fast_to_slow'] = self.fast_to_slow.get_stats()
        if self.slow_to_fast is not None:
            stats['slow_to_fast'] = self.slow_to_fast.get_stats()
        
        return stats
    
    def reset_bridge_stats(self) -> None:
        """Reset statistics for all enabled bridges."""
        self.fast_to_medium.reset_stats()
        self.medium_to_slow.reset_stats()
        
        # Adjacent reverse bridges
        if self.medium_to_fast is not None:
            self.medium_to_fast.reset_stats()
        if self.slow_to_medium is not None:
            self.slow_to_medium.reset_stats()
        
        # Non-adjacent bridges
        if self.fast_to_slow is not None:
            self.fast_to_slow.reset_stats()
        if self.slow_to_fast is not None:
            self.slow_to_fast.reset_stats()
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics including bridge information.
        
        Returns:
            Extended diagnostics dictionary.
        """
        diag = super().get_diagnostics()
        diag['bridge_threshold'] = self.bridge_threshold
        diag['bridge_frequency'] = self.bridge_frequency
        diag['enable_reverse_bridges'] = self.enable_reverse_bridges
        diag['adjacent_only'] = self.adjacent_only
        diag['bridge_stats'] = self.get_bridge_stats()
        diag['last_bridge_info'] = self._last_bridge_info
        return diag


def _test_knowledge_bridge():
    """Test the KnowledgeBridge class."""
    print("Testing KnowledgeBridge...")
    
    bridge = KnowledgeBridge(hidden_dim=64, gate_threshold=0.3)
    
    # Test forward pass
    source = torch.randn(64)
    target = torch.randn(64)
    
    knowledge, gate = bridge(source, target)
    
    print(f"  Gate value: {gate:.4f}")
    print(f"  Knowledge shape: {knowledge.shape}")
    print(f"  Knowledge norm: {knowledge.norm().item():.4f}")
    
    # Test multiple calls to build statistics
    for _ in range(100):
        source = torch.randn(64)
        target = torch.randn(64)
        bridge(source, target)
    
    stats = bridge.get_stats()
    print(f"  Stats: {stats}")
    
    print("✓ KnowledgeBridge test passed!")


def _test_collaborative_optimizer():
    """Test the CollaborativeNestedOptimizer."""
    import torch.nn.functional as F
    
    print("\nTesting CollaborativeNestedOptimizer...")
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    
    # Create collaborative optimizer
    optimizer = CollaborativeNestedOptimizer(
        model.parameters(),
        fast_lr=0.01,
        medium_lr=0.005,
        slow_lr=0.001,
        fast_freq=1,
        medium_freq=5,
        slow_freq=10,
        bridge_threshold=0.3,
        bridge_frequency=5,
        hidden_dim=32,
    )
    
    # Dummy data
    torch.manual_seed(42)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    # Training loop
    initial_loss = None
    bridge_transfers = 0
    
    for step in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        result = optimizer.step()
        
        # Count bridge transfers
        if 'bridges' in result and result['bridges']:
            for bridge_name, bridge_info in result['bridges'].items():
                if bridge_info.get('transferred', False):
                    bridge_transfers += 1
        
        if step % 20 == 0:
            bridges_active = len(result.get('bridges', {}))
            print(f"  Step {step}: loss = {loss.item():.4f}, bridges_checked = {bridges_active}")
    
    final_loss = F.cross_entropy(model(x), y).item()
    
    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {initial_loss - final_loss:.4f}")
    print(f"  Total bridge transfers: {bridge_transfers}")
    
    # Get bridge statistics
    stats = optimizer.get_bridge_stats()
    print(f"\n  Bridge statistics:")
    for bridge_name, bridge_stats in stats.items():
        print(f"    {bridge_name}: transfer_rate={bridge_stats['transfer_rate']:.2f}, "
              f"mean_gate={bridge_stats['mean_gate']:.3f}")
    
    assert final_loss < initial_loss, "Optimizer should reduce loss!"
    print("\n✓ CollaborativeNestedOptimizer test passed!")


if __name__ == "__main__":
    _test_knowledge_bridge()
    _test_collaborative_optimizer()