"""Nested Optimizer: Multi-timescale optimization with three update frequencies.

This implements the nested optimization structure from the Nested Learning paper,
where three optimizers operate at different timescales:
- Fast: updates every step (captures immediate patterns)
- Medium: updates every N steps (captures short-term patterns)
- Slow: updates every M steps (captures long-term principles)

Reference: Nested Learning (Behrouz et al., NeurIPS 2025), Section 2.2
https://abehrouz.github.io/files/NL.pdf

Key insight from paper (Definition 2):
    Components have different update frequencies f_A.
    A ≻ B means f_A > f_B (A updates more frequently than B).
    Components are sorted into "levels" by frequency.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from src.optimizers.deep_momentum import DeepMomentumOptimizer


class NestedOptimizer:
    """Multi-timescale optimizer with three update frequencies.
    
    This optimizer manages three DeepMomentumOptimizers that share the same
    parameters but maintain separate momentum states and update at different
    frequencies.
    
    Reference: Nested Learning (Behrouz et al., NeurIPS 2025), Section 2.2
    
    Args:
        params: Iterable of parameters to optimize.
        fast_lr: Learning rate for fast optimizer (default: 0.01).
        medium_lr: Learning rate for medium optimizer (default: 0.005).
        slow_lr: Learning rate for slow optimizer (default: 0.001).
        fast_freq: Update frequency for fast (1 = every step).
        medium_freq: Update frequency for medium (10 = every 10 steps).
        slow_freq: Update frequency for slow (100 = every 100 steps).
        hidden_dim: Hidden dimension for all memory networks.
        **kwargs: Additional arguments passed to DeepMomentumOptimizer.
    
    Example:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = NestedOptimizer(model.parameters())
        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(input), target)
        ...     loss.backward()
        ...     result = optimizer.step()
        ...     print(f"Fast stepped: {result['fast_stepped']}")
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        fast_lr: float = 0.01,
        medium_lr: float = 0.005,
        slow_lr: float = 0.001,
        fast_freq: int = 1,
        medium_freq: int = 10,
        slow_freq: int = 100,
        hidden_dim: int = 64,
        alpha: float = 0.9,
        memory_size: int = 10,
        meta_lr: float = 0.0001,
        **kwargs: Any,
    ):
        if fast_freq < 1:
            raise ValueError(f"fast_freq must be >= 1, got {fast_freq}")
        if medium_freq < fast_freq:
            raise ValueError(f"medium_freq ({medium_freq}) must be >= fast_freq ({fast_freq})")
        if slow_freq < medium_freq:
            raise ValueError(f"slow_freq ({slow_freq}) must be >= medium_freq ({medium_freq})")
        
        # Convert params to list to allow multiple iterations
        params_list = list(params)
        
        # Store configuration
        self.fast_freq = fast_freq
        self.medium_freq = medium_freq
        self.slow_freq = slow_freq
        self.hidden_dim = hidden_dim
        
        # Create three optimizers with shared parameters but different learning rates
        # Each optimizer maintains its own momentum state
        self.fast = DeepMomentumOptimizer(
            iter(params_list),  # Fresh iterator
            lr=fast_lr,
            alpha=alpha,
            memory_size=memory_size,
            hidden_dim=hidden_dim,
            meta_lr=meta_lr,
            **kwargs,
        )
        
        self.medium = DeepMomentumOptimizer(
            iter(params_list),  # Fresh iterator
            lr=medium_lr,
            alpha=alpha,
            memory_size=memory_size,
            hidden_dim=hidden_dim,
            meta_lr=meta_lr,
            **kwargs,
        )
        
        self.slow = DeepMomentumOptimizer(
            iter(params_list),  # Fresh iterator
            lr=slow_lr,
            alpha=alpha,
            memory_size=memory_size,
            hidden_dim=hidden_dim,
            meta_lr=meta_lr,
            **kwargs,
        )
        
        # Step counter
        self._step_count = 0
        
        # Track which optimizers stepped in the last call
        self._last_step_info: Dict[str, bool] = {
            'fast_stepped': False,
            'medium_stepped': False,
            'slow_stepped': False,
        }
        
        # Accumulated gradients for medium and slow optimizers
        # These accumulate between their update steps
        self._medium_grad_accum: Dict[int, torch.Tensor] = {}
        self._slow_grad_accum: Dict[int, torch.Tensor] = {}
        self._medium_accum_count = 0
        self._slow_accum_count = 0
    
    @property
    def step_count(self) -> int:
        """Current global step count."""
        return self._step_count
    
    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Return param_groups from the fast optimizer (they share params)."""
        return self.fast.param_groups
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients of all optimized parameters.
        
        Note: This only needs to be called once since all optimizers
        share the same parameters.
        
        Args:
            set_to_none: If True, set gradients to None instead of zero.
        """
        self.fast.zero_grad(set_to_none=set_to_none)
    
    def _accumulate_gradients(self) -> None:
        """Accumulate gradients for medium and slow optimizers."""
        for group in self.fast.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = id(p)
                
                # Accumulate for medium optimizer
                if param_id not in self._medium_grad_accum:
                    self._medium_grad_accum[param_id] = torch.zeros_like(p.grad)
                self._medium_grad_accum[param_id].add_(p.grad)
                
                # Accumulate for slow optimizer
                if param_id not in self._slow_grad_accum:
                    self._slow_grad_accum[param_id] = torch.zeros_like(p.grad)
                self._slow_grad_accum[param_id].add_(p.grad)
        
        self._medium_accum_count += 1
        self._slow_accum_count += 1
    
    def _apply_accumulated_gradients(
        self, 
        optimizer: DeepMomentumOptimizer,
        grad_accum: Dict[int, torch.Tensor],
        accum_count: int,
    ) -> None:
        """Apply accumulated gradients to an optimizer.
        
        Args:
            optimizer: The optimizer to apply gradients to.
            grad_accum: Dictionary of accumulated gradients.
            accum_count: Number of steps accumulated.
        """
        if accum_count == 0:
            return
        
        # Temporarily replace gradients with accumulated averages
        original_grads: Dict[int, Optional[torch.Tensor]] = {}
        
        for group in optimizer.param_groups:
            for p in group['params']:
                param_id = id(p)
                original_grads[param_id] = p.grad
                
                if param_id in grad_accum:
                    # Use average of accumulated gradients
                    p.grad = grad_accum[param_id] / accum_count
    
    def _clear_accumulated_gradients(
        self,
        grad_accum: Dict[int, torch.Tensor],
    ) -> int:
        """Clear accumulated gradients and return the count.
        
        Args:
            grad_accum: Dictionary of accumulated gradients to clear.
        
        Returns:
            The accumulation count before clearing.
        """
        for param_id in grad_accum:
            grad_accum[param_id].zero_()
        return 0
    
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Dict[str, Any]:
        """Perform optimization step at appropriate timescales.
        
        The fast optimizer always steps. Medium and slow optimizers
        step according to their frequencies.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Dictionary indicating which optimizers stepped and diagnostics:
            {
                'fast_stepped': bool,
                'medium_stepped': bool,
                'slow_stepped': bool,
                'step_count': int,
            }
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        # Accumulate gradients for medium and slow
        self._accumulate_gradients()
        
        # Track which optimizers step
        fast_stepped = False
        medium_stepped = False
        slow_stepped = False
        
        # Fast optimizer always steps (if freq is 1)
        if self._step_count % self.fast_freq == 0:
            self.fast.step()
            fast_stepped = True
        
        # Medium optimizer steps every medium_freq steps
        if self._step_count % self.medium_freq == 0:
            # Apply accumulated gradients
            self._apply_accumulated_gradients(
                self.medium,
                self._medium_grad_accum,
                self._medium_accum_count,
            )
            self.medium.step()
            medium_stepped = True
            # Clear accumulator
            self._medium_accum_count = self._clear_accumulated_gradients(
                self._medium_grad_accum
            )
        
        # Slow optimizer steps every slow_freq steps
        if self._step_count % self.slow_freq == 0:
            # Apply accumulated gradients
            self._apply_accumulated_gradients(
                self.slow,
                self._slow_grad_accum,
                self._slow_accum_count,
            )
            self.slow.step()
            slow_stepped = True
            # Clear accumulator
            self._slow_accum_count = self._clear_accumulated_gradients(
                self._slow_grad_accum
            )
        
        self._last_step_info = {
            'fast_stepped': fast_stepped,
            'medium_stepped': medium_stepped,
            'slow_stepped': slow_stepped,
            'step_count': self._step_count,
        }
        
        return self._last_step_info
    
    def meta_step(self, loss: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Train memory networks for all active optimizers.
        
        Only trains the memory networks for optimizers that stepped
        in the most recent step() call.
        
        Args:
            loss: Current loss value.
        
        Returns:
            Nested dict with meta-learning stats per optimizer:
            {
                'fast': {'meta_loss': float, ...},
                'medium': {'meta_loss': float, ...},
                'slow': {'meta_loss': float, ...},
            }
        """
        results: Dict[str, Dict[str, float]] = {}
        
        if self._last_step_info.get('fast_stepped', False):
            results['fast'] = self.fast.meta_step(loss)
        else:
            results['fast'] = {'meta_loss': 0.0, 'loss_improvement': 0.0, 'weight_change': 0.0}
        
        if self._last_step_info.get('medium_stepped', False):
            results['medium'] = self.medium.meta_step(loss)
        else:
            results['medium'] = {'meta_loss': 0.0, 'loss_improvement': 0.0, 'weight_change': 0.0}
        
        if self._last_step_info.get('slow_stepped', False):
            results['slow'] = self.slow.meta_step(loss)
        else:
            results['slow'] = {'meta_loss': 0.0, 'loss_improvement': 0.0, 'weight_change': 0.0}
        
        return results
    
    def get_memory_states(self) -> Dict[str, torch.Tensor]:
        """Get knowledge states from all three optimizers.
        
        Returns:
            Dictionary with knowledge states:
            {
                'fast': Tensor(hidden_dim,),
                'medium': Tensor(hidden_dim,),
                'slow': Tensor(hidden_dim,),
            }
        """
        return {
            'fast': self.fast.get_knowledge_state(),
            'medium': self.medium.get_knowledge_state(),
            'slow': self.slow.get_knowledge_state(),
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from all optimizers.
        
        Returns:
            Dictionary with diagnostics from each optimizer.
        """
        return {
            'step_count': self._step_count,
            'fast_freq': self.fast_freq,
            'medium_freq': self.medium_freq,
            'slow_freq': self.slow_freq,
            'fast': self.fast.get_diagnostics(),
            'medium': self.medium.get_diagnostics(),
            'slow': self.slow.get_diagnostics(),
            'medium_accum_count': self._medium_accum_count,
            'slow_accum_count': self._slow_accum_count,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict.
        
        Returns:
            Dictionary containing optimizer state.
        """
        return {
            'step_count': self._step_count,
            'fast': self.fast.state_dict(),
            'medium': self.medium.state_dict(),
            'slow': self.slow.state_dict(),
            'medium_accum_count': self._medium_accum_count,
            'slow_accum_count': self._slow_accum_count,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state.
        
        Args:
            state_dict: Dictionary containing optimizer state.
        """
        self._step_count = state_dict['step_count']
        self.fast.load_state_dict(state_dict['fast'])
        self.medium.load_state_dict(state_dict['medium'])
        self.slow.load_state_dict(state_dict['slow'])
        self._medium_accum_count = state_dict.get('medium_accum_count', 0)
        self._slow_accum_count = state_dict.get('slow_accum_count', 0)


def _test_nested_optimizer():
    """Basic test to verify the nested optimizer works."""
    import torch.nn.functional as F
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    
    # Create nested optimizer
    optimizer = NestedOptimizer(
        model.parameters(),
        fast_lr=0.01,
        medium_lr=0.005,
        slow_lr=0.001,
        fast_freq=1,
        medium_freq=5,
        slow_freq=20,
        hidden_dim=32,
    )
    
    # Dummy data
    torch.manual_seed(42)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    # Training loop
    print("Testing NestedOptimizer...")
    initial_loss = None
    
    fast_steps = 0
    medium_steps = 0
    slow_steps = 0
    
    for step in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        result = optimizer.step()
        
        if result['fast_stepped']:
            fast_steps += 1
        if result['medium_stepped']:
            medium_steps += 1
        if result['slow_stepped']:
            slow_steps += 1
        
        if step % 20 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}, "
                  f"fast={result['fast_stepped']}, "
                  f"medium={result['medium_stepped']}, "
                  f"slow={result['slow_stepped']}")
    
    final_loss = F.cross_entropy(model(x), y).item()
    
    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {initial_loss - final_loss:.4f}")
    print(f"\n  Fast steps: {fast_steps} (expected: 100)")
    print(f"  Medium steps: {medium_steps} (expected: 20)")
    print(f"  Slow steps: {slow_steps} (expected: 5)")
    
    # Verify step counts
    assert fast_steps == 100, f"Expected 100 fast steps, got {fast_steps}"
    assert medium_steps == 20, f"Expected 20 medium steps, got {medium_steps}"
    assert slow_steps == 5, f"Expected 5 slow steps, got {slow_steps}"
    
    # Verify loss decreased
    assert final_loss < initial_loss, "Optimizer should reduce loss!"
    
    # Check memory states
    states = optimizer.get_memory_states()
    print(f"\n  Memory state shapes:")
    for name, state in states.items():
        print(f"    {name}: {state.shape}")
    
    print("\n✓ Nested optimizer test passed!")


if __name__ == "__main__":
    _test_nested_optimizer()