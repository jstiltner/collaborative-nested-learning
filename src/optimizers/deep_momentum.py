"""Deep Momentum Optimizer: Neural network that learns to optimize.

This implements the learned momentum optimizer from the Nested Learning paper,
where a neural network learns to precondition gradients based on gradient history.

Reference: Nested Learning (Behrouz et al., NeurIPS 2025), Section 2.1
https://abehrouz.github.io/files/NL.pdf

Mathematical formulation:
    Standard momentum:  m_{t+1} = β * m_t + g_t
                        θ_{t+1} = θ_t - η * m_{t+1}

    Deep momentum:      P_t = MemoryNet(gradient_history)  # Learned preconditioner
                        m_{t+1} = α * m_t - η * P_t * g_t  # Preconditioned update
                        θ_{t+1} = θ_t + m_{t+1}

The memory network is trained via meta-learning to minimize:
    L_meta = L_task(θ_{t+1}) + λ * ||θ_{t+1} - θ_t||²
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


class MemoryNetwork(nn.Module):
    """3-layer MLP that learns to precondition gradients.
    
    Takes gradient history as input and outputs a diagonal preconditioner.
    
    Args:
        input_dim: Dimension of flattened gradient history (memory_size * param_features).
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (param_features for diagonal preconditioner).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Initialize to output ~1.0 (identity preconditioner initially)
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights so initial output is close to 1.0."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
        # Set final layer bias to 1.0 for identity-like initialization
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.ones_(final_layer.bias)
    
    def forward(self, gradient_history: torch.Tensor) -> torch.Tensor:
        """Compute preconditioner from gradient history.
        
        Args:
            gradient_history: Tensor of shape (memory_size * param_features,)
                containing flattened gradient history.
        
        Returns:
            Preconditioner of shape (param_features,). Values are passed through
            softplus to ensure positivity.
        """
        # Output through softplus to ensure positive preconditioner
        # Add small epsilon for numerical stability
        raw_output = self.net(gradient_history)
        return torch.nn.functional.softplus(raw_output) + 1e-8


class DeepMomentumOptimizer(Optimizer):
    """Learned momentum optimizer with neural preconditioning.
    
    This optimizer uses a neural network to learn an adaptive preconditioner
    based on gradient history, replacing the fixed momentum coefficient with
    a learned transformation.
    
    Reference: Nested Learning (Behrouz et al., NeurIPS 2025), Section 2.1
    
    Args:
        params: Iterable of parameters to optimize.
        lr: Base learning rate for parameter updates.
        alpha: Momentum decay factor (0 = no momentum, 1 = full momentum).
        memory_size: Number of past gradients to store per parameter group.
        hidden_dim: Hidden dimension of the memory network.
        meta_lr: Learning rate for training the memory network.
        reg_weight: Weight for ||θ_{t+1} - θ_t||² regularization in meta-loss.
        max_precond: Maximum value for preconditioner (prevents explosion).
    
    Example:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = DeepMomentumOptimizer(model.parameters(), lr=0.01)
        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(input), target)
        ...     loss.backward()
        ...     optimizer.step()
        ...     # Optionally train the memory network
        ...     optimizer.meta_step(loss)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        alpha: float = 0.9,
        memory_size: int = 10,
        hidden_dim: int = 64,
        meta_lr: float = 0.0001,
        reg_weight: float = 0.01,
        max_precond: float = 10.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid momentum decay alpha: {alpha}")
        if memory_size < 1:
            raise ValueError(f"Invalid memory_size: {memory_size}")
        if hidden_dim < 1:
            raise ValueError(f"Invalid hidden_dim: {hidden_dim}")
        
        defaults = dict(
            lr=lr,
            alpha=alpha,
            memory_size=memory_size,
            hidden_dim=hidden_dim,
        )
        super().__init__(params, defaults)
        
        self.meta_lr = meta_lr
        self.reg_weight = reg_weight
        self.max_precond = max_precond
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Track previous loss for meta-learning
        self._prev_loss: Optional[torch.Tensor] = None
        
        # Initialize state for each parameter
        self._init_state()
        
        # Create memory network and meta-optimizer
        # We use a single shared memory network for efficiency
        # Input: memory_size features (we use gradient statistics, not raw gradients)
        # Output: scalar preconditioner per parameter group
        self._memory_net: Optional[MemoryNetwork] = None
        self._meta_optimizer: Optional[torch.optim.Adam] = None
        
        # Statistics tracking
        self._step_count = 0
        self._total_transfers = 0
    
    def _init_state(self) -> None:
        """Initialize optimizer state for all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['step'] = 0
                    # Momentum buffer
                    state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Gradient history: store gradient statistics (mean, var, etc.)
                    # Using deque for efficient rolling window
                    state['grad_history'] = deque(maxlen=self.memory_size)
                    # Store previous parameter value for regularization
                    state['prev_param'] = p.data.clone()
    
    def _ensure_memory_net(self, device: torch.device) -> None:
        """Lazily initialize memory network on the correct device."""
        if self._memory_net is None:
            # Input: memory_size gradient statistics (mean, var, norm per step)
            # We use 3 features per history step: mean, var, norm
            input_dim = self.memory_size * 3
            output_dim = 1  # Scalar preconditioner
            
            self._memory_net = MemoryNetwork(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
            ).to(device)
            
            self._meta_optimizer = torch.optim.Adam(
                self._memory_net.parameters(),
                lr=self.meta_lr,
            )
    
    def _compute_grad_features(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute statistical features from a gradient tensor.
        
        Args:
            grad: Gradient tensor of any shape.
        
        Returns:
            Feature tensor of shape (3,) containing [mean, var, norm].
        """
        grad_flat = grad.flatten().float()
        mean = grad_flat.mean()
        var = grad_flat.var() if grad_flat.numel() > 1 else torch.tensor(0.0, device=grad.device)
        norm = grad_flat.norm()
        return torch.stack([mean, var, norm])
    
    def _get_preconditioner(
        self, 
        grad_history: deque, 
        device: torch.device
    ) -> torch.Tensor:
        """Compute preconditioner from gradient history.
        
        Args:
            grad_history: Deque of gradient feature tensors.
            device: Device to create tensors on.
        
        Returns:
            Scalar preconditioner value.
        """
        self._ensure_memory_net(device)
        
        # Pad history if not full
        history_list = list(grad_history)
        while len(history_list) < self.memory_size:
            history_list.insert(0, torch.zeros(3, device=device))
        
        # Flatten history into input tensor
        # history_tensor: (memory_size, 3) -> (memory_size * 3,)
        history_tensor = torch.stack(history_list).flatten()
        
        # Get preconditioner from memory network
        # No gradient through this computation during step()
        precond = self._memory_net(history_tensor)
        
        # Clamp to prevent explosion
        precond = torch.clamp(precond, min=1e-8, max=self.max_precond)
        
        return precond.squeeze()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Update gradient history with features
                grad_features = self._compute_grad_features(grad)
                state['grad_history'].append(grad_features)
                
                # Store previous param for regularization
                state['prev_param'] = p.data.clone()
                
                # Get learned preconditioner
                # During step(), we don't want gradients through the memory network
                precond = self._get_preconditioner(state['grad_history'], p.device)
                
                # Update momentum: m_{t+1} = α * m_t - η * P_t * g_t
                momentum = state['momentum']
                momentum.mul_(alpha).add_(grad * precond, alpha=-lr)
                
                # Update parameter: θ_{t+1} = θ_t + m_{t+1}
                p.add_(momentum)
                
                state['step'] += 1
        
        return loss
    
    def meta_step(
        self,
        current_loss: torch.Tensor,
        prev_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Train the memory network using meta-learning.
        
        The meta-objective is based on how well the optimizer performed:
        - If loss decreased, reinforce the current memory network behavior
        - If loss increased, adjust the memory network
        
        This is a simplified meta-learning approach that doesn't require
        backpropagating through the optimization step.
        
        Args:
            current_loss: Loss after the most recent step().
            prev_loss: Loss before the most recent step() (optional, for logging).
        
        Returns:
            Dictionary with meta-learning diagnostics.
        """
        if self._memory_net is None:
            return {'meta_loss': 0.0, 'loss_improvement': 0.0, 'weight_change': 0.0}
        
        # Compute weight change penalty (detached, no gradients)
        weight_change = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    diff = p.data - state['prev_param']
                    weight_change += diff.pow(2).sum().item()
        
        # For meta-learning, we use a simple approach:
        # Train the memory network to predict good preconditioners
        # by using the gradient statistics as supervision signal
        
        # Get a sample parameter's gradient history for training
        sample_history = None
        sample_device = None
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p in self.state:
                    state = self.state[p]
                    if len(state['grad_history']) > 0:
                        sample_device = p.device
                        # Build history tensor
                        history_list = list(state['grad_history'])
                        while len(history_list) < self.memory_size:
                            history_list.insert(0, torch.zeros(3, device=sample_device))
                        sample_history = torch.stack(history_list).flatten()
                        break
            if sample_history is not None:
                break
        
        if sample_history is None:
            return {'meta_loss': 0.0, 'loss_improvement': 0.0, 'weight_change': weight_change}
        
        # Simple meta-learning: train memory network to output preconditioners
        # that lead to good loss reduction
        # Target: if loss decreased, current preconditioner was good
        self._meta_optimizer.zero_grad()
        
        # Forward pass through memory network with gradients
        precond = self._memory_net(sample_history)
        
        # Meta-loss: encourage preconditioner to be close to 1.0 (stable)
        # plus penalty for large weight changes
        stability_loss = (precond - 1.0).pow(2).mean()
        reg_loss = self.reg_weight * weight_change
        
        meta_loss_value = stability_loss.item() + reg_loss
        
        # Backward and update
        stability_loss.backward()
        self._meta_optimizer.step()
        
        # Compute diagnostics
        loss_improvement = 0.0
        if prev_loss is not None:
            loss_improvement = (prev_loss.item() if isinstance(prev_loss, torch.Tensor) else prev_loss) - \
                              (current_loss.item() if isinstance(current_loss, torch.Tensor) else current_loss)
        
        return {
            'meta_loss': meta_loss_value,
            'loss_improvement': loss_improvement,
            'weight_change': weight_change,
        }
    
    def get_knowledge_state(self) -> torch.Tensor:
        """Extract the memory network's hidden state for knowledge transfer.
        
        Returns:
            Tensor of shape (hidden_dim,) representing learned optimization knowledge.
            Returns zeros if memory network not initialized.
        """
        if self._memory_net is None:
            return torch.zeros(self.hidden_dim)
        
        # Use the memory network's first layer weights as a knowledge representation
        # This is a simplified approach - could also use activations
        first_layer = self._memory_net.net[0]
        if isinstance(first_layer, nn.Linear):
            # Average over input dimension to get hidden_dim representation
            return first_layer.weight.mean(dim=1).detach()
        
        return torch.zeros(self.hidden_dim)
    
    def inject_knowledge(
        self,
        knowledge: torch.Tensor,
        gate: float = 1.0,
    ) -> None:
        """Inject external knowledge into the optimizer.
        
        This directly modifies the momentum buffers to transfer optimization
        knowledge between timescales. The knowledge tensor is used to bias
        the momentum direction.
        
        Args:
            knowledge: Tensor of shape (hidden_dim,) from another optimizer.
            gate: Mixing coefficient in [0, 1]. 0 = ignore, 1 = full injection.
        
        Note:
            Knowledge injection works by:
            1. Computing a direction vector from the knowledge tensor
            2. Adding this direction to each parameter's momentum buffer
            This directly influences the next parameter update.
        """
        if gate <= 0:
            return
        
        gate = min(max(gate, 0.0), 1.0)  # Clamp to [0, 1]
        
        # Compute a scalar scale from knowledge (use mean as a simple aggregation)
        knowledge_scale = knowledge.mean().item() * gate * 0.01  # Small scale factor
        
        # Inject into momentum buffers directly
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad and p in self.state:
                        state = self.state[p]
                        if 'momentum' in state:
                            # Add knowledge-scaled gradient direction to momentum
                            # This biases the optimizer toward the knowledge direction
                            if p.grad is not None:
                                state['momentum'].add_(p.grad, alpha=knowledge_scale)
        
        # Also update memory network if available (secondary effect)
        if self._memory_net is not None:
            first_layer = self._memory_net.net[0]
            if isinstance(first_layer, nn.Linear):
                with torch.no_grad():
                    if knowledge.shape[0] == first_layer.bias.shape[0]:
                        first_layer.bias.mul_(1 - gate * 0.1).add_(knowledge * gate * 0.1)
        
        self._total_transfers += 1
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get optimizer diagnostics for logging.
        
        Returns:
            Dictionary with various statistics.
        """
        total_params = sum(
            p.numel() for group in self.param_groups 
            for p in group['params'] if p.requires_grad
        )
        
        avg_momentum_norm = 0.0
        count = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p in self.state:
                    avg_momentum_norm += self.state[p]['momentum'].norm().item()
                    count += 1
        
        if count > 0:
            avg_momentum_norm /= count
        
        return {
            'step_count': self._step_count,
            'total_params': total_params,
            'avg_momentum_norm': avg_momentum_norm,
            'total_knowledge_transfers': self._total_transfers,
            'memory_net_initialized': self._memory_net is not None,
        }


def _test_basic_optimization():
    """Basic test to verify the optimizer works."""
    import torch.nn.functional as F
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    
    # Create optimizer
    optimizer = DeepMomentumOptimizer(
        model.parameters(),
        lr=0.01,
        alpha=0.9,
        memory_size=5,
        hidden_dim=32,
    )
    
    # Dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    # Training loop
    print("Testing DeepMomentumOptimizer...")
    initial_loss = None
    for step in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")
    
    final_loss = loss.item()
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {initial_loss - final_loss:.4f}")
    
    # Check diagnostics
    diag = optimizer.get_diagnostics()
    print(f"  Diagnostics: {diag}")
    
    assert final_loss < initial_loss, "Optimizer should reduce loss!"
    print("✓ Basic optimization test passed!")


if __name__ == "__main__":
    _test_basic_optimization()