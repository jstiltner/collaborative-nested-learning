"""Tests for DeepMomentumOptimizer.

Tests cover:
- Basic initialization and parameter validation
- Gradient history storage
- Memory network functionality
- Optimization step behavior
- Meta-learning step
- Knowledge transfer methods
- Comparison with SGD baseline
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.optimizers.deep_momentum import DeepMomentumOptimizer, MemoryNetwork


class TestMemoryNetwork:
    """Tests for the MemoryNetwork class."""

    def test_initialization(self):
        """Test that MemoryNetwork initializes correctly."""
        net = MemoryNetwork(input_dim=30, hidden_dim=64, output_dim=1)

        # Check architecture
        assert len(net.net) == 5  # 3 linear + 2 relu
        assert isinstance(net.net[0], nn.Linear)
        assert isinstance(net.net[1], nn.ReLU)

    def test_output_shape(self):
        """Test that output shape is correct."""
        net = MemoryNetwork(input_dim=30, hidden_dim=64, output_dim=1)
        x = torch.randn(30)
        output = net(x)

        assert output.shape == (1,)

    def test_output_positive(self):
        """Test that output is always positive (due to softplus)."""
        net = MemoryNetwork(input_dim=30, hidden_dim=64, output_dim=1)

        # Test with various inputs
        for _ in range(10):
            x = torch.randn(30) * 10  # Large random values
            output = net(x)
            assert output.item() > 0, "Output should always be positive"

    def test_initial_output_near_one(self):
        """Test that initial output is close to 1.0 (identity preconditioner)."""
        net = MemoryNetwork(input_dim=30, hidden_dim=64, output_dim=1)
        x = torch.zeros(30)  # Zero input
        output = net(x)

        # Should be close to 1.0 due to initialization
        assert (
            0.5 < output.item() < 2.0
        ), f"Initial output should be near 1.0, got {output.item()}"


class TestDeepMomentumOptimizer:
    """Tests for the DeepMomentumOptimizer class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 2)

    @pytest.fixture
    def optimizer(self, simple_model):
        """Create an optimizer instance."""
        return DeepMomentumOptimizer(
            simple_model.parameters(),
            lr=0.01,
            alpha=0.9,
            memory_size=5,
            hidden_dim=32,
        )

    def test_initialization(self, simple_model):
        """Test optimizer initialization."""
        opt = DeepMomentumOptimizer(simple_model.parameters(), lr=0.01)

        assert opt.defaults["lr"] == 0.01
        assert opt.defaults["alpha"] == 0.9
        assert opt.defaults["memory_size"] == 10

    def test_invalid_lr(self, simple_model):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            DeepMomentumOptimizer(simple_model.parameters(), lr=-0.01)

    def test_invalid_alpha(self, simple_model):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="Invalid momentum decay"):
            DeepMomentumOptimizer(simple_model.parameters(), alpha=1.5)

    def test_invalid_memory_size(self, simple_model):
        """Test that invalid memory_size raises error."""
        with pytest.raises(ValueError, match="Invalid memory_size"):
            DeepMomentumOptimizer(simple_model.parameters(), memory_size=0)

    def test_state_initialization(self, optimizer, simple_model):
        """Test that state is properly initialized for all parameters."""
        for p in simple_model.parameters():
            assert p in optimizer.state
            state = optimizer.state[p]
            assert "step" in state
            assert "momentum" in state
            assert "grad_history" in state
            assert "prev_param" in state
            assert state["step"] == 0
            assert state["momentum"].shape == p.shape

    def test_zero_grad(self, optimizer, simple_model):
        """Test zero_grad clears gradients."""
        # Create some gradients
        x = torch.randn(4, 10)
        y = simple_model(x).sum()
        y.backward()

        # Verify gradients exist
        for p in simple_model.parameters():
            assert p.grad is not None

        # Zero gradients
        optimizer.zero_grad()

        # Verify gradients are cleared
        for p in simple_model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

    def test_step_updates_params(self, optimizer, simple_model):
        """Test that step() updates parameters."""
        # Store initial params
        initial_params = [p.clone() for p in simple_model.parameters()]

        # Create gradients
        x = torch.randn(4, 10)
        y = simple_model(x).sum()
        y.backward()

        # Take step
        optimizer.step()

        # Verify params changed
        for p, initial in zip(simple_model.parameters(), initial_params):
            assert not torch.allclose(p, initial), "Parameters should change after step"

    def test_step_count_increments(self, optimizer, simple_model):
        """Test that step count increments."""
        assert optimizer._step_count == 0

        x = torch.randn(4, 10)
        y = simple_model(x).sum()
        y.backward()
        optimizer.step()

        assert optimizer._step_count == 1

    def test_gradient_history_updates(self, optimizer, simple_model):
        """Test that gradient history is updated after step."""
        x = torch.randn(4, 10)
        y = simple_model(x).sum()
        y.backward()
        optimizer.step()

        for p in simple_model.parameters():
            state = optimizer.state[p]
            assert len(state["grad_history"]) == 1

    def test_gradient_history_rolling_window(self, optimizer, simple_model):
        """Test that gradient history respects memory_size."""
        memory_size = optimizer.memory_size

        # Take more steps than memory_size
        for _ in range(memory_size + 5):
            optimizer.zero_grad()
            x = torch.randn(4, 10)
            y = simple_model(x).sum()
            y.backward()
            optimizer.step()

        # Check history length is capped
        for p in simple_model.parameters():
            state = optimizer.state[p]
            assert len(state["grad_history"]) == memory_size

    def test_memory_net_lazy_init(self, optimizer, simple_model):
        """Test that memory network is lazily initialized."""
        assert optimizer._memory_net is None

        # Take a step to trigger initialization
        x = torch.randn(4, 10)
        y = simple_model(x).sum()
        y.backward()
        optimizer.step()

        assert optimizer._memory_net is not None

    def test_get_knowledge_state(self, optimizer, simple_model):
        """Test get_knowledge_state returns correct shape."""
        # Before initialization
        state = optimizer.get_knowledge_state()
        assert state.shape == (optimizer.hidden_dim,)

        # After initialization
        x = torch.randn(4, 10)
        y = simple_model(x).sum()
        y.backward()
        optimizer.step()

        state = optimizer.get_knowledge_state()
        assert state.shape == (optimizer.hidden_dim,)

    def test_inject_knowledge(self, optimizer, simple_model):
        """Test inject_knowledge modifies memory network."""
        # Initialize memory network
        x = torch.randn(4, 10)
        y = simple_model(x).sum()
        y.backward()
        optimizer.step()

        # Get initial state
        initial_bias = optimizer._memory_net.net[0].bias.clone()

        # Inject knowledge
        knowledge = torch.randn(optimizer.hidden_dim)
        optimizer.inject_knowledge(knowledge, gate=0.5)

        # Verify bias changed
        new_bias = optimizer._memory_net.net[0].bias
        assert not torch.allclose(initial_bias, new_bias)

    def test_inject_knowledge_gate_zero(self, optimizer, simple_model):
        """Test that gate=0 doesn't modify anything."""
        # Initialize memory network
        x = torch.randn(4, 10)
        y = simple_model(x).sum()
        y.backward()
        optimizer.step()

        # Get initial state
        initial_bias = optimizer._memory_net.net[0].bias.clone()

        # Inject with gate=0
        knowledge = torch.randn(optimizer.hidden_dim)
        optimizer.inject_knowledge(knowledge, gate=0.0)

        # Verify bias unchanged
        new_bias = optimizer._memory_net.net[0].bias
        assert torch.allclose(initial_bias, new_bias)

    def test_diagnostics(self, optimizer, simple_model):
        """Test get_diagnostics returns expected keys."""
        diag = optimizer.get_diagnostics()

        assert "step_count" in diag
        assert "total_params" in diag
        assert "avg_momentum_norm" in diag
        assert "total_knowledge_transfers" in diag
        assert "memory_net_initialized" in diag


class TestOptimizationBehavior:
    """Tests for actual optimization behavior."""

    def test_reduces_loss(self):
        """Test that optimizer reduces loss on a simple problem."""
        # Simple model
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        optimizer = DeepMomentumOptimizer(model.parameters(), lr=0.01)

        # Fixed data
        torch.manual_seed(42)
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        # Initial loss
        initial_loss = F.cross_entropy(model(x), y).item()

        # Train
        for _ in range(50):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = F.cross_entropy(model(x), y).item()

        assert final_loss < initial_loss, "Optimizer should reduce loss"

    def test_comparable_to_sgd(self):
        """Test that optimizer performs comparably to SGD."""
        torch.manual_seed(42)

        # Create two identical models
        model_deep = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        model_sgd = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        # Copy weights
        model_sgd.load_state_dict(model_deep.state_dict())

        opt_deep = DeepMomentumOptimizer(model_deep.parameters(), lr=0.01)
        opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)

        # Fixed data
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        # Train both
        for _ in range(100):
            # Deep momentum
            opt_deep.zero_grad()
            loss_deep = F.cross_entropy(model_deep(x), y)
            loss_deep.backward()
            opt_deep.step()

            # SGD
            opt_sgd.zero_grad()
            loss_sgd = F.cross_entropy(model_sgd(x), y)
            loss_sgd.backward()
            opt_sgd.step()

        final_deep = F.cross_entropy(model_deep(x), y).item()
        final_sgd = F.cross_entropy(model_sgd(x), y).item()

        # Deep momentum should be within 2x of SGD performance
        # (it may be better or slightly worse depending on the problem)
        assert (
            final_deep < 2 * final_sgd
        ), f"Deep momentum ({final_deep:.4f}) should be comparable to SGD ({final_sgd:.4f})"

    def test_different_batch_sizes(self):
        """Test optimizer works with different batch sizes."""
        model = nn.Linear(10, 2)
        optimizer = DeepMomentumOptimizer(model.parameters(), lr=0.01)

        for batch_size in [1, 4, 16, 64]:
            optimizer.zero_grad()
            x = torch.randn(batch_size, 10)
            y = model(x).sum()
            y.backward()
            optimizer.step()  # Should not raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test optimizer works on CUDA."""
        model = nn.Linear(10, 2).cuda()
        optimizer = DeepMomentumOptimizer(model.parameters(), lr=0.01)

        x = torch.randn(4, 10).cuda()
        y = model(x).sum()
        y.backward()
        optimizer.step()

        # Memory network should be on CUDA
        assert optimizer._memory_net is not None
        assert next(optimizer._memory_net.parameters()).is_cuda


class TestMetaLearning:
    """Tests for meta-learning functionality."""

    def test_meta_step_returns_diagnostics(self):
        """Test that meta_step returns expected diagnostics."""
        model = nn.Linear(10, 2)
        optimizer = DeepMomentumOptimizer(model.parameters(), lr=0.01)

        # Take a step first
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Meta step
        optimizer.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        result = optimizer.meta_step(loss)

        assert "meta_loss" in result
        assert "loss_improvement" in result
        assert "weight_change" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
