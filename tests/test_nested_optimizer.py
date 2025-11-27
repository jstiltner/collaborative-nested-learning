"""Tests for NestedOptimizer.

Tests cover:
- Initialization and parameter validation
- Multi-timescale step behavior
- Gradient accumulation
- Memory state extraction
- Meta-learning across timescales
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.optimizers.nested_optimizer import NestedOptimizer


class TestNestedOptimizerInit:
    """Tests for NestedOptimizer initialization."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 2)

    def test_basic_initialization(self, simple_model):
        """Test basic initialization with default parameters."""
        opt = NestedOptimizer(simple_model.parameters())

        assert opt.fast_freq == 1
        assert opt.medium_freq == 10
        assert opt.slow_freq == 100
        assert opt.step_count == 0

    def test_custom_frequencies(self, simple_model):
        """Test initialization with custom frequencies."""
        opt = NestedOptimizer(
            simple_model.parameters(),
            fast_freq=1,
            medium_freq=5,
            slow_freq=25,
        )

        assert opt.fast_freq == 1
        assert opt.medium_freq == 5
        assert opt.slow_freq == 25

    def test_custom_learning_rates(self, simple_model):
        """Test initialization with custom learning rates."""
        opt = NestedOptimizer(
            simple_model.parameters(),
            fast_lr=0.1,
            medium_lr=0.05,
            slow_lr=0.01,
        )

        assert opt.fast.defaults["lr"] == 0.1
        assert opt.medium.defaults["lr"] == 0.05
        assert opt.slow.defaults["lr"] == 0.01

    def test_invalid_fast_freq(self, simple_model):
        """Test that invalid fast_freq raises error."""
        with pytest.raises(ValueError, match="fast_freq must be >= 1"):
            NestedOptimizer(simple_model.parameters(), fast_freq=0)

    def test_invalid_medium_freq(self, simple_model):
        """Test that medium_freq < fast_freq raises error."""
        with pytest.raises(ValueError, match="medium_freq.*must be >= fast_freq"):
            NestedOptimizer(
                simple_model.parameters(),
                fast_freq=10,
                medium_freq=5,
            )

    def test_invalid_slow_freq(self, simple_model):
        """Test that slow_freq < medium_freq raises error."""
        with pytest.raises(ValueError, match="slow_freq.*must be >= medium_freq"):
            NestedOptimizer(
                simple_model.parameters(),
                medium_freq=20,
                slow_freq=10,
            )

    def test_three_optimizers_created(self, simple_model):
        """Test that three separate optimizers are created."""
        opt = NestedOptimizer(simple_model.parameters())

        assert opt.fast is not None
        assert opt.medium is not None
        assert opt.slow is not None
        assert opt.fast is not opt.medium
        assert opt.medium is not opt.slow


class TestNestedOptimizerStep:
    """Tests for step behavior."""

    @pytest.fixture
    def model_and_optimizer(self):
        """Create model and optimizer for testing."""
        model = nn.Linear(10, 2)
        opt = NestedOptimizer(
            model.parameters(),
            fast_freq=1,
            medium_freq=5,
            slow_freq=10,
        )
        return model, opt

    def test_step_count_increments(self, model_and_optimizer):
        """Test that step count increments."""
        model, opt = model_and_optimizer

        assert opt.step_count == 0

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        assert opt.step_count == 1

    def test_fast_always_steps(self, model_and_optimizer):
        """Test that fast optimizer steps every time."""
        model, opt = model_and_optimizer

        for i in range(10):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            result = opt.step()

            assert result["fast_stepped"] == True, f"Fast should step at step {i+1}"

    def test_medium_steps_at_frequency(self, model_and_optimizer):
        """Test that medium optimizer steps at correct frequency."""
        model, opt = model_and_optimizer

        medium_steps = []
        for i in range(20):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            result = opt.step()

            if result["medium_stepped"]:
                medium_steps.append(i + 1)

        # Should step at 5, 10, 15, 20
        assert medium_steps == [5, 10, 15, 20]

    def test_slow_steps_at_frequency(self, model_and_optimizer):
        """Test that slow optimizer steps at correct frequency."""
        model, opt = model_and_optimizer

        slow_steps = []
        for i in range(30):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            result = opt.step()

            if result["slow_stepped"]:
                slow_steps.append(i + 1)

        # Should step at 10, 20, 30
        assert slow_steps == [10, 20, 30]

    def test_step_returns_correct_info(self, model_and_optimizer):
        """Test that step returns correct information."""
        model, opt = model_and_optimizer

        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        result = opt.step()

        assert "fast_stepped" in result
        assert "medium_stepped" in result
        assert "slow_stepped" in result
        assert "step_count" in result
        assert result["step_count"] == 1

    def test_params_updated_after_step(self, model_and_optimizer):
        """Test that parameters are updated after step."""
        model, opt = model_and_optimizer

        initial_params = [p.clone() for p in model.parameters()]

        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p, initial in zip(model.parameters(), initial_params):
            assert not torch.allclose(p, initial), "Parameters should change"


class TestNestedOptimizerGradientAccumulation:
    """Tests for gradient accumulation behavior."""

    def test_gradient_accumulation_for_medium(self):
        """Test that gradients accumulate for medium optimizer."""
        model = nn.Linear(10, 2)
        opt = NestedOptimizer(
            model.parameters(),
            fast_freq=1,
            medium_freq=3,
            slow_freq=6,
        )

        # Take 2 steps (medium hasn't stepped yet)
        for _ in range(2):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            opt.step()

        # Check accumulation count
        assert opt._medium_accum_count == 2

        # Take one more step (medium should step and reset)
        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        result = opt.step()

        assert result["medium_stepped"] == True
        assert opt._medium_accum_count == 0


class TestNestedOptimizerMemoryStates:
    """Tests for memory state extraction."""

    def test_get_memory_states_shape(self):
        """Test that memory states have correct shape."""
        model = nn.Linear(10, 2)
        opt = NestedOptimizer(
            model.parameters(),
            hidden_dim=64,
        )

        # Take a step to initialize memory networks
        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        states = opt.get_memory_states()

        assert "fast" in states
        assert "medium" in states
        assert "slow" in states
        assert states["fast"].shape == (64,)
        assert states["medium"].shape == (64,)
        assert states["slow"].shape == (64,)

    def test_memory_states_before_step(self):
        """Test memory states before any step (should be zeros)."""
        model = nn.Linear(10, 2)
        opt = NestedOptimizer(
            model.parameters(),
            hidden_dim=32,
        )

        states = opt.get_memory_states()

        # Before initialization, should return zeros
        assert torch.allclose(states["fast"], torch.zeros(32))


class TestNestedOptimizerMetaLearning:
    """Tests for meta-learning functionality."""

    def test_meta_step_returns_results(self):
        """Test that meta_step returns results for all optimizers."""
        model = nn.Linear(10, 2)
        opt = NestedOptimizer(
            model.parameters(),
            fast_freq=1,
            medium_freq=2,
            slow_freq=4,
        )

        # Take a step
        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # Meta step
        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        results = opt.meta_step(loss)

        assert "fast" in results
        assert "medium" in results
        assert "slow" in results

    def test_meta_step_only_for_active_optimizers(self):
        """Test that meta_step only trains active optimizers."""
        model = nn.Linear(10, 2)
        opt = NestedOptimizer(
            model.parameters(),
            fast_freq=1,
            medium_freq=10,
            slow_freq=100,
        )

        # Take one step (only fast should be active)
        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        results = opt.meta_step(loss)

        # Fast should have non-zero meta_loss (it stepped)
        # Medium and slow should have zero (they didn't step)
        assert results["medium"]["meta_loss"] == 0.0
        assert results["slow"]["meta_loss"] == 0.0


class TestNestedOptimizerDiagnostics:
    """Tests for diagnostics functionality."""

    def test_get_diagnostics(self):
        """Test that diagnostics returns expected keys."""
        model = nn.Linear(10, 2)
        opt = NestedOptimizer(model.parameters())

        diag = opt.get_diagnostics()

        assert "step_count" in diag
        assert "fast_freq" in diag
        assert "medium_freq" in diag
        assert "slow_freq" in diag
        assert "fast" in diag
        assert "medium" in diag
        assert "slow" in diag


class TestNestedOptimizerOptimization:
    """Tests for actual optimization behavior."""

    def test_reduces_loss(self):
        """Test that nested optimizer reduces loss."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        opt = NestedOptimizer(
            model.parameters(),
            fast_lr=0.01,
            medium_lr=0.005,
            slow_lr=0.001,
            fast_freq=1,
            medium_freq=5,
            slow_freq=20,
        )

        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        initial_loss = F.cross_entropy(model(x), y).item()

        for _ in range(100):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        final_loss = F.cross_entropy(model(x), y).item()

        assert final_loss < initial_loss, "Optimizer should reduce loss"

    def test_all_timescales_contribute(self):
        """Test that all timescales contribute to optimization."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        opt = NestedOptimizer(
            model.parameters(),
            fast_freq=1,
            medium_freq=5,
            slow_freq=10,
        )

        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        fast_count = 0
        medium_count = 0
        slow_count = 0

        for _ in range(50):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            result = opt.step()

            if result["fast_stepped"]:
                fast_count += 1
            if result["medium_stepped"]:
                medium_count += 1
            if result["slow_stepped"]:
                slow_count += 1

        assert fast_count == 50
        assert medium_count == 10
        assert slow_count == 5


class TestNestedOptimizerStateDict:
    """Tests for state dict functionality."""

    def test_state_dict_save_load(self):
        """Test saving and loading state dict."""
        model = nn.Linear(10, 2)
        opt = NestedOptimizer(model.parameters())

        # Take some steps
        for _ in range(15):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            opt.step()

        # Save state
        state = opt.state_dict()

        # Create new optimizer and load state
        opt2 = NestedOptimizer(model.parameters())
        opt2.load_state_dict(state)

        assert opt2.step_count == opt.step_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
