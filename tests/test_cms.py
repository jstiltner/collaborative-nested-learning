"""Tests for Continuum Memory System.

Tests the MemoryBank and ContinuumMemorySystem classes.
"""

import pytest
import torch
import torch.nn as nn

from src.memory.continuum import CMSConfig, ContinuumMemorySystem
from src.memory.memory_bank import MemoryBank
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer


class TestMemoryBank:
    """Tests for MemoryBank class."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    @pytest.fixture
    def bank(self) -> MemoryBank:
        """Create a memory bank for testing."""
        return MemoryBank(update_frequency=5, decay_rate=0.9, name="test")

    def test_init(self, bank: MemoryBank) -> None:
        """Test memory bank initialization."""
        assert bank.update_frequency == 5
        assert bank.decay_rate == 0.9
        assert bank.name == "test"
        assert bank.step_count == 0
        assert bank.num_params == 0

    def test_init_validation(self) -> None:
        """Test initialization validation."""
        with pytest.raises(ValueError, match="update_frequency"):
            MemoryBank(update_frequency=0)

        with pytest.raises(ValueError, match="decay_rate"):
            MemoryBank(decay_rate=1.5)

    def test_maybe_update(self, bank: MemoryBank, model: nn.Module) -> None:
        """Test conditional update based on frequency."""
        params = list(model.parameters())

        # First 4 steps should not update
        for i in range(4):
            updated = bank.maybe_update(params)
            assert not updated
            assert bank.step_count == i + 1

        # 5th step should update
        updated = bank.maybe_update(params)
        assert updated
        assert bank.step_count == 5
        assert bank.num_params == 2  # weight and bias

    def test_force_update(self, bank: MemoryBank, model: nn.Module) -> None:
        """Test forced update."""
        params = list(model.parameters())

        bank.force_update(params)
        assert bank.num_params == 2
        assert bank.stats.total_updates == 1

    def test_get_memory(self, bank: MemoryBank, model: nn.Module) -> None:
        """Test memory retrieval."""
        params = list(model.parameters())

        # No memory before update
        for p in params:
            assert bank.get_memory(id(p)) is None

        # Update and check memory
        bank.force_update(params)

        for p in params:
            memory = bank.get_memory(id(p))
            assert memory is not None
            assert memory.shape == p.shape
            assert torch.allclose(memory, p.data)

    def test_decay_update(self, model: nn.Module) -> None:
        """Test EMA decay during updates."""
        bank = MemoryBank(update_frequency=1, decay_rate=0.5)
        params = list(model.parameters())

        # First update
        bank.force_update(params)
        first_memory = bank.get_memory(id(params[0])).clone()

        # Modify parameters
        with torch.no_grad():
            params[0].add_(torch.ones_like(params[0]))

        # Second update with decay
        bank.force_update(params)
        second_memory = bank.get_memory(id(params[0]))

        # Memory should be between old and new values
        expected = 0.5 * first_memory + 0.5 * params[0].data
        assert torch.allclose(second_memory, expected)

    def test_compute_drift(self, bank: MemoryBank, model: nn.Module) -> None:
        """Test drift computation."""
        params = list(model.parameters())

        # Store initial state
        bank.force_update(params)

        # Modify parameters
        with torch.no_grad():
            for p in params:
                p.add_(torch.randn_like(p) * 0.1)

        # Compute drift
        drift = bank.compute_drift(params)

        assert len(drift) == 2
        for param_id, d in drift.items():
            assert d > 0  # Should have drifted

    def test_regularization_loss(self, bank: MemoryBank, model: nn.Module) -> None:
        """Test regularization loss computation."""
        params = list(model.parameters())

        # Store initial state
        bank.force_update(params)

        # No drift = zero loss
        reg_loss = bank.compute_regularization_loss(params, strength=1.0)
        assert torch.allclose(reg_loss, torch.tensor(0.0), atol=1e-6)

        # Modify parameters
        with torch.no_grad():
            for p in params:
                p.add_(torch.ones_like(p) * 0.1)

        # Should have positive loss now
        reg_loss = bank.compute_regularization_loss(params, strength=1.0)
        assert reg_loss.item() > 0

    def test_consolidation(self, model: nn.Module) -> None:
        """Test memory consolidation between banks."""
        fast_bank = MemoryBank(update_frequency=1, name="fast")
        slow_bank = MemoryBank(update_frequency=10, name="slow")

        params = list(model.parameters())

        # Update fast bank
        fast_bank.force_update(params)

        # Consolidate to slow bank
        consolidated = fast_bank.consolidate_to(slow_bank, params, threshold=0.0)

        assert consolidated == 2  # Both params consolidated
        assert slow_bank.num_params == 2

    def test_state_dict(self, bank: MemoryBank, model: nn.Module) -> None:
        """Test state serialization."""
        params = list(model.parameters())
        bank.force_update(params)

        # Save state
        state = bank.state_dict()

        # Create new bank and load
        new_bank = MemoryBank()
        new_bank.load_state_dict(state)

        assert new_bank.name == bank.name
        assert new_bank.update_frequency == bank.update_frequency
        assert new_bank.num_params == bank.num_params


class TestContinuumMemorySystem:
    """Tests for ContinuumMemorySystem class."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    @pytest.fixture
    def cms(self, model: nn.Module) -> ContinuumMemorySystem:
        """Create a CMS for testing."""
        config = CMSConfig(
            fast_frequency=1,
            medium_frequency=5,
            slow_frequency=20,
        )
        return ContinuumMemorySystem(model.parameters(), config)

    def test_init(self, cms: ContinuumMemorySystem) -> None:
        """Test CMS initialization."""
        assert cms.config.fast_frequency == 1
        assert cms.config.medium_frequency == 5
        assert cms.config.slow_frequency == 20
        assert cms.step_count == 0

    def test_init_validation(self) -> None:
        """Test initialization validation."""
        with pytest.raises(ValueError, match="fast_frequency"):
            ContinuumMemorySystem(
                config=CMSConfig(
                    fast_frequency=10,
                    medium_frequency=5,
                )
            )

    def test_update(self, cms: ContinuumMemorySystem, model: nn.Module) -> None:
        """Test memory bank updates."""
        params = list(model.parameters())

        # Run 25 steps
        fast_updates = 0
        medium_updates = 0
        slow_updates = 0

        for _ in range(25):
            result = cms.update(params)
            if result["fast"]:
                fast_updates += 1
            if result["medium"]:
                medium_updates += 1
            if result["slow"]:
                slow_updates += 1

        assert fast_updates == 25  # Every step
        assert medium_updates == 5  # Every 5 steps
        assert slow_updates == 1  # Every 20 steps

    def test_regularization_loss(
        self, cms: ContinuumMemorySystem, model: nn.Module
    ) -> None:
        """Test regularization loss computation."""
        params = list(model.parameters())

        # Initial loss should be zero (no drift)
        reg_loss = cms.compute_regularization_loss(params)
        assert torch.allclose(reg_loss, torch.tensor(0.0), atol=1e-6)

        # Modify parameters
        with torch.no_grad():
            for p in params:
                p.add_(torch.randn_like(p) * 0.1)

        # Should have positive loss now
        reg_loss = cms.compute_regularization_loss(params)
        assert reg_loss.item() > 0

    def test_importance_accumulation(
        self, cms: ContinuumMemorySystem, model: nn.Module
    ) -> None:
        """Test importance score accumulation."""
        # Create fake gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p)

        # Accumulate importance
        cms.accumulate_importance(model.parameters())
        cms.accumulate_importance(model.parameters())

        importance = cms.get_importance()
        assert len(importance) > 0

        # Reset
        cms.reset_importance()
        importance = cms.get_importance()
        assert len(importance) == 0

    def test_consolidation(self, cms: ContinuumMemorySystem, model: nn.Module) -> None:
        """Test memory consolidation."""
        params = list(model.parameters())

        # Run some steps
        for _ in range(10):
            cms.update(params)

        # Consolidate
        result = cms.consolidate(params)

        assert "fast_to_medium" in result
        assert "medium_to_slow" in result

    def test_memory_retrieval(
        self, cms: ContinuumMemorySystem, model: nn.Module
    ) -> None:
        """Test memory retrieval from different banks."""
        params = list(model.parameters())
        param_id = id(params[0])

        # Get memory from each bank
        fast_mem = cms.get_memory(param_id, "fast")
        medium_mem = cms.get_memory(param_id, "medium")
        slow_mem = cms.get_memory(param_id, "slow")

        assert fast_mem is not None
        assert medium_mem is not None
        assert slow_mem is not None

    def test_consolidated_memory(
        self, cms: ContinuumMemorySystem, model: nn.Module
    ) -> None:
        """Test weighted memory consolidation."""
        params = list(model.parameters())
        param_id = id(params[0])

        consolidated = cms.get_consolidated_memory(param_id)

        assert consolidated is not None
        assert consolidated.shape == params[0].shape

    def test_drift_computation(
        self, cms: ContinuumMemorySystem, model: nn.Module
    ) -> None:
        """Test drift computation from all banks."""
        params = list(model.parameters())

        # Modify parameters
        with torch.no_grad():
            for p in params:
                p.add_(torch.randn_like(p) * 0.1)

        drift = cms.compute_drift(params)

        assert "fast" in drift
        assert "medium" in drift
        assert "slow" in drift

    def test_state_dict(self, cms: ContinuumMemorySystem, model: nn.Module) -> None:
        """Test state serialization."""
        params = list(model.parameters())

        # Run some steps
        for _ in range(10):
            cms.update(params)

        # Save state
        state = cms.state_dict()

        # Create new CMS and load
        new_cms = ContinuumMemorySystem()
        new_cms.load_state_dict(state)

        assert new_cms.step_count == cms.step_count


class TestCollaborativeCMSOptimizer:
    """Tests for CollaborativeCMSOptimizer class."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    @pytest.fixture
    def optimizer(self, model: nn.Module) -> CollaborativeCMSOptimizer:
        """Create an optimizer for testing."""
        return CollaborativeCMSOptimizer(
            model.parameters(),
            fast_lr=0.01,
            medium_lr=0.005,
            slow_lr=0.001,
            fast_freq=1,
            medium_freq=5,
            slow_freq=20,
            hidden_dim=32,
        )

    def test_init(self, optimizer: CollaborativeCMSOptimizer) -> None:
        """Test optimizer initialization."""
        assert optimizer.use_cms_regularization
        assert optimizer.consolidate_on_task_switch
        assert optimizer.cms is not None

    def test_step_updates_cms(
        self, optimizer: CollaborativeCMSOptimizer, model: nn.Module
    ) -> None:
        """Test that step updates CMS."""
        # Create fake gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p)

        result = optimizer.step()

        assert "cms_update" in result
        assert result["cms_update"]["fast"]

    def test_regularization_loss(
        self, optimizer: CollaborativeCMSOptimizer, model: nn.Module
    ) -> None:
        """Test regularization loss retrieval."""
        reg_loss = optimizer.get_regularization_loss()

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.ndim == 0  # Scalar

    def test_task_switching(
        self, optimizer: CollaborativeCMSOptimizer, model: nn.Module
    ) -> None:
        """Test task switching with consolidation."""
        # Set first task
        result = optimizer.set_task(0)
        assert result["previous_task"] is None
        assert result["new_task"] == 0
        assert not result["consolidated"]

        # Run some steps
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        for _ in range(10):
            optimizer.step()

        # Switch to second task
        result = optimizer.set_task(1)
        assert result["previous_task"] == 0
        assert result["new_task"] == 1
        assert result["consolidated"]

    def test_memory_drift(
        self, optimizer: CollaborativeCMSOptimizer, model: nn.Module
    ) -> None:
        """Test memory drift computation."""
        drift = optimizer.get_memory_drift()

        assert "fast" in drift
        assert "medium" in drift
        assert "slow" in drift

    def test_diagnostics(
        self, optimizer: CollaborativeCMSOptimizer, model: nn.Module
    ) -> None:
        """Test diagnostics retrieval."""
        diag = optimizer.get_diagnostics()

        assert "cms" in diag
        assert "memory_drift" in diag
        assert "current_task" in diag

    def test_training_loop(
        self, optimizer: CollaborativeCMSOptimizer, model: nn.Module
    ) -> None:
        """Test a complete training loop."""
        import torch.nn.functional as F

        torch.manual_seed(42)
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        initial_loss = None

        for step in range(50):
            optimizer.zero_grad()

            output = model(x)
            loss = F.cross_entropy(output, y)

            if initial_loss is None:
                initial_loss = loss.item()

            # Add regularization
            reg_loss = optimizer.get_regularization_loss()
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

        final_loss = F.cross_entropy(model(x), y).item()

        # Should have reduced loss
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
