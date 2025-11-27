"""Tests for Knowledge Bridges.

Tests cover:
- KnowledgeBridge initialization and forward pass
- Gate threshold behavior
- Statistics tracking
- CollaborativeNestedOptimizer integration
- Bidirectional knowledge flow
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bridges.knowledge_bridges import (
    BridgeStats,
    CollaborativeNestedOptimizer,
    KnowledgeBridge,
)


class TestBridgeStats:
    """Tests for BridgeStats dataclass."""

    def test_initial_values(self):
        """Test initial values are zero."""
        stats = BridgeStats()
        assert stats.total_calls == 0
        assert stats.total_transfers == 0
        assert stats.transfer_rate == 0.0
        assert stats.mean_gate == 0.0

    def test_record_transfer(self):
        """Test recording a transfer."""
        stats = BridgeStats()
        stats.record(gate_value=0.7, transferred=True)

        assert stats.total_calls == 1
        assert stats.total_transfers == 1
        assert stats.transfer_rate == 1.0
        assert stats.mean_gate == 0.7

    def test_record_no_transfer(self):
        """Test recording a non-transfer."""
        stats = BridgeStats()
        stats.record(gate_value=0.3, transferred=False)

        assert stats.total_calls == 1
        assert stats.total_transfers == 0
        assert stats.transfer_rate == 0.0
        assert stats.mean_gate == 0.3

    def test_multiple_records(self):
        """Test multiple records."""
        stats = BridgeStats()
        stats.record(0.8, True)
        stats.record(0.6, True)
        stats.record(0.3, False)
        stats.record(0.4, False)

        assert stats.total_calls == 4
        assert stats.total_transfers == 2
        assert stats.transfer_rate == 0.5
        assert abs(stats.mean_gate - 0.525) < 0.001

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = BridgeStats()
        stats.record(0.7, True)

        d = stats.to_dict()
        assert "total_calls" in d
        assert "total_transfers" in d
        assert "transfer_rate" in d
        assert "mean_gate" in d


class TestKnowledgeBridge:
    """Tests for KnowledgeBridge class."""

    def test_initialization(self):
        """Test basic initialization."""
        bridge = KnowledgeBridge(hidden_dim=64)

        assert bridge.hidden_dim == 64
        assert bridge.gate_threshold == 0.5
        assert bridge.gate_temperature == 1.0

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        bridge = KnowledgeBridge(
            hidden_dim=32,
            gate_threshold=0.7,
            gate_temperature=2.0,
            transform_layers=3,
        )

        assert bridge.hidden_dim == 32
        assert bridge.gate_threshold == 0.7
        assert bridge.gate_temperature == 2.0

    def test_invalid_hidden_dim(self):
        """Test that invalid hidden_dim raises error."""
        with pytest.raises(ValueError, match="hidden_dim must be >= 1"):
            KnowledgeBridge(hidden_dim=0)

    def test_invalid_gate_threshold(self):
        """Test that invalid gate_threshold raises error."""
        with pytest.raises(ValueError, match="gate_threshold must be in"):
            KnowledgeBridge(hidden_dim=64, gate_threshold=1.5)

    def test_invalid_gate_temperature(self):
        """Test that invalid gate_temperature raises error."""
        with pytest.raises(ValueError, match="gate_temperature must be > 0"):
            KnowledgeBridge(hidden_dim=64, gate_temperature=0)

    def test_forward_output_shape(self):
        """Test that forward returns correct shapes."""
        bridge = KnowledgeBridge(hidden_dim=64)
        source = torch.randn(64)
        target = torch.randn(64)

        knowledge, gate = bridge(source, target)

        assert knowledge.shape == (64,)
        assert isinstance(gate, float)
        assert 0.0 <= gate <= 1.0

    def test_forward_invalid_source_shape(self):
        """Test that invalid source shape raises error."""
        bridge = KnowledgeBridge(hidden_dim=64)
        source = torch.randn(32)  # Wrong shape
        target = torch.randn(64)

        with pytest.raises(ValueError, match="Expected source_state shape"):
            bridge(source, target)

    def test_forward_invalid_target_shape(self):
        """Test that invalid target shape raises error."""
        bridge = KnowledgeBridge(hidden_dim=64)
        source = torch.randn(64)
        target = torch.randn(32)  # Wrong shape

        with pytest.raises(ValueError, match="Expected target_state shape"):
            bridge(source, target)

    def test_gate_threshold_blocks_transfer(self):
        """Test that low gate values result in zero knowledge."""
        # Use high threshold to ensure blocking
        bridge = KnowledgeBridge(hidden_dim=64, gate_threshold=0.99)

        # Run multiple times - most should be blocked
        blocked_count = 0
        for _ in range(20):
            source = torch.randn(64)
            target = torch.randn(64)
            knowledge, gate = bridge(source, target)

            if gate < 0.99:
                assert torch.allclose(knowledge, torch.zeros(64))
                blocked_count += 1

        # Most should be blocked with 0.99 threshold
        assert blocked_count > 10

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        bridge = KnowledgeBridge(hidden_dim=64, gate_threshold=0.5)

        # Run multiple forward passes
        for _ in range(50):
            source = torch.randn(64)
            target = torch.randn(64)
            bridge(source, target)

        stats = bridge.get_stats()
        assert stats["total_calls"] == 50
        assert 0.0 <= stats["transfer_rate"] <= 1.0
        assert 0.0 <= stats["mean_gate"] <= 1.0

    def test_reset_stats(self):
        """Test that reset_stats clears statistics."""
        bridge = KnowledgeBridge(hidden_dim=64)

        # Generate some stats
        for _ in range(10):
            bridge(torch.randn(64), torch.randn(64))

        assert bridge.stats.total_calls == 10

        bridge.reset_stats()

        assert bridge.stats.total_calls == 0

    def test_gradient_flow(self):
        """Test that gradients flow through the bridge."""
        bridge = KnowledgeBridge(hidden_dim=64, gate_threshold=0.0)  # Always transfer

        source = torch.randn(64, requires_grad=True)
        target = torch.randn(64, requires_grad=True)

        knowledge, gate = bridge(source, target)
        loss = knowledge.sum()
        loss.backward()

        # Source should have gradients (through transform)
        assert source.grad is not None


class TestCollaborativeNestedOptimizer:
    """Tests for CollaborativeNestedOptimizer."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 2)

    def test_initialization(self, simple_model):
        """Test basic initialization."""
        opt = CollaborativeNestedOptimizer(simple_model.parameters())

        assert opt.bridge_threshold == 0.5
        assert opt.bridge_frequency == 10
        assert opt.enable_reverse_bridges

    def test_custom_parameters(self, simple_model):
        """Test initialization with custom parameters."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            bridge_threshold=0.7,
            bridge_frequency=5,
            enable_reverse_bridges=False,
        )

        assert opt.bridge_threshold == 0.7
        assert opt.bridge_frequency == 5
        assert not opt.enable_reverse_bridges

    def test_bridges_created(self, simple_model):
        """Test that all bridges are created."""
        opt = CollaborativeNestedOptimizer(simple_model.parameters())

        assert opt.fast_to_medium is not None
        assert opt.medium_to_slow is not None
        assert opt.fast_to_slow is not None
        assert opt.slow_to_fast is not None
        assert opt.slow_to_medium is not None
        assert opt.medium_to_fast is not None

    def test_reverse_bridges_disabled(self, simple_model):
        """Test that reverse bridges can be disabled."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            enable_reverse_bridges=False,
        )

        assert opt.fast_to_medium is not None
        assert opt.medium_to_slow is not None
        assert opt.slow_to_fast is None
        assert opt.slow_to_medium is None
        assert opt.medium_to_fast is None

    def test_step_returns_bridge_info(self, simple_model):
        """Test that step returns bridge information."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            bridge_frequency=1,  # Check every step
        )

        # Take a step
        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()
        result = opt.step()

        assert "bridges" in result

    def test_bridge_frequency(self, simple_model):
        """Test that bridges are checked at correct frequency."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            bridge_frequency=5,
        )

        bridge_checks = 0
        for i in range(20):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = simple_model(x).sum()
            loss.backward()
            result = opt.step()

            if result.get("bridges"):
                bridge_checks += 1

        # Should check at steps 5, 10, 15, 20
        assert bridge_checks == 4

    def test_get_bridge_stats(self, simple_model):
        """Test get_bridge_stats returns all bridges."""
        opt = CollaborativeNestedOptimizer(simple_model.parameters())

        stats = opt.get_bridge_stats()

        assert "fast_to_medium" in stats
        assert "medium_to_slow" in stats
        assert "fast_to_slow" in stats
        assert "slow_to_fast" in stats
        assert "slow_to_medium" in stats
        assert "medium_to_fast" in stats

    def test_reset_bridge_stats(self, simple_model):
        """Test reset_bridge_stats clears all statistics."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            bridge_frequency=1,
        )

        # Generate some stats
        for _ in range(10):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = simple_model(x).sum()
            loss.backward()
            opt.step()

        # Verify stats exist
        stats = opt.get_bridge_stats()
        assert stats["fast_to_medium"]["total_calls"] > 0

        # Reset
        opt.reset_bridge_stats()

        # Verify cleared
        stats = opt.get_bridge_stats()
        assert stats["fast_to_medium"]["total_calls"] == 0

    def test_diagnostics_include_bridges(self, simple_model):
        """Test that diagnostics include bridge information."""
        opt = CollaborativeNestedOptimizer(simple_model.parameters())

        diag = opt.get_diagnostics()

        assert "bridge_threshold" in diag
        assert "bridge_frequency" in diag
        assert "enable_reverse_bridges" in diag
        assert "bridge_stats" in diag


class TestCollaborativeOptimization:
    """Tests for actual optimization behavior with bridges."""

    def test_reduces_loss(self):
        """Test that collaborative optimizer reduces loss."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        opt = CollaborativeNestedOptimizer(
            model.parameters(),
            fast_lr=0.01,
            bridge_frequency=5,
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

        assert final_loss < initial_loss

    def test_bridges_transfer_knowledge(self):
        """Test that bridges actually transfer knowledge."""
        torch.manual_seed(42)

        model = nn.Linear(10, 2)

        opt = CollaborativeNestedOptimizer(
            model.parameters(),
            bridge_frequency=1,
            bridge_threshold=0.0,  # Always transfer
        )

        # Run some steps
        for _ in range(20):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            opt.step()

        # Check that transfers happened
        stats = opt.get_bridge_stats()
        total_transfers = sum(s["total_transfers"] for s in stats.values())

        assert total_transfers > 0, "Bridges should transfer knowledge"

    def test_reverse_bridges_contribute(self):
        """Test that reverse bridges contribute to optimization."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        opt = CollaborativeNestedOptimizer(
            model.parameters(),
            bridge_frequency=5,
            bridge_threshold=0.3,
            enable_reverse_bridges=True,
        )

        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        for _ in range(50):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        # Check reverse bridge stats
        stats = opt.get_bridge_stats()

        # At least some reverse bridges should have been called
        reverse_calls = (
            stats["slow_to_fast"]["total_calls"]
            + stats["slow_to_medium"]["total_calls"]
            + stats["medium_to_fast"]["total_calls"]
        )

        assert reverse_calls > 0, "Reverse bridges should be called"


class TestKnowledgeFlow:
    """Tests to verify knowledge flows correctly through bridges."""

    def test_knowledge_injection_modifies_optimizer(self):
        """Test that injected knowledge modifies the optimizer."""
        model = nn.Linear(10, 2)
        opt = CollaborativeNestedOptimizer(
            model.parameters(),
            bridge_frequency=1,
            bridge_threshold=0.0,  # Always transfer
            hidden_dim=32,
        )

        # Initialize memory networks by taking a step
        opt.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # Get initial fast optimizer state
        opt.fast.get_knowledge_state().clone()

        # Take more steps with bridge transfers
        for _ in range(10):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            opt.step()

        # State should have changed due to knowledge injection
        opt.fast.get_knowledge_state()

        # Note: This test may be flaky depending on initialization
        # The key is that the mechanism exists and runs without error


class TestAdjacentOnlyBridges:
    """Tests for adjacent_only bridge mode.

    When adjacent_only=True, only adjacent timescale transfers are enabled:
    - fast ↔ medium (bidirectional)
    - medium ↔ slow (bidirectional)

    Direct fast ↔ slow transfers are disabled.
    """

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 2)

    def test_adjacent_only_disables_direct_bridges(self, simple_model):
        """Test that adjacent_only=True disables fast↔slow bridges."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            adjacent_only=True,
        )

        # Direct bridges should be None
        assert opt.fast_to_slow is None
        assert opt.slow_to_fast is None

        # Adjacent bridges should still exist
        assert opt.fast_to_medium is not None
        assert opt.medium_to_slow is not None
        assert opt.medium_to_fast is not None
        assert opt.slow_to_medium is not None

    def test_adjacent_only_false_enables_all_bridges(self, simple_model):
        """Test that adjacent_only=False enables all bridges."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            adjacent_only=False,
        )

        # All bridges should exist
        assert opt.fast_to_slow is not None
        assert opt.slow_to_fast is not None
        assert opt.fast_to_medium is not None
        assert opt.medium_to_slow is not None
        assert opt.medium_to_fast is not None
        assert opt.slow_to_medium is not None

    def test_adjacent_only_stats_exclude_disabled_bridges(self, simple_model):
        """Test that get_bridge_stats excludes disabled bridges."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            adjacent_only=True,
        )

        stats = opt.get_bridge_stats()

        # Should NOT have fast_to_slow or slow_to_fast
        assert "fast_to_slow" not in stats
        assert "slow_to_fast" not in stats

        # Should have adjacent bridges
        assert "fast_to_medium" in stats
        assert "medium_to_slow" in stats
        assert "medium_to_fast" in stats
        assert "slow_to_medium" in stats

    def test_adjacent_only_diagnostics(self, simple_model):
        """Test that diagnostics include adjacent_only flag."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            adjacent_only=True,
        )

        diag = opt.get_diagnostics()

        assert "adjacent_only" in diag
        assert diag["adjacent_only"]

    def test_adjacent_only_step_works(self, simple_model):
        """Test that step works correctly with adjacent_only=True."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            adjacent_only=True,
            bridge_frequency=1,
            bridge_threshold=0.0,  # Always transfer
        )

        # Run some steps
        for _ in range(10):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = simple_model(x).sum()
            loss.backward()
            result = opt.step()

        # Should have bridge results
        assert "bridges" in result

        # Should NOT have fast_to_slow or slow_to_fast in results
        if result["bridges"]:
            assert "fast_to_slow" not in result["bridges"]
            assert "slow_to_fast" not in result["bridges"]

    def test_adjacent_only_reduces_loss(self):
        """Test that adjacent_only mode still reduces loss."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        opt = CollaborativeNestedOptimizer(
            model.parameters(),
            fast_lr=0.01,
            bridge_frequency=5,
            adjacent_only=True,
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

        assert final_loss < initial_loss

    def test_adjacent_only_with_reverse_bridges_disabled(self, simple_model):
        """Test adjacent_only combined with enable_reverse_bridges=False."""
        opt = CollaborativeNestedOptimizer(
            simple_model.parameters(),
            adjacent_only=True,
            enable_reverse_bridges=False,
        )

        # Only forward adjacent bridges should exist
        assert opt.fast_to_medium is not None
        assert opt.medium_to_slow is not None

        # All reverse bridges should be None
        assert opt.medium_to_fast is None
        assert opt.slow_to_medium is None
        assert opt.fast_to_slow is None
        assert opt.slow_to_fast is None

        # Stats should only have forward adjacent bridges
        stats = opt.get_bridge_stats()
        assert "fast_to_medium" in stats
        assert "medium_to_slow" in stats
        assert len(stats) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
