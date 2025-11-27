"""Depth Ablation Study: Compare 3, 5, and 7 level architectures.

This experiment tests whether deeper timescale hierarchies improve
continual learning performance. Brain-inspired hypothesis: more
timescales = better temporal abstraction.

Configurations:
- 3 levels: gamma, beta, alpha (freq_ratio=10)
- 5 levels: gamma, beta, alpha, theta, delta (freq_ratio=5)
- 7 levels: gamma, beta, alpha, theta, delta, infra_slow, ultra_slow (freq_ratio=3)

All configurations span similar total frequency range (~100-1000x).
"""

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.memory.multi_scale_cms import MultiScaleCMS, MultiScaleCMSConfig
from src.optimizers.multi_scale import MultiScaleConfig, MultiScaleNestedOptimizer


@dataclass
class DepthConfig:
    """Configuration for a depth ablation experiment."""

    name: str
    num_levels: int
    freq_ratio: float

    def get_total_range(self) -> float:
        """Get the total frequency range (slowest / fastest)."""
        return self.freq_ratio ** (self.num_levels - 1)


# Configurations that span similar total frequency ranges
DEPTH_CONFIGS = [
    DepthConfig(name="3-level", num_levels=3, freq_ratio=10.0),  # Range: 100x
    DepthConfig(name="5-level", num_levels=5, freq_ratio=5.0),  # Range: 625x
    DepthConfig(name="7-level", num_levels=7, freq_ratio=3.0),  # Range: 729x
]


class SimpleMLP(nn.Module):
    """Simple MLP for continual learning experiments."""

    def __init__(
        self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def create_synthetic_task(
    task_id: int,
    num_samples: int = 1000,
    input_dim: int = 784,
    num_classes: int = 10,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic classification task.

    Each task has different input distribution but same output structure.
    """
    if seed is not None:
        torch.manual_seed(seed + task_id)

    # Different mean/std for each task
    mean = task_id * 0.5
    std = 1.0 + task_id * 0.1

    X = torch.randn(num_samples, input_dim) * std + mean
    y = torch.randint(0, num_classes, (num_samples,))

    return X, y


def train_epoch(
    model: nn.Module,
    optimizer: MultiScaleNestedOptimizer,
    cms: Optional[MultiScaleCMS],
    dataloader: DataLoader,
    reg_strength: float = 0.1,
    log_bridges: bool = False,
) -> Tuple[float, Dict]:
    """Train for one epoch.

    Returns:
        Tuple of (average loss, bridge activity dict)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Track bridge activity
    bridge_transfers = 0
    bridge_attempts = 0
    gate_values = []

    params = list(model.parameters())

    for X, y in dataloader:
        optimizer.zero_grad()

        # Forward pass
        output = model(X)
        task_loss = F.cross_entropy(output, y)

        # Add regularization from CMS
        if cms is not None:
            reg_loss = cms.compute_regularization_loss(params)
            loss = task_loss + reg_strength * reg_loss
        else:
            loss = task_loss

        # Backward pass
        loss.backward()

        # Optimizer step - capture bridge info
        step_result = optimizer.step()

        # Log bridge activity
        if "bridges" in step_result and step_result["bridges"]:
            bridge_attempts += 1
            for bridge_name, info in step_result["bridges"].items():
                gate_values.append(info["gate"])
                if info["transferred"]:
                    bridge_transfers += 1

        # Update CMS
        if cms is not None:
            cms.update(params)
            cms.accumulate_importance(params)

        total_loss += task_loss.item()
        num_batches += 1

    bridge_activity = {
        "attempts": bridge_attempts,
        "transfers": bridge_transfers,
        "avg_gate": sum(gate_values) / len(gate_values) if gate_values else 0.0,
        "min_gate": min(gate_values) if gate_values else 0.0,
        "max_gate": max(gate_values) if gate_values else 0.0,
    }

    return total_loss / num_batches, bridge_activity


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
) -> Tuple[float, float]:
    """Evaluate model accuracy and loss."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            output = model(X)
            loss = F.cross_entropy(output, y)

            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
            num_batches += 1

    accuracy = correct / total
    avg_loss = total_loss / num_batches

    return accuracy, avg_loss


def run_depth_experiment(
    depth_config: DepthConfig,
    num_tasks: int = 5,
    epochs_per_task: int = 10,
    batch_size: int = 64,
    lr: float = 0.01,
    reg_strength: float = 0.1,
    use_bridges: bool = True,
    seed: int = 42,
) -> Dict:
    """Run a single depth configuration experiment.

    Returns:
        Dictionary with results including per-task accuracies and forgetting.
    """
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(
        f"Running {depth_config.name} (levels={depth_config.num_levels}, "
        f"freq_ratio={depth_config.freq_ratio}, range={depth_config.get_total_range():.0f}x)"
    )
    print(f"Bridges: {'enabled' if use_bridges else 'disabled'}")
    print(f"{'='*60}")

    # Create model
    model = SimpleMLP()
    params = list(model.parameters())

    # Create multi-scale optimizer
    opt_config = MultiScaleConfig(
        num_levels=depth_config.num_levels,
        freq_ratio=depth_config.freq_ratio,
        base_lr=lr,
        bridge_threshold=0.3,
        bridge_frequency=10 if use_bridges else 1000000,  # Very high = never transfer
    )
    optimizer = MultiScaleNestedOptimizer(params, opt_config)

    # Create multi-scale CMS
    cms_config = MultiScaleCMSConfig(
        num_levels=depth_config.num_levels,
        freq_ratio=depth_config.freq_ratio,
        regularization_strength=reg_strength,
    )
    cms = MultiScaleCMS(params, cms_config)

    # Create tasks
    tasks = []
    for task_id in range(num_tasks):
        X, y = create_synthetic_task(task_id, seed=seed)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        tasks.append(dataloader)

    # Track accuracies
    accuracy_matrix = []  # [task_trained][task_evaluated]

    # Track bridge activity across all training
    total_bridge_activity = {
        "attempts": 0,
        "transfers": 0,
        "gate_values": [],
    }

    # Train on each task sequentially
    for task_id, train_loader in enumerate(tasks):
        print(f"\n  Training on Task {task_id + 1}/{num_tasks}...")

        for epoch in range(epochs_per_task):
            loss, bridge_activity = train_epoch(
                model, optimizer, cms, train_loader, reg_strength, log_bridges=True
            )

            # Accumulate bridge stats
            total_bridge_activity["attempts"] += bridge_activity["attempts"]
            total_bridge_activity["transfers"] += bridge_activity["transfers"]
            if bridge_activity["avg_gate"] > 0:
                total_bridge_activity["gate_values"].append(bridge_activity["avg_gate"])

            if epoch == epochs_per_task - 1:
                print(f"    Final epoch loss: {loss:.4f}")
                if bridge_activity["attempts"] > 0:
                    print(
                        f"    Bridge activity: {bridge_activity['transfers']}/{bridge_activity['attempts']} transfers, "
                        f"avg_gate={bridge_activity['avg_gate']:.3f}, "
                        f"range=[{bridge_activity['min_gate']:.3f}, {bridge_activity['max_gate']:.3f}]"
                    )
                elif use_bridges:
                    print(
                        f"    Bridge activity: 0 attempts (bridge_frequency too high)"
                    )

        # Consolidate after task
        cms.consolidate(params)
        cms.reset_importance()

        # Evaluate on all tasks seen so far
        task_accuracies = []
        for eval_task_id in range(task_id + 1):
            acc, _ = evaluate(model, tasks[eval_task_id])
            task_accuracies.append(acc)

        accuracy_matrix.append(task_accuracies)
        print(
            f"    Accuracies after Task {task_id + 1}: {[f'{a:.2%}' for a in task_accuracies]}"
        )

    # Compute metrics
    final_accuracies = accuracy_matrix[-1]
    avg_accuracy = sum(final_accuracies) / len(final_accuracies)

    # Compute forgetting
    forgetting = []
    for task_id in range(num_tasks - 1):
        # Best accuracy on this task (right after training)
        best_acc = accuracy_matrix[task_id][task_id]
        # Final accuracy on this task
        final_acc = accuracy_matrix[-1][task_id]
        forgetting.append(best_acc - final_acc)

    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0

    # Get bridge stats if enabled
    bridge_stats = optimizer.get_bridge_stats() if use_bridges else None

    # Print overall bridge summary
    if use_bridges:
        total_attempts = total_bridge_activity["attempts"]
        total_transfers = total_bridge_activity["transfers"]
        avg_gate = (
            sum(total_bridge_activity["gate_values"])
            / len(total_bridge_activity["gate_values"])
            if total_bridge_activity["gate_values"]
            else 0.0
        )
        print(f"\n  Bridge Summary:")
        print(f"    Total attempts: {total_attempts}")
        print(f"    Total transfers: {total_transfers}")
        print(f"    Transfer rate: {total_transfers/max(1,total_attempts):.1%}")
        print(f"    Average gate value: {avg_gate:.3f}")

    results = {
        "config": depth_config.name,
        "num_levels": depth_config.num_levels,
        "freq_ratio": depth_config.freq_ratio,
        "total_range": depth_config.get_total_range(),
        "use_bridges": use_bridges,
        "final_accuracies": final_accuracies,
        "avg_accuracy": avg_accuracy,
        "forgetting": forgetting,
        "avg_forgetting": avg_forgetting,
        "bridge_stats": bridge_stats,
    }

    print(f"\n  Results for {depth_config.name}:")
    print(f"    Average Accuracy: {avg_accuracy:.2%}")
    print(f"    Average Forgetting: {avg_forgetting:.2%}")

    return results


def run_full_ablation(
    num_tasks: int = 5,
    epochs_per_task: int = 10,
    reg_strength: float = 0.1,
    seed: int = 42,
) -> List[Dict]:
    """Run full depth ablation study.

    Tests all depth configurations with and without bridges.
    """
    print("\n" + "=" * 70)
    print("DEPTH ABLATION STUDY")
    print("Comparing 3, 5, and 7 level architectures")
    print("=" * 70)

    all_results = []

    for depth_config in DEPTH_CONFIGS:
        # With bridges
        results_with = run_depth_experiment(
            depth_config,
            num_tasks=num_tasks,
            epochs_per_task=epochs_per_task,
            reg_strength=reg_strength,
            use_bridges=True,
            seed=seed,
        )
        all_results.append(results_with)

        # Without bridges
        results_without = run_depth_experiment(
            depth_config,
            num_tasks=num_tasks,
            epochs_per_task=epochs_per_task,
            reg_strength=reg_strength,
            use_bridges=False,
            seed=seed,
        )
        all_results.append(results_without)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<15} {'Bridges':<10} {'Avg Acc':<12} {'Avg Forget':<12}")
    print("-" * 50)

    for r in all_results:
        bridges_str = "Yes" if r["use_bridges"] else "No"
        print(
            f"{r['config']:<15} {bridges_str:<10} {r['avg_accuracy']:.2%}       {r['avg_forgetting']:.2%}"
        )

    # Find best configuration
    best_result = max(
        all_results, key=lambda x: x["avg_accuracy"] - x["avg_forgetting"]
    )
    print(
        f"\nBest configuration: {best_result['config']} "
        f"(bridges={'Yes' if best_result['use_bridges'] else 'No'})"
    )
    print(f"  Accuracy: {best_result['avg_accuracy']:.2%}")
    print(f"  Forgetting: {best_result['avg_forgetting']:.2%}")

    return all_results


if __name__ == "__main__":
    start_time = time.time()

    results = run_full_ablation(
        num_tasks=5,
        epochs_per_task=10,
        reg_strength=0.1,  # Lower regularization to allow learning
        seed=42,
    )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
