"""Ablation Study: Comparing Optimizers on Split-MNIST.

This script runs the main experiment to test our hypothesis:
"Bidirectional knowledge bridges improve continual learning over vanilla nested learning"

Compares:
1. SGD (baseline)
2. Adam (baseline)
3. NestedOptimizer (paper's approach)
4. CollaborativeNestedOptimizer (our contribution)
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.metrics import ContinualMetrics, compute_accuracy_matrix
from benchmarks.split_mnist import SplitMNIST
from src.bridges import CollaborativeNestedOptimizer
from src.optimizers import NestedOptimizer


@dataclass
class ExperimentConfig:
    """Configuration for the ablation experiment."""

    # Data
    num_tasks: int = 5
    batch_size: int = 64

    # Training
    steps_per_task: int = 500
    eval_every: int = 100

    # Model
    input_dim: int = 784
    hidden_dims: List[int] = None
    output_dim: int = 10

    # Optimizer settings
    learning_rate: float = 0.01

    # Nested optimizer settings
    fast_freq: int = 1
    medium_freq: int = 10
    slow_freq: int = 50
    hidden_dim: int = 32

    # Bridge settings
    bridge_frequency: int = 10
    bridge_threshold: float = 0.3

    # Reproducibility
    seed: int = 42

    # Output
    results_dir: str = "./experiments/results"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST classification."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = None,
        output_dim: int = 10,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(
    name: str,
    params,
    config: ExperimentConfig,
) -> Any:
    """Create an optimizer by name.

    Args:
        name: Optimizer name ('sgd', 'adam', 'nested', 'collaborative').
        params: Model parameters.
        config: Experiment configuration.

    Returns:
        Optimizer instance.
    """
    if name == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate, momentum=0.9)
    elif name == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate)
    elif name == "nested":
        return NestedOptimizer(
            params,
            fast_lr=config.learning_rate,
            medium_lr=config.learning_rate * 0.5,
            slow_lr=config.learning_rate * 0.1,
            fast_freq=config.fast_freq,
            medium_freq=config.medium_freq,
            slow_freq=config.slow_freq,
            hidden_dim=config.hidden_dim,
        )
    elif name == "collaborative":
        return CollaborativeNestedOptimizer(
            params,
            fast_lr=config.learning_rate,
            medium_lr=config.learning_rate * 0.5,
            slow_lr=config.learning_rate * 0.1,
            fast_freq=config.fast_freq,
            medium_freq=config.medium_freq,
            slow_freq=config.slow_freq,
            hidden_dim=config.hidden_dim,
            bridge_frequency=config.bridge_frequency,
            bridge_threshold=config.bridge_threshold,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_task(
    model: nn.Module,
    optimizer: Any,
    train_loader,
    steps: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train on a single task.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        train_loader: Training data loader.
        steps: Number of training steps.
        device: Device to use.

    Returns:
        Dictionary with training statistics.
    """
    model.train()

    total_loss = 0.0
    step_count = 0
    bridge_transfers = 0

    # Create infinite iterator
    data_iter = iter(train_loader)

    for step in range(steps):
        # Get batch (restart iterator if needed)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Flatten for MLP
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)

        # Backward pass
        loss.backward()

        # Optimizer step
        result = optimizer.step()

        # Track bridge transfers for collaborative optimizer
        if isinstance(result, dict) and "bridges" in result:
            for bridge_info in result.get("bridges", {}).values():
                if bridge_info.get("transferred", False):
                    bridge_transfers += 1

        total_loss += loss.item()
        step_count += 1

    return {
        "avg_loss": total_loss / step_count,
        "bridge_transfers": bridge_transfers,
    }


def run_experiment(
    optimizer_name: str,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Run a single experiment with one optimizer.

    Args:
        optimizer_name: Name of optimizer to use.
        config: Experiment configuration.
        device: Device to use.

    Returns:
        Dictionary with experiment results.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment with: {optimizer_name}")
    print(f"{'='*60}")

    # Set seed
    set_seed(config.seed)

    # Create model
    model = SimpleMLP(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
    ).to(device)

    # Create optimizer
    optimizer = create_optimizer(optimizer_name, model.parameters(), config)

    # Create benchmark
    benchmark = SplitMNIST(
        root="./data",
        num_tasks=config.num_tasks,
        batch_size=config.batch_size,
    )

    # Get all test loaders for evaluation
    test_loaders = benchmark.get_all_test_loaders()

    # Track metrics
    metrics = ContinualMetrics(num_tasks=config.num_tasks)
    total_bridge_transfers = 0
    training_time = 0.0

    # Train on each task sequentially
    for task_id in range(config.num_tasks):
        print(
            f"\n  Task {task_id}: Training on digits {benchmark.get_task_classes(task_id)}"
        )

        train_loader, _ = benchmark.get_task(task_id)

        # Train
        start_time = time.time()
        train_stats = train_task(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            steps=config.steps_per_task,
            device=device,
        )
        training_time += time.time() - start_time

        total_bridge_transfers += train_stats.get("bridge_transfers", 0)

        # Evaluate on all tasks
        accuracies = compute_accuracy_matrix(model, test_loaders, device)
        metrics.update(task_id, accuracies)

        print(f"    Loss: {train_stats['avg_loss']:.4f}")
        print(f"    Accuracies: {[f'{a:.2f}' for a in accuracies]}")

    # Get bridge stats if available
    bridge_stats = {}
    if hasattr(optimizer, "get_bridge_stats"):
        bridge_stats = optimizer.get_bridge_stats()

    # Compile results
    results = {
        "optimizer": optimizer_name,
        "config": asdict(config),
        "metrics": metrics.summary(),
        "training_time": training_time,
        "total_bridge_transfers": total_bridge_transfers,
        "bridge_stats": bridge_stats,
    }

    print("\n  Final Results:")
    print(f"    Average Accuracy: {metrics.average_accuracy:.4f}")
    print(f"    Forgetting: {metrics.forgetting:.4f}")
    print(f"    Forward Transfer: {metrics.forward_transfer:.4f}")
    print(f"    Backward Transfer: {metrics.backward_transfer:.4f}")
    if total_bridge_transfers > 0:
        print(f"    Bridge Transfers: {total_bridge_transfers}")

    return results


def run_ablation(config: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """Run the full ablation study.

    Args:
        config: Experiment configuration (uses defaults if None).

    Returns:
        Dictionary with all results.
    """
    if config is None:
        config = ExperimentConfig()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optimizers to compare
    optimizers = ["sgd", "adam", "nested", "collaborative"]

    # Run experiments
    all_results = {}
    for opt_name in optimizers:
        results = run_experiment(opt_name, config, device)
        all_results[opt_name] = results

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(
        f"\n{'Optimizer':<25} | {'Avg Acc':>8} | {'Forgetting':>10} | {'BWT':>8} | {'Bridges':>8}"
    )
    print("-" * 80)

    for opt_name, results in all_results.items():
        metrics = results["metrics"]
        bridges = results.get("total_bridge_transfers", 0)
        bridge_str = str(bridges) if bridges > 0 else "N/A"

        print(
            f"{opt_name:<25} | {metrics['average_accuracy']:>8.4f} | "
            f"{metrics['forgetting']:>10.4f} | {metrics['backward_transfer']:>8.4f} | "
            f"{bridge_str:>8}"
        )

    print("-" * 80)

    # Save results
    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.results_dir, f"ablation_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Analyze hypothesis
    print("\n" + "=" * 80)
    print("HYPOTHESIS ANALYSIS")
    print("=" * 80)

    nested_forgetting = all_results["nested"]["metrics"]["forgetting"]
    collab_forgetting = all_results["collaborative"]["metrics"]["forgetting"]
    nested_acc = all_results["nested"]["metrics"]["average_accuracy"]
    collab_acc = all_results["collaborative"]["metrics"]["average_accuracy"]

    print("\nNested Optimizer:")
    print(f"  Forgetting: {nested_forgetting:.4f}")
    print(f"  Accuracy: {nested_acc:.4f}")

    print("\nCollaborative Nested Optimizer (ours):")
    print(f"  Forgetting: {collab_forgetting:.4f}")
    print(f"  Accuracy: {collab_acc:.4f}")

    forgetting_improvement = nested_forgetting - collab_forgetting
    acc_improvement = collab_acc - nested_acc

    print("\nImprovement:")
    print(
        f"  Forgetting reduction: {forgetting_improvement:.4f} ({forgetting_improvement/nested_forgetting*100:.1f}%)"
    )
    print(
        f"  Accuracy improvement: {acc_improvement:.4f} ({acc_improvement/nested_acc*100:.1f}%)"
    )

    if collab_forgetting < nested_forgetting:
        print("\n✓ HYPOTHESIS SUPPORTED: Collaborative optimizer has LOWER forgetting!")
    elif collab_acc > nested_acc:
        print(
            "\n✓ HYPOTHESIS PARTIALLY SUPPORTED: Collaborative optimizer has HIGHER accuracy!"
        )
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED: Need to investigate further.")

    return all_results


if __name__ == "__main__":
    # Run with default config
    config = ExperimentConfig(
        steps_per_task=500,
        seed=42,
    )

    results = run_ablation(config)
