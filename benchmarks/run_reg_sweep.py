"""Regularization Strength Sweep for CMS.

Tests different regularization strengths to find optimal forgetting prevention.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.metrics import ContinualMetrics
from benchmarks.split_mnist import SplitMNIST
from src.memory.continuum import CMSConfig
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer


@dataclass
class SweepConfig:
    """Configuration for regularization sweep."""

    # Model
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10

    # Training
    num_epochs: int = 3
    batch_size: int = 64

    # Optimizer
    learning_rate: float = 0.01
    optimizer_hidden_dim: int = 64

    # Sweep
    reg_strengths: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

    # Experiment
    num_tasks: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleMLP(nn.Module):
    """Simple MLP for Split-MNIST."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_epoch(
    model: nn.Module,
    optimizer: CollaborativeCMSOptimizer,
    train_loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (task_loss, reg_loss)."""
    model.train()
    total_task_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        task_loss = F.cross_entropy(output, y)

        # Get regularization loss
        reg_loss = optimizer.get_regularization_loss()
        total_loss = task_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        total_task_loss += task_loss.item()
        total_reg_loss += reg_loss.item()
        num_batches += 1

    return total_task_loss / num_batches, total_reg_loss / num_batches


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def run_single_strength(
    config: SweepConfig,
    reg_strength: float,
) -> Dict[str, Any]:
    """Run experiment with a single regularization strength."""
    print(f"\n  Testing reg_strength = {reg_strength}")

    # Set seed
    torch.manual_seed(config.seed)

    # Create model
    model = SimpleMLP(
        config.input_dim,
        config.hidden_dim,
        config.output_dim,
    ).to(config.device)

    # Create CMS config with specified strength
    cms_config = CMSConfig(
        fast_frequency=1,
        medium_frequency=10,
        slow_frequency=100,
        regularization_strength=reg_strength,
    )

    # Create optimizer
    optimizer = CollaborativeCMSOptimizer(
        model.parameters(),
        fast_lr=config.learning_rate,
        medium_lr=config.learning_rate * 0.5,
        slow_lr=config.learning_rate * 0.1,
        hidden_dim=config.optimizer_hidden_dim,
        bridge_threshold=0.3,
        cms_config=cms_config,
        use_cms_regularization=True,
    )

    # Create dataset
    benchmark = SplitMNIST(
        root="./data",
        num_tasks=config.num_tasks,
        batch_size=config.batch_size,
    )

    # Accuracy matrix
    accuracy_matrix = []

    # Training loop
    for task_id in range(config.num_tasks):
        train_loader, test_loader = benchmark.get_task(task_id)

        # Signal task switch
        optimizer.set_task(task_id)

        # Train on current task
        for epoch in range(config.num_epochs):
            task_loss, reg_loss = train_epoch(
                model, optimizer, train_loader, config.device
            )

        print(f"    Task {task_id}: task_loss={task_loss:.4f}, reg_loss={reg_loss:.4f}")

        # Evaluate on all tasks
        task_accuracies = []
        for eval_task_id in range(config.num_tasks):
            _, eval_test_loader = benchmark.get_task(eval_task_id)
            acc = evaluate(model, eval_test_loader, config.device)
            task_accuracies.append(acc)

        accuracy_matrix.append(task_accuracies)

    # Compute metrics
    num_tasks = len(accuracy_matrix)
    metrics = ContinualMetrics(num_tasks)
    for task_id, task_accs in enumerate(accuracy_matrix):
        metrics.update(task_id, np.array(task_accs))

    return {
        "reg_strength": reg_strength,
        "accuracy_matrix": accuracy_matrix,
        "average_accuracy": metrics.average_accuracy,
        "forgetting": metrics.forgetting,
        "backward_transfer": metrics.backward_transfer,
    }


def run_regularization_sweep(config: Optional[SweepConfig] = None) -> Dict[str, Any]:
    """Run the full regularization strength sweep."""
    if config is None:
        config = SweepConfig()

    print("=" * 60)
    print("Regularization Strength Sweep")
    print("=" * 60)
    print(f"Testing strengths: {config.reg_strengths}")
    print()

    results = {}

    for strength in config.reg_strengths:
        result = run_single_strength(config, strength)
        results[str(strength)] = result

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strength':<12} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10}")
    print("-" * 44)

    best_forgetting = float("inf")
    best_strength = None

    for strength in config.reg_strengths:
        r = results[str(strength)]
        print(
            f"{strength:<12} {r['average_accuracy']:>10.4f} "
            f"{r['forgetting']:>12.4f} {r['backward_transfer']:>10.4f}"
        )

        if r["forgetting"] < best_forgetting:
            best_forgetting = r["forgetting"]
            best_strength = strength

    print(
        f"\nBest strength for forgetting: {best_strength} (forgetting={best_forgetting:.4f})"
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f"reg_sweep_{timestamp}.json")

    serializable_results = {
        "config": {
            "reg_strengths": list(config.reg_strengths),
            "num_epochs": config.num_epochs,
            "num_tasks": config.num_tasks,
            "seed": config.seed,
        },
        "timestamp": timestamp,
        "results": results,
        "best_strength": best_strength,
        "best_forgetting": best_forgetting,
    }

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    config = SweepConfig(
        num_epochs=3,
        num_tasks=5,
        reg_strengths=(0.01, 0.1, 1.0, 10.0),
    )
    run_regularization_sweep(config)
