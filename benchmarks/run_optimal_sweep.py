"""Fine-grained Regularization Sweep + Bridge Ablation.

This script:
1. Tests intermediate strengths (2.0, 5.0, 7.5, 15.0, 20.0) to find optimal
2. Runs bridge ablation at the optimal strength
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.metrics import ContinualMetrics
from benchmarks.split_mnist import SplitMNIST
from src.bridges.knowledge_bridges import CollaborativeNestedOptimizer
from src.memory.continuum import CMSConfig
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer


@dataclass
class OptimalSweepConfig:
    """Configuration for optimal strength sweep."""

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

    # Sweep - finer granularity around 10.0
    reg_strengths: Tuple[float, ...] = (2.0, 5.0, 7.5, 10.0, 15.0, 20.0)

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


def train_epoch_cms(
    model: nn.Module,
    optimizer: CollaborativeCMSOptimizer,
    train_loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[float, float]:
    """Train for one epoch with CMS. Returns (task_loss, reg_loss)."""
    model.train()
    total_task_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        task_loss = F.cross_entropy(output, y)

        reg_loss = optimizer.get_regularization_loss()
        total_loss = task_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        total_task_loss += task_loss.item()
        total_reg_loss += reg_loss.item()
        num_batches += 1

    return total_task_loss / num_batches, total_reg_loss / num_batches


def train_epoch_no_cms(
    model: nn.Module,
    optimizer: CollaborativeNestedOptimizer,
    train_loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    """Train for one epoch without CMS. Returns task_loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = F.cross_entropy(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


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


def run_with_cms(
    config: OptimalSweepConfig,
    reg_strength: float,
    enable_bridges: bool = True,
) -> Dict[str, Any]:
    """Run experiment with CMS at specified strength."""
    torch.manual_seed(config.seed)

    model = SimpleMLP(
        config.input_dim,
        config.hidden_dim,
        config.output_dim,
    ).to(config.device)

    cms_config = CMSConfig(
        fast_frequency=1,
        medium_frequency=10,
        slow_frequency=100,
        regularization_strength=reg_strength,
    )

    optimizer = CollaborativeCMSOptimizer(
        model.parameters(),
        fast_lr=config.learning_rate,
        medium_lr=config.learning_rate * 0.5,
        slow_lr=config.learning_rate * 0.1,
        hidden_dim=config.optimizer_hidden_dim,
        bridge_threshold=0.3,
        cms_config=cms_config,
        use_cms_regularization=True,
        enable_reverse_bridges=enable_bridges,
    )

    benchmark = SplitMNIST(
        root="./data",
        num_tasks=config.num_tasks,
        batch_size=config.batch_size,
    )

    accuracy_matrix = []
    bridge_transfers = 0

    for task_id in range(config.num_tasks):
        train_loader, test_loader = benchmark.get_task(task_id)
        optimizer.set_task(task_id)

        for epoch in range(config.num_epochs):
            task_loss, reg_loss = train_epoch_cms(
                model, optimizer, train_loader, config.device
            )

        # Count bridge transfers
        if hasattr(optimizer, "get_bridge_stats"):
            stats = optimizer.get_bridge_stats()
            for bridge_stats in stats.values():
                bridge_transfers += bridge_stats.get("total_transfers", 0)

        task_accuracies = []
        for eval_task_id in range(config.num_tasks):
            _, eval_test_loader = benchmark.get_task(eval_task_id)
            acc = evaluate(model, eval_test_loader, config.device)
            task_accuracies.append(acc)

        accuracy_matrix.append(task_accuracies)

    num_tasks = len(accuracy_matrix)
    metrics = ContinualMetrics(num_tasks)
    for task_id, task_accs in enumerate(accuracy_matrix):
        metrics.update(task_id, np.array(task_accs))

    return {
        "reg_strength": reg_strength,
        "enable_bridges": enable_bridges,
        "accuracy_matrix": accuracy_matrix,
        "average_accuracy": metrics.average_accuracy,
        "forgetting": metrics.forgetting,
        "backward_transfer": metrics.backward_transfer,
        "bridge_transfers": bridge_transfers,
    }


def run_optimal_sweep(config: Optional[OptimalSweepConfig] = None) -> Dict[str, Any]:
    """Run the optimal strength sweep and bridge ablation."""
    if config is None:
        config = OptimalSweepConfig()

    print("=" * 60)
    print("Optimal Strength Sweep + Bridge Ablation")
    print("=" * 60)
    print(f"Testing strengths: {config.reg_strengths}")
    print()

    # Phase 1: Find optimal strength
    print("PHASE 1: Finding Optimal Regularization Strength")
    print("-" * 60)

    strength_results = {}

    for strength in config.reg_strengths:
        print(f"\n  Testing strength = {strength}")
        result = run_with_cms(config, strength, enable_bridges=True)
        strength_results[str(strength)] = result
        print(
            f"    Accuracy: {result['average_accuracy']:.4f}, "
            f"Forgetting: {result['forgetting']:.4f}"
        )

    # Find best strength (balance accuracy and forgetting)
    # Use a simple score: accuracy - 0.5 * forgetting
    best_score = -float("inf")
    best_strength = None

    print("\n  Strength Scores (accuracy - 0.5 * forgetting):")
    for strength in config.reg_strengths:
        r = strength_results[str(strength)]
        score = r["average_accuracy"] - 0.5 * r["forgetting"]
        print(f"    {strength}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_strength = strength

    print(f"\n  Best strength: {best_strength} (score={best_score:.4f})")

    # Phase 2: Bridge ablation at optimal strength
    print("\n" + "=" * 60)
    print(f"PHASE 2: Bridge Ablation at strength={best_strength}")
    print("-" * 60)

    print("\n  Running CMS + Bridges...")
    with_bridges = run_with_cms(config, best_strength, enable_bridges=True)

    print("\n  Running CMS only (no bridges)...")
    without_bridges = run_with_cms(config, best_strength, enable_bridges=False)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\nStrength Sweep Results:")
    print(f"{'Strength':<12} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10}")
    print("-" * 44)
    for strength in config.reg_strengths:
        r = strength_results[str(strength)]
        print(
            f"{strength:<12} {r['average_accuracy']:>10.4f} "
            f"{r['forgetting']:>12.4f} {r['backward_transfer']:>10.4f}"
        )

    print(f"\nOptimal Strength: {best_strength}")

    print("\nBridge Ablation Results:")
    print(f"{'Configuration':<20} {'Avg Acc':>10} {'Forgetting':>12} {'Bridges':>10}")
    print("-" * 52)
    print(
        f"{'CMS + Bridges':<20} {with_bridges['average_accuracy']:>10.4f} "
        f"{with_bridges['forgetting']:>12.4f} {with_bridges['bridge_transfers']:>10}"
    )
    print(
        f"{'CMS only':<20} {without_bridges['average_accuracy']:>10.4f} "
        f"{without_bridges['forgetting']:>12.4f} {without_bridges['bridge_transfers']:>10}"
    )

    # Compute bridge contribution
    acc_diff = with_bridges["average_accuracy"] - without_bridges["average_accuracy"]
    forget_diff = without_bridges["forgetting"] - with_bridges["forgetting"]

    print(f"\nBridge Contribution:")
    print(
        f"  Accuracy: {acc_diff:+.4f} ({acc_diff/without_bridges['average_accuracy']*100:+.2f}%)"
    )
    print(
        f"  Forgetting reduction: {forget_diff:+.4f} ({forget_diff/without_bridges['forgetting']*100:+.2f}%)"
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f"optimal_sweep_{timestamp}.json")

    all_results = {
        "config": {
            "reg_strengths": list(config.reg_strengths),
            "num_epochs": config.num_epochs,
            "num_tasks": config.num_tasks,
            "seed": config.seed,
        },
        "timestamp": timestamp,
        "strength_sweep": strength_results,
        "best_strength": best_strength,
        "bridge_ablation": {
            "with_bridges": with_bridges,
            "without_bridges": without_bridges,
            "accuracy_diff": acc_diff,
            "forgetting_diff": forget_diff,
        },
    }

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    config = OptimalSweepConfig(
        num_epochs=3,
        num_tasks=5,
        reg_strengths=(2.0, 5.0, 7.5, 10.0, 15.0, 20.0),
    )
    run_optimal_sweep(config)
