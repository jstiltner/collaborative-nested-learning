"""
Adjacent-Only Bridge Ablation Study

Compares full bridges (6 directions) vs adjacent-only (4 directions)
across multiple regularization strengths.

Full bridges: fast↔medium, medium↔slow, fast↔slow (6 directions)
Adjacent-only: fast↔medium, medium↔slow (4 directions)

Hypothesis: Adjacent-only may reduce forgetting by preventing noise
propagation from fast directly to slow memory.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.split_mnist import SplitMNIST
from src.memory.continuum import CMSConfig
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer


def create_model():
    """Create a simple MLP for Split-MNIST."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def run_single_experiment(
    adjacent_only: bool,
    reg_strength: float,
    num_tasks: int = 5,
    epochs_per_task: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single experiment with specified configuration."""
    torch.manual_seed(seed)

    # Create model and benchmark
    model = create_model()
    benchmark = SplitMNIST(root="./data", num_tasks=num_tasks, batch_size=64)

    # Create optimizer with specified adjacent_only setting
    cms_config = CMSConfig(
        regularization_strength=reg_strength,
        fast_frequency=1,
        medium_frequency=10,
        slow_frequency=50,
    )

    optimizer = CollaborativeCMSOptimizer(
        model.parameters(),
        cms_config=cms_config,
        fast_lr=0.01,
        medium_lr=0.005,
        slow_lr=0.001,
        bridge_threshold=0.3,
        bridge_frequency=10,
        adjacent_only=adjacent_only,  # KEY PARAMETER
        hidden_dim=64,
    )

    # Track accuracy matrix: acc[i][j] = accuracy on task j after training on task i
    accuracy_matrix = [[0.0] * num_tasks for _ in range(num_tasks)]

    # Train on each task
    for task_id in range(num_tasks):
        train_loader, test_loader = benchmark.get_task(task_id)
        optimizer.set_task(task_id)

        # Train
        model.train()
        for epoch in range(epochs_per_task):
            for x, y in train_loader:
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output, y)
                reg_loss = optimizer.get_regularization_loss()
                total_loss = loss + reg_loss
                total_loss.backward()
                optimizer.step()

        # Evaluate on all tasks seen so far
        model.eval()
        with torch.no_grad():
            for eval_task_id in range(task_id + 1):
                _, eval_loader = benchmark.get_task(eval_task_id)
                correct = 0
                total = 0
                for x, y in eval_loader:
                    output = model(x)
                    pred = output.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)
                accuracy = correct / total
                accuracy_matrix[task_id][eval_task_id] = accuracy

    # Compute metrics
    # Final accuracy: average accuracy on all tasks after training on all
    final_accuracy = sum(accuracy_matrix[-1]) / num_tasks

    # Forgetting: average drop from max accuracy to final accuracy
    forgetting = 0.0
    for j in range(num_tasks - 1):
        max_acc = max(accuracy_matrix[i][j] for i in range(j, num_tasks))
        final_acc = accuracy_matrix[-1][j]
        forgetting += max_acc - final_acc
    forgetting /= num_tasks - 1

    return {
        "final_accuracy": final_accuracy,
        "forgetting": forgetting,
        "accuracy_matrix": accuracy_matrix,
        "bridge_stats": {k: v for k, v in optimizer.get_bridge_stats().items()},
    }


def run_adjacent_ablation():
    """Run full ablation comparing adjacent-only vs full bridges."""
    strengths = [1.0, 2.0, 5.0, 10.0]
    results = {}

    print("=" * 60)
    print("ADJACENT-ONLY BRIDGE ABLATION STUDY")
    print("=" * 60)
    print("\nComparing:")
    print("  - Full Bridges: fast↔medium, medium↔slow, fast↔slow (6 directions)")
    print("  - Adjacent-Only: fast↔medium, medium↔slow (4 directions)")
    print()

    for strength in strengths:
        print(f"\n--- Regularization Strength: {strength} ---")

        # Full bridges (baseline)
        print("  Running: Full Bridges (6 directions)...", end=" ", flush=True)
        full_result = run_single_experiment(
            adjacent_only=False,
            reg_strength=strength,
        )
        print(f"acc={full_result['final_accuracy']:.4f}")

        # Adjacent-only bridges
        print("  Running: Adjacent-Only Bridges (4 directions)...", end=" ", flush=True)
        adj_result = run_single_experiment(
            adjacent_only=True,
            reg_strength=strength,
        )
        print(f"acc={adj_result['final_accuracy']:.4f}")

        # Compute differences
        acc_diff = adj_result["final_accuracy"] - full_result["final_accuracy"]
        forget_diff = adj_result["forgetting"] - full_result["forgetting"]

        results[str(strength)] = {
            "full_bridges": {
                "final_accuracy": full_result["final_accuracy"],
                "forgetting": full_result["forgetting"],
            },
            "adjacent_only": {
                "final_accuracy": adj_result["final_accuracy"],
                "forgetting": adj_result["forgetting"],
            },
            "diff": {
                "accuracy": acc_diff,
                "forgetting": forget_diff,
            },
        }

        print(f"\n  Results at strength={strength}:")
        print(
            f"    Full Bridges:    acc={full_result['final_accuracy']:.4f}, "
            f"forget={full_result['forgetting']:.4f}"
        )
        print(
            f"    Adjacent-Only:   acc={adj_result['final_accuracy']:.4f}, "
            f"forget={adj_result['forgetting']:.4f}"
        )

        # Interpret results
        acc_better = (
            "✓ BETTER" if acc_diff > 0 else "✗ worse" if acc_diff < 0 else "= same"
        )
        forget_better = (
            "✓ BETTER"
            if forget_diff < 0
            else "✗ worse" if forget_diff > 0 else "= same"
        )

        print(f"    Adjacent-only accuracy:   {acc_diff:+.4f} ({acc_better})")
        print(f"    Adjacent-only forgetting: {forget_diff:+.4f} ({forget_better})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    acc_wins = sum(1 for s in strengths if results[str(s)]["diff"]["accuracy"] > 0)
    forget_wins = sum(1 for s in strengths if results[str(s)]["diff"]["forgetting"] < 0)

    print(f"\nAdjacent-only improves accuracy in {acc_wins}/{len(strengths)} settings")
    print(
        f"Adjacent-only reduces forgetting in {forget_wins}/{len(strengths)} settings"
    )

    # Detailed breakdown
    print("\n--- Detailed Results ---")
    print(
        f"{'Strength':<10} {'Full Acc':<12} {'Adj Acc':<12} {'Diff':<10} {'Full Fgt':<12} {'Adj Fgt':<12} {'Diff':<10}"
    )
    print("-" * 80)
    for strength in strengths:
        r = results[str(strength)]
        print(
            f"{strength:<10} "
            f"{r['full_bridges']['final_accuracy']:<12.4f} "
            f"{r['adjacent_only']['final_accuracy']:<12.4f} "
            f"{r['diff']['accuracy']:+<10.4f} "
            f"{r['full_bridges']['forgetting']:<12.4f} "
            f"{r['adjacent_only']['forgetting']:<12.4f} "
            f"{r['diff']['forgetting']:+<10.4f}"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"experiments/results/adjacent_ablation_{timestamp}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_adjacent_ablation()
