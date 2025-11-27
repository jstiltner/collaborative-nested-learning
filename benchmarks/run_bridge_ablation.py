"""Clean Bridge Ablation: Test our NOVEL contribution (bidirectional bridges).

This benchmark tests bridges at multiple CMS regularization strengths to see
if they provide consistent benefit across the stability-plasticity spectrum.

Comparisons:
1. CMS only (no bridges) at each strength
2. CMS + Bridges at each strength

This isolates the bridge contribution from the CMS contribution.
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
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer
from src.memory.continuum import CMSConfig


@dataclass
class BridgeAblationConfig:
    """Configuration for bridge ablation."""
    
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
    
    # Strengths to test
    reg_strengths: Tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)
    
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


def run_experiment(
    config: BridgeAblationConfig,
    reg_strength: float,
    enable_bridges: bool,
) -> Dict[str, Any]:
    """Run a single experiment."""
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
        bridge_frequency=10,
        cms_config=cms_config,
        use_cms_regularization=True,
        enable_reverse_bridges=enable_bridges,
    )
    
    benchmark = SplitMNIST(
        root='./data',
        num_tasks=config.num_tasks,
        batch_size=config.batch_size,
    )
    
    accuracy_matrix = []
    total_bridge_transfers = 0
    
    for task_id in range(config.num_tasks):
        train_loader, test_loader = benchmark.get_task(task_id)
        optimizer.set_task(task_id)
        
        for epoch in range(config.num_epochs):
            task_loss, reg_loss = train_epoch(model, optimizer, train_loader, config.device)
        
        # Count bridge transfers
        if enable_bridges:
            bridge_stats = optimizer.get_bridge_stats()
            for bridge_name, stats in bridge_stats.items():
                total_bridge_transfers += stats.get('total_transfers', 0)
        
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
        'reg_strength': reg_strength,
        'enable_bridges': enable_bridges,
        'accuracy_matrix': accuracy_matrix,
        'average_accuracy': metrics.average_accuracy,
        'forgetting': metrics.forgetting,
        'backward_transfer': metrics.backward_transfer,
        'bridge_transfers': total_bridge_transfers,
    }


def run_bridge_ablation(config: Optional[BridgeAblationConfig] = None) -> Dict[str, Any]:
    """Run bridge ablation at multiple strengths."""
    if config is None:
        config = BridgeAblationConfig()
    
    print("=" * 70)
    print("BRIDGE ABLATION ACROSS REGULARIZATION STRENGTHS")
    print("Testing our NOVEL contribution: Bidirectional Bridges")
    print("=" * 70)
    print(f"\nStrengths to test: {config.reg_strengths}")
    print("For each strength, comparing: CMS only vs CMS + Bridges")
    print()
    
    results = {}
    
    for strength in config.reg_strengths:
        print(f"\n{'='*50}")
        print(f"Testing strength = {strength}")
        print("=" * 50)
        
        # CMS only (no bridges)
        print(f"  Running CMS only...")
        cms_only = run_experiment(config, strength, enable_bridges=False)
        
        # CMS + Bridges
        print(f"  Running CMS + Bridges...")
        cms_bridges = run_experiment(config, strength, enable_bridges=True)
        
        results[str(strength)] = {
            'cms_only': cms_only,
            'cms_bridges': cms_bridges,
        }
        
        # Quick comparison
        acc_diff = cms_bridges['average_accuracy'] - cms_only['average_accuracy']
        forget_diff = cms_only['forgetting'] - cms_bridges['forgetting']
        
        print(f"\n  Results at strength={strength}:")
        print(f"    CMS only:    acc={cms_only['average_accuracy']:.4f}, forget={cms_only['forgetting']:.4f}")
        print(f"    CMS+Bridges: acc={cms_bridges['average_accuracy']:.4f}, forget={cms_bridges['forgetting']:.4f}")
        print(f"    Bridge contribution: acc={acc_diff:+.4f}, forget_reduction={forget_diff:+.4f}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: BRIDGE CONTRIBUTION AT EACH STRENGTH")
    print("=" * 70)
    print(f"\n{'Strength':<10} {'CMS Acc':>10} {'CMS+B Acc':>12} {'Acc Diff':>10} | "
          f"{'CMS Fgt':>10} {'CMS+B Fgt':>12} {'Fgt Diff':>10}")
    print("-" * 80)
    
    bridge_helps_accuracy = 0
    bridge_helps_forgetting = 0
    
    for strength in config.reg_strengths:
        r = results[str(strength)]
        cms = r['cms_only']
        bridges = r['cms_bridges']
        
        acc_diff = bridges['average_accuracy'] - cms['average_accuracy']
        forget_diff = cms['forgetting'] - bridges['forgetting']
        
        if acc_diff > 0:
            bridge_helps_accuracy += 1
        if forget_diff > 0:
            bridge_helps_forgetting += 1
        
        print(f"{strength:<10} {cms['average_accuracy']:>10.4f} {bridges['average_accuracy']:>12.4f} "
              f"{acc_diff:>+10.4f} | {cms['forgetting']:>10.4f} {bridges['forgetting']:>12.4f} "
              f"{forget_diff:>+10.4f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\nBridges improve accuracy in {bridge_helps_accuracy}/{len(config.reg_strengths)} settings")
    print(f"Bridges reduce forgetting in {bridge_helps_forgetting}/{len(config.reg_strengths)} settings")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, f"bridge_ablation_{timestamp}.json")
    
    all_results = {
        'config': {
            'reg_strengths': list(config.reg_strengths),
            'num_epochs': config.num_epochs,
            'num_tasks': config.num_tasks,
            'seed': config.seed,
        },
        'timestamp': timestamp,
        'results': results,
        'summary': {
            'bridge_helps_accuracy': bridge_helps_accuracy,
            'bridge_helps_forgetting': bridge_helps_forgetting,
            'total_settings': len(config.reg_strengths),
        },
    }
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    config = BridgeAblationConfig(
        num_epochs=3,
        num_tasks=5,
        reg_strengths=(0.1, 1.0, 5.0, 10.0),
    )
    run_bridge_ablation(config)