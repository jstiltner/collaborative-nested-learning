"""CMS Ablation Study: Compare optimizers with and without Continuum Memory System.

This benchmark compares:
1. SGD (baseline - no continual learning)
2. Collaborative (bridges only, no CMS)
3. Collaborative + CMS (full implementation)

The goal is to demonstrate that CMS reduces catastrophic forgetting.
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
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer


def create_split_mnist(
    num_tasks: int = 5,
    batch_size: int = 64,
) -> List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]:
    """Create Split-MNIST tasks.
    
    Args:
        num_tasks: Number of tasks.
        batch_size: Batch size.
    
    Returns:
        List of (train_loader, test_loader) tuples.
    """
    benchmark = SplitMNIST(
        root='./data',
        num_tasks=num_tasks,
        batch_size=batch_size,
    )
    
    tasks = []
    for task_id in range(num_tasks):
        train_loader, test_loader = benchmark.get_task(task_id)
        tasks.append((train_loader, test_loader))
    
    return tasks


@dataclass
class CMSExperimentConfig:
    """Configuration for CMS ablation experiment."""
    
    # Model
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    
    # Training
    num_epochs: int = 5
    batch_size: int = 64
    
    # Optimizer
    learning_rate: float = 0.01
    optimizer_hidden_dim: int = 64
    
    # CMS
    cms_regularization_strength: float = 0.01
    
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
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_epoch(
    model: nn.Module,
    optimizer: Any,
    train_loader: torch.utils.data.DataLoader,
    device: str,
    use_cms_reg: bool = False,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        if hasattr(optimizer, 'zero_grad'):
            optimizer.zero_grad()
        else:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        # Add CMS regularization if available
        if use_cms_reg and hasattr(optimizer, 'get_regularization_loss'):
            reg_loss = optimizer.get_regularization_loss()
            loss = loss + reg_loss
        
        loss.backward()
        
        if hasattr(optimizer, 'step'):
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


def run_experiment(
    config: CMSExperimentConfig,
    optimizer_type: str,
) -> Dict[str, Any]:
    """Run a single experiment with specified optimizer.
    
    Args:
        config: Experiment configuration.
        optimizer_type: One of "sgd", "collaborative", "collaborative_cms".
    
    Returns:
        Dictionary with results.
    """
    # Set seed
    torch.manual_seed(config.seed)
    
    # Create model
    model = SimpleMLP(
        config.input_dim,
        config.hidden_dim,
        config.output_dim,
    ).to(config.device)
    
    # Create optimizer
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        use_cms_reg = False
    elif optimizer_type == "collaborative":
        optimizer = CollaborativeNestedOptimizer(
            model.parameters(),
            fast_lr=config.learning_rate,
            medium_lr=config.learning_rate * 0.5,
            slow_lr=config.learning_rate * 0.1,
            hidden_dim=config.optimizer_hidden_dim,
            bridge_threshold=0.3,
        )
        use_cms_reg = False
    elif optimizer_type == "collaborative_cms":
        optimizer = CollaborativeCMSOptimizer(
            model.parameters(),
            fast_lr=config.learning_rate,
            medium_lr=config.learning_rate * 0.5,
            slow_lr=config.learning_rate * 0.1,
            hidden_dim=config.optimizer_hidden_dim,
            bridge_threshold=0.3,
            use_cms_regularization=True,
        )
        use_cms_reg = True
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Create dataset
    tasks = create_split_mnist(
        num_tasks=config.num_tasks,
        batch_size=config.batch_size,
    )
    
    # Accuracy matrix: acc[i][j] = accuracy on task j after training on task i
    accuracy_matrix = []
    
    # Training loop
    for task_id, (train_loader, test_loader) in enumerate(tasks):
        print(f"  Training on task {task_id}...")
        
        # Signal task switch if supported
        if hasattr(optimizer, 'set_task'):
            optimizer.set_task(task_id)
        
        # Train on current task
        for epoch in range(config.num_epochs):
            loss = train_epoch(model, optimizer, train_loader, config.device, use_cms_reg)
            if epoch == config.num_epochs - 1:
                print(f"    Epoch {epoch}: loss = {loss:.4f}")
        
        # Evaluate on all tasks seen so far
        task_accuracies = []
        for eval_task_id, (_, eval_test_loader) in enumerate(tasks):
            acc = evaluate(model, eval_test_loader, config.device)
            task_accuracies.append(acc)
        
        accuracy_matrix.append(task_accuracies)
        print(f"    Accuracies: {[f'{a:.3f}' for a in task_accuracies]}")
    
    # Compute metrics
    num_tasks = len(accuracy_matrix)
    metrics = ContinualMetrics(num_tasks)
    for task_id, task_accs in enumerate(accuracy_matrix):
        metrics.update(task_id, np.array(task_accs))
    
    # Get additional stats
    extra_stats = {}
    if hasattr(optimizer, 'get_bridge_stats'):
        extra_stats['bridge_stats'] = optimizer.get_bridge_stats()
    if hasattr(optimizer, 'get_memory_drift'):
        extra_stats['memory_drift'] = optimizer.get_memory_drift()
    
    return {
        'optimizer_type': optimizer_type,
        'accuracy_matrix': accuracy_matrix,
        'average_accuracy': metrics.average_accuracy,
        'forgetting': metrics.forgetting,
        'backward_transfer': metrics.backward_transfer,
        'extra_stats': extra_stats,
    }


def run_cms_ablation(config: Optional[CMSExperimentConfig] = None) -> Dict[str, Any]:
    """Run the full CMS ablation study.
    
    Compares SGD, Collaborative, and Collaborative+CMS.
    
    Args:
        config: Experiment configuration.
    
    Returns:
        Dictionary with all results.
    """
    if config is None:
        config = CMSExperimentConfig()
    
    print("=" * 60)
    print("CMS Ablation Study")
    print("=" * 60)
    print(f"Config: {asdict(config)}")
    print()
    
    results = {}
    
    # Run each optimizer
    for optimizer_type in ["sgd", "collaborative", "collaborative_cms"]:
        print(f"\n{'=' * 40}")
        print(f"Running: {optimizer_type}")
        print("=" * 40)
        
        result = run_experiment(config, optimizer_type)
        results[optimizer_type] = result
        
        print(f"\nResults for {optimizer_type}:")
        print(f"  Average Accuracy: {result['average_accuracy']:.4f}")
        print(f"  Forgetting: {result['forgetting']:.4f}")
        print(f"  Backward Transfer: {result['backward_transfer']:.4f}")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Optimizer':<20} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10}")
    print("-" * 52)
    for opt_type, result in results.items():
        print(f"{opt_type:<20} {result['average_accuracy']:>10.4f} "
              f"{result['forgetting']:>12.4f} {result['backward_transfer']:>10.4f}")
    
    # Compute improvements
    if 'sgd' in results and 'collaborative_cms' in results:
        sgd_acc = results['sgd']['average_accuracy']
        cms_acc = results['collaborative_cms']['average_accuracy']
        improvement = (cms_acc - sgd_acc) / sgd_acc * 100
        
        sgd_forget = results['sgd']['forgetting']
        cms_forget = results['collaborative_cms']['forgetting']
        forget_reduction = (sgd_forget - cms_forget) / sgd_forget * 100 if sgd_forget > 0 else 0
        
        print(f"\nCMS vs SGD:")
        print(f"  Accuracy improvement: {improvement:+.2f}%")
        print(f"  Forgetting reduction: {forget_reduction:+.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, f"cms_ablation_{timestamp}.json")
    
    # Convert to serializable format
    serializable_results = {
        'config': asdict(config),
        'timestamp': timestamp,
        'results': {},
    }
    
    for opt_type, result in results.items():
        serializable_results['results'][opt_type] = {
            'accuracy_matrix': result['accuracy_matrix'],
            'average_accuracy': result['average_accuracy'],
            'forgetting': result['forgetting'],
            'backward_transfer': result['backward_transfer'],
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Run with default config
    config = CMSExperimentConfig(
        num_epochs=3,  # Faster for testing
        num_tasks=5,
    )
    run_cms_ablation(config)