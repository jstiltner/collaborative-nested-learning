"""Metrics for Continual Learning Evaluation.

Implements standard continual learning metrics:
- Average Accuracy: Mean accuracy across all tasks after training
- Forgetting: How much accuracy drops on old tasks
- Forward Transfer: Zero-shot performance on future tasks
- Backward Transfer: Improvement on past tasks from learning new ones
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """Compute accuracy on a dataset.
    
    Args:
        model: PyTorch model.
        dataloader: DataLoader for evaluation.
        device: Device to use (defaults to model's device).
    
    Returns:
        Accuracy as a float in [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    correct = 0
    total = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        # Flatten if needed (for MNIST with MLP)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        outputs = model(x)
        _, predicted = outputs.max(1)
        
        correct += (predicted == y).sum().item()
        total += y.size(0)
    
    return correct / total if total > 0 else 0.0


def compute_accuracy_matrix(
    model: nn.Module,
    test_loaders: List[DataLoader],
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute accuracy on all tasks.
    
    Args:
        model: PyTorch model.
        test_loaders: List of test DataLoaders, one per task.
        device: Device to use.
    
    Returns:
        Array of accuracies, shape (num_tasks,).
    """
    accuracies = []
    for loader in test_loaders:
        acc = compute_accuracy(model, loader, device)
        accuracies.append(acc)
    return np.array(accuracies)


def compute_forgetting(accuracy_matrix: np.ndarray) -> float:
    """Compute average forgetting across tasks.
    
    Forgetting measures how much accuracy drops on previous tasks
    after learning new ones.
    
    Forgetting_j = max_{i <= j}(A_{i,j}) - A_{T,j}
    
    Where:
    - A_{i,j} is accuracy on task j after training on task i
    - T is the final task
    
    Args:
        accuracy_matrix: Matrix of shape (num_tasks, num_tasks) where
            accuracy_matrix[i, j] is accuracy on task j after training on task i.
    
    Returns:
        Average forgetting (higher = more forgetting = worse).
    """
    num_tasks = accuracy_matrix.shape[0]
    
    if num_tasks <= 1:
        return 0.0
    
    forgetting = 0.0
    count = 0
    
    for j in range(num_tasks - 1):  # For each task except the last
        # Maximum accuracy achieved on task j (up to and including when we trained on it)
        max_acc = np.max(accuracy_matrix[:j+1, j])
        # Final accuracy on task j
        final_acc = accuracy_matrix[-1, j]
        # Forgetting for this task
        forgetting += max(0, max_acc - final_acc)
        count += 1
    
    return forgetting / count if count > 0 else 0.0


def compute_forward_transfer(
    accuracy_matrix: np.ndarray,
    random_accuracy: float = 0.1,
) -> float:
    """Compute average forward transfer.
    
    Forward transfer measures zero-shot performance on future tasks
    before training on them.
    
    FWT_j = A_{j-1, j} - random_accuracy
    
    Args:
        accuracy_matrix: Matrix of shape (num_tasks, num_tasks).
        random_accuracy: Random baseline accuracy (0.1 for 10-class).
    
    Returns:
        Average forward transfer (higher = better).
    """
    num_tasks = accuracy_matrix.shape[0]
    
    if num_tasks <= 1:
        return 0.0
    
    transfer = 0.0
    count = 0
    
    for j in range(1, num_tasks):  # For each task except the first
        # Zero-shot accuracy on task j (before training on it)
        zero_shot = accuracy_matrix[j-1, j]
        transfer += zero_shot - random_accuracy
        count += 1
    
    return transfer / count if count > 0 else 0.0


def compute_backward_transfer(accuracy_matrix: np.ndarray) -> float:
    """Compute average backward transfer.
    
    Backward transfer measures improvement on past tasks from learning new ones.
    Positive = learning new tasks helps old ones.
    Negative = learning new tasks hurts old ones (forgetting).
    
    BWT_j = A_{T, j} - A_{j, j}
    
    Args:
        accuracy_matrix: Matrix of shape (num_tasks, num_tasks).
    
    Returns:
        Average backward transfer (higher = better, negative = forgetting).
    """
    num_tasks = accuracy_matrix.shape[0]
    
    if num_tasks <= 1:
        return 0.0
    
    transfer = 0.0
    count = 0
    
    for j in range(num_tasks - 1):  # For each task except the last
        # Accuracy right after training on task j
        initial_acc = accuracy_matrix[j, j]
        # Final accuracy after all training
        final_acc = accuracy_matrix[-1, j]
        transfer += final_acc - initial_acc
        count += 1
    
    return transfer / count if count > 0 else 0.0


def compute_average_accuracy(accuracy_matrix: np.ndarray) -> float:
    """Compute average accuracy after training on all tasks.
    
    Args:
        accuracy_matrix: Matrix of shape (num_tasks, num_tasks).
    
    Returns:
        Average accuracy across all tasks.
    """
    # Final row contains accuracies after training on all tasks
    return float(np.mean(accuracy_matrix[-1, :]))


class ContinualMetrics:
    """Container for continual learning metrics.
    
    Tracks accuracy matrix and computes all metrics.
    
    Example:
        >>> metrics = ContinualMetrics(num_tasks=5)
        >>> for task_id in range(5):
        ...     # Train on task
        ...     accuracies = evaluate_all_tasks(model, test_loaders)
        ...     metrics.update(task_id, accuracies)
        >>> print(metrics.summary())
    """
    
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        self._current_task = 0
    
    def update(self, task_id: int, accuracies: np.ndarray) -> None:
        """Update accuracy matrix after training on a task.
        
        Args:
            task_id: Task that was just trained on.
            accuracies: Accuracies on all tasks, shape (num_tasks,).
        """
        self.accuracy_matrix[task_id, :] = accuracies
        self._current_task = task_id + 1
    
    @property
    def average_accuracy(self) -> float:
        """Average accuracy after training on all tasks."""
        return compute_average_accuracy(self.accuracy_matrix)
    
    @property
    def forgetting(self) -> float:
        """Average forgetting."""
        return compute_forgetting(self.accuracy_matrix)
    
    @property
    def forward_transfer(self) -> float:
        """Average forward transfer."""
        return compute_forward_transfer(self.accuracy_matrix)
    
    @property
    def backward_transfer(self) -> float:
        """Average backward transfer."""
        return compute_backward_transfer(self.accuracy_matrix)
    
    def summary(self) -> dict:
        """Get summary of all metrics.
        
        Returns:
            Dictionary with all metrics.
        """
        return {
            'average_accuracy': self.average_accuracy,
            'forgetting': self.forgetting,
            'forward_transfer': self.forward_transfer,
            'backward_transfer': self.backward_transfer,
            'accuracy_matrix': self.accuracy_matrix.tolist(),
        }
    
    def __repr__(self) -> str:
        return (
            f"ContinualMetrics(\n"
            f"  avg_accuracy={self.average_accuracy:.4f},\n"
            f"  forgetting={self.forgetting:.4f},\n"
            f"  forward_transfer={self.forward_transfer:.4f},\n"
            f"  backward_transfer={self.backward_transfer:.4f}\n"
            f")"
        )


def _test_metrics():
    """Test the metrics functions."""
    print("Testing continual learning metrics...")
    
    # Create a sample accuracy matrix
    # Rows = after training on task i
    # Cols = accuracy on task j
    accuracy_matrix = np.array([
        [0.95, 0.10, 0.10, 0.10, 0.10],  # After task 0
        [0.85, 0.93, 0.12, 0.11, 0.10],  # After task 1
        [0.75, 0.82, 0.94, 0.13, 0.11],  # After task 2
        [0.65, 0.72, 0.85, 0.92, 0.12],  # After task 3
        [0.55, 0.62, 0.75, 0.83, 0.91],  # After task 4
    ])
    
    avg_acc = compute_average_accuracy(accuracy_matrix)
    forgetting = compute_forgetting(accuracy_matrix)
    fwt = compute_forward_transfer(accuracy_matrix)
    bwt = compute_backward_transfer(accuracy_matrix)
    
    print(f"  Average Accuracy: {avg_acc:.4f}")
    print(f"  Forgetting: {forgetting:.4f}")
    print(f"  Forward Transfer: {fwt:.4f}")
    print(f"  Backward Transfer: {bwt:.4f}")
    
    # Test ContinualMetrics class
    metrics = ContinualMetrics(num_tasks=5)
    for i in range(5):
        metrics.update(i, accuracy_matrix[i])
    
    print(f"\n  ContinualMetrics:\n{metrics}")
    
    print("✓ Metrics test passed!")


if __name__ == "__main__":
    _test_metrics()