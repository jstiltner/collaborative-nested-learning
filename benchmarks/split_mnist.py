"""Split-MNIST Benchmark for Continual Learning.

Splits MNIST into 5 sequential tasks:
- Task 0: Digits 0-1
- Task 1: Digits 2-3
- Task 2: Digits 4-5
- Task 3: Digits 6-7
- Task 4: Digits 8-9

This is a standard benchmark for evaluating catastrophic forgetting.
"""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class SplitMNIST:
    """Split-MNIST benchmark dataset.

    Splits MNIST into sequential tasks, each containing 2 digit classes.

    Args:
        root: Root directory for MNIST data.
        num_tasks: Number of tasks (default 5 for 10 digits).
        batch_size: Batch size for data loaders.
        download: Whether to download MNIST if not present.

    Example:
        >>> benchmark = SplitMNIST(root='./data', batch_size=64)
        >>> for task_id in range(benchmark.num_tasks):
        ...     train_loader, test_loader = benchmark.get_task(task_id)
        ...     # Train on this task
        ...     for x, y in train_loader:
        ...         ...
    """

    def __init__(
        self,
        root: str = "./data",
        num_tasks: int = 5,
        batch_size: int = 64,
        download: bool = True,
    ):
        self.root = root
        self.num_tasks = num_tasks
        self.batch_size = batch_size

        # Standard MNIST transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Load full MNIST
        self.train_dataset = datasets.MNIST(
            root=root,
            train=True,
            download=download,
            transform=self.transform,
        )

        self.test_dataset = datasets.MNIST(
            root=root,
            train=False,
            download=download,
            transform=self.transform,
        )

        # Compute classes per task
        self.classes_per_task = 10 // num_tasks

        # Pre-compute task indices
        self._train_indices = self._compute_task_indices(self.train_dataset)
        self._test_indices = self._compute_task_indices(self.test_dataset)

    def _compute_task_indices(self, dataset: Dataset) -> List[List[int]]:
        """Compute indices for each task.

        Args:
            dataset: MNIST dataset.

        Returns:
            List of index lists, one per task.
        """
        task_indices = [[] for _ in range(self.num_tasks)]

        for idx, (_, label) in enumerate(dataset):
            task_id = label // self.classes_per_task
            if task_id < self.num_tasks:
                task_indices[task_id].append(idx)

        return task_indices

    def get_task(self, task_id: int) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for a specific task.

        Args:
            task_id: Task index (0 to num_tasks-1).

        Returns:
            Tuple of (train_loader, test_loader) for the task.
        """
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(
                f"task_id must be in [0, {self.num_tasks-1}], got {task_id}"
            )

        train_subset = Subset(self.train_dataset, self._train_indices[task_id])
        test_subset = Subset(self.test_dataset, self._test_indices[task_id])

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, test_loader

    def get_all_test_loaders(self) -> List[DataLoader]:
        """Get test loaders for all tasks.

        Returns:
            List of test loaders, one per task.
        """
        return [self.get_task(i)[1] for i in range(self.num_tasks)]

    def get_task_classes(self, task_id: int) -> List[int]:
        """Get the digit classes for a task.

        Args:
            task_id: Task index.

        Returns:
            List of digit classes in this task.
        """
        start = task_id * self.classes_per_task
        end = start + self.classes_per_task
        return list(range(start, end))


def create_split_mnist_loaders(
    root: str = "./data",
    num_tasks: int = 5,
    batch_size: int = 64,
    download: bool = True,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Create train and test loaders for all Split-MNIST tasks.

    Args:
        root: Root directory for MNIST data.
        num_tasks: Number of tasks.
        batch_size: Batch size.
        download: Whether to download MNIST.

    Returns:
        Tuple of (train_loaders, test_loaders), each a list of DataLoaders.
    """
    benchmark = SplitMNIST(
        root=root,
        num_tasks=num_tasks,
        batch_size=batch_size,
        download=download,
    )

    train_loaders = []
    test_loaders = []

    for task_id in range(num_tasks):
        train_loader, test_loader = benchmark.get_task(task_id)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


def _test_split_mnist():
    """Test the Split-MNIST benchmark."""
    print("Testing Split-MNIST benchmark...")

    benchmark = SplitMNIST(root="./data", batch_size=32)

    print(f"  Number of tasks: {benchmark.num_tasks}")
    print(f"  Classes per task: {benchmark.classes_per_task}")

    for task_id in range(benchmark.num_tasks):
        train_loader, test_loader = benchmark.get_task(task_id)
        classes = benchmark.get_task_classes(task_id)

        print(
            f"  Task {task_id}: classes {classes}, "
            f"train batches: {len(train_loader)}, "
            f"test batches: {len(test_loader)}"
        )

        # Verify classes are correct
        for x, y in train_loader:
            assert all(
                label in classes for label in y.tolist()
            ), f"Found wrong classes in task {task_id}"
            break

    print("✓ Split-MNIST test passed!")


if __name__ == "__main__":
    _test_split_mnist()
