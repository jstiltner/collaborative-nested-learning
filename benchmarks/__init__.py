"""Benchmarks for Collaborative Nested Learning.

This module provides:
- Split-MNIST: Sequential task learning benchmark
- Metrics: Forgetting, accuracy, forward/backward transfer
"""

from benchmarks.metrics import (
    compute_accuracy,
    compute_backward_transfer,
    compute_forgetting,
    compute_forward_transfer,
)
from benchmarks.split_mnist import SplitMNIST, create_split_mnist_loaders

__all__ = [
    "SplitMNIST",
    "create_split_mnist_loaders",
    "compute_accuracy",
    "compute_forgetting",
    "compute_forward_transfer",
    "compute_backward_transfer",
]
