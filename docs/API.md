# API Reference

## Optimizers

### CollaborativeCMSOptimizer

The main optimizer implementing multi-timescale learning with knowledge bridges.

```python
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer

optimizer = CollaborativeCMSOptimizer(
    params,                      # Model parameters
    lr=0.01,                     # Learning rate
    hidden_dim=64,               # Hidden dimension for memory networks
    fast_freq=1,                 # Fast memory update frequency
    medium_freq=10,              # Medium memory update frequency  
    slow_freq=50,                # Slow memory update frequency
    regularization_strength=1.0, # CMS regularization strength
    enable_bridges=True,         # Enable knowledge bridges
    bridge_threshold=0.3,        # Threshold for bridge activation
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | required | Model parameters to optimize |
| `lr` | float | 0.01 | Learning rate |
| `hidden_dim` | int | 64 | Hidden dimension for memory networks |
| `fast_freq` | int | 1 | Steps between fast memory updates |
| `medium_freq` | int | 10 | Steps between medium memory updates |
| `slow_freq` | int | 50 | Steps between slow memory updates |
| `regularization_strength` | float | 1.0 | Strength of CMS regularization (higher = more retention) |
| `enable_bridges` | bool | True | Whether to enable knowledge bridges |
| `bridge_threshold` | float | 0.3 | Minimum gate value for bridge transfer |

#### Methods

##### `step(closure=None)`
Performs a single optimization step.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

##### `get_bridge_stats()`
Returns statistics about bridge activations.

```python
stats = optimizer.get_bridge_stats()
# Returns dict with transfer counts and gate values per bridge
```

---

### DeepMomentumOptimizer

Neural network-based momentum optimizer that learns to combine gradient history.

```python
from src.optimizers.deep_momentum import DeepMomentumOptimizer

optimizer = DeepMomentumOptimizer(
    params,
    lr=0.01,
    memory_size=10,    # Number of gradients to remember
    hidden_dim=32,     # Hidden dimension of momentum network
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | required | Model parameters to optimize |
| `lr` | float | 0.01 | Learning rate |
| `memory_size` | int | 10 | Number of past gradients to store |
| `hidden_dim` | int | 32 | Hidden dimension of momentum network |

---

### NestedOptimizer

Multi-timescale optimizer without bridges (baseline).

```python
from src.optimizers.nested_optimizer import NestedOptimizer

optimizer = NestedOptimizer(
    params,
    lr=0.01,
    fast_freq=1,
    medium_freq=10,
    slow_freq=50,
)
```

---

## Memory Systems

### MemoryBank

Stores gradient history for a single timescale.

```python
from src.memory.memory_bank import MemoryBank

bank = MemoryBank(
    param_dim=1000,     # Total number of parameters
    memory_size=10,     # Number of gradients to store
    hidden_dim=32,      # Hidden dimension for processing
)
```

### ContinuumMemorySystem

Implements importance-weighted regularization for continual learning.

```python
from src.memory.continuum import ContinuumMemorySystem

cms = ContinuumMemorySystem(
    param_shapes=[(256, 784), (10, 256)],  # Shapes of model parameters
    regularization_strength=1.0,
)

# After training on a task
cms.consolidate(model.parameters())

# Get regularization loss
reg_loss = cms.compute_regularization(model.parameters())
```

---

## Knowledge Bridges

### KnowledgeBridge

Enables knowledge transfer between memory banks.

```python
from src.bridges.knowledge_bridges import KnowledgeBridge

bridge = KnowledgeBridge(
    source_dim=64,
    target_dim=64,
    hidden_dim=32,
)

# Transfer knowledge
transferred = bridge(source_state, target_state)
gate_value = bridge.get_gate_value()
```

---

## Benchmarks

### Split-MNIST Dataset

```python
from benchmarks.split_mnist import create_split_mnist_tasks

tasks = create_split_mnist_tasks(
    num_tasks=5,
    batch_size=64,
)

for task_id, (train_loader, test_loader) in enumerate(tasks):
    # Train on task
    ...
```

### Metrics

```python
from benchmarks.metrics import compute_continual_metrics

metrics = compute_continual_metrics(accuracy_matrix)
# Returns: average_accuracy, forgetting, backward_transfer
```

---

## Example: Complete Training Loop

```python
import torch
from src.optimizers.collaborative_cms import CollaborativeCMSOptimizer
from benchmarks.split_mnist import create_split_mnist_tasks
from benchmarks.metrics import compute_continual_metrics

# Model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# Optimizer
optimizer = CollaborativeCMSOptimizer(
    model.parameters(),
    lr=0.01,
    regularization_strength=5.0,
    enable_bridges=True,
)

# Training
criterion = torch.nn.CrossEntropyLoss()
accuracy_matrix = []

for task_id, (train_loader, test_loader) in enumerate(create_split_mnist_tasks()):
    # Train on task
    for epoch in range(3):
        for x, y in train_loader:
            x = x.view(-1, 784)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate on all tasks seen so far
    task_accuracies = evaluate_all_tasks(model, tasks[:task_id+1])
    accuracy_matrix.append(task_accuracies)

# Compute metrics
metrics = compute_continual_metrics(accuracy_matrix)
print(f"Average Accuracy: {metrics['average_accuracy']:.2%}")
print(f"Forgetting: {metrics['forgetting']:.2%}")