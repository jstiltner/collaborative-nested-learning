# Contributing to Collaborative Nested Learning

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Git

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/collaborative-nested-learning
   cd collaborative-nested-learning
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

5. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Standards

### Style Guide

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **mypy** for type checking

Run formatting before committing:
```bash
black src/ tests/ benchmarks/
isort src/ tests/ benchmarks/
```

### Type Hints

All functions should have complete type hints:

```python
def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute weighted loss."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the knowledge bridge.
    
    Args:
        x: Input tensor of shape (batch_size, hidden_dim).
    
    Returns:
        Output tensor of shape (batch_size, hidden_dim).
    
    Raises:
        ValueError: If x has incorrect dimensions.
    """
```

### Tensor Shape Comments

Always annotate tensor shapes:

```python
# x: (B, T, H) where B=batch, T=time, H=hidden
x = self.projection(x)  # (B, T, H) -> (B, T, D)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_deep_momentum.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_<module_name>.py`
- Use pytest fixtures for common setup
- Test edge cases (empty batches, single samples, etc.)

Example:
```python
import pytest
import torch
from src.optimizers.deep_momentum import DeepMomentumOptimizer

class TestDeepMomentumOptimizer:
    @pytest.fixture
    def optimizer(self):
        model = torch.nn.Linear(10, 10)
        return DeepMomentumOptimizer(model.parameters(), lr=0.01)
    
    def test_step_updates_params(self, optimizer):
        # Test implementation
        ...
```

## Pull Request Process

### Before Submitting

1. Ensure all tests pass: `pytest tests/`
2. Format your code: `black src/ tests/`
3. Update documentation if needed
4. Add tests for new functionality

### PR Guidelines

1. **Title**: Use a clear, descriptive title
   - Good: "Add adaptive bridge topology selection"
   - Bad: "Fix stuff"

2. **Description**: Include:
   - What the PR does
   - Why it's needed
   - How to test it

3. **Size**: Keep PRs focused and reasonably sized
   - Large changes should be split into smaller PRs

### Commit Messages

Use conventional commit format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Example:
```
feat(bridges): add adaptive topology selection

Implement dynamic bridge topology that adjusts based on
regularization strength. Adjacent-only bridges are used
at high regularization for stability.

Closes #12
```

## Areas for Contribution

### High Priority

- **Additional benchmarks**: CIFAR-100, language modeling tasks
- **Performance optimization**: CUDA kernels, memory efficiency
- **Documentation**: API docs, tutorials, examples

### Medium Priority

- **Adaptive bridge topology**: Dynamic selection based on training dynamics
- **Integration**: PyTorch Lightning, HuggingFace Trainer
- **Visualization**: Interactive training dashboards

### Research Directions

- **Theoretical analysis**: Convergence guarantees, optimal frequencies
- **Alternative architectures**: Attention-based bridges, graph neural networks
- **Multi-modal learning**: Cross-modal knowledge transfer

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Email: jason@stiltner.com

## Code of Conduct

Be respectful and constructive. We're all here to learn and build something useful.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.