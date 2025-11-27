"""Multi-Scale Continuum Memory System.

Generalizes CMS to N memory banks with geometric frequency progression.
Each bank stores parameter snapshots at its timescale for forgetting prevention.

Reference: Nested Learning (Behrouz et al., NeurIPS 2025)
Our extension: Arbitrary depth memory hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from src.memory.memory_bank import MemoryBank


@dataclass
class MultiScaleCMSConfig:
    """Configuration for multi-scale CMS.
    
    Args:
        num_levels: Number of memory bank levels.
        freq_ratio: Frequency ratio between adjacent levels.
        regularization_strength: Strength of regularization loss.
        decay: Exponential decay for memory updates.
    """
    
    num_levels: int = 3
    freq_ratio: float = 5.0
    regularization_strength: float = 1.0
    decay: float = 0.99
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {self.num_levels}")
        if self.freq_ratio <= 1.0:
            raise ValueError(f"freq_ratio must be > 1.0, got {self.freq_ratio}")
        if self.regularization_strength < 0:
            raise ValueError(f"regularization_strength must be >= 0, got {self.regularization_strength}")
    
    def get_frequencies(self) -> List[int]:
        """Get update frequency for each level.
        
        Returns:
            List of frequencies (steps between updates).
        """
        return [max(1, int(self.freq_ratio ** i)) for i in range(self.num_levels)]
    
    def get_level_names(self) -> List[str]:
        """Get human-readable names for each level.
        
        Returns:
            List of level names.
        """
        names = ['gamma', 'beta', 'alpha', 'theta', 'delta', 'infra_slow', 'ultra_slow']
        if self.num_levels <= len(names):
            return names[:self.num_levels]
        return [f'level_{i}' for i in range(self.num_levels)]


class MultiScaleCMS:
    """Multi-scale Continuum Memory System with N banks.
    
    Each bank stores parameter snapshots at its timescale:
    - Fast banks (gamma, beta): Capture recent patterns
    - Slow banks (theta, delta): Preserve long-term knowledge
    
    Regularization loss penalizes drift from stored memories,
    preventing catastrophic forgetting.
    
    Args:
        params: Model parameters to track.
        config: MultiScaleCMSConfig with all hyperparameters.
    
    Example:
        >>> config = MultiScaleCMSConfig(num_levels=5, freq_ratio=5.0)
        >>> cms = MultiScaleCMS(model.parameters(), config)
        >>> for step in range(1000):
        ...     cms.update(list(model.parameters()))
        ...     reg_loss = cms.compute_regularization_loss(list(model.parameters()))
    """
    
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        config: Optional[MultiScaleCMSConfig] = None,
    ):
        if config is None:
            config = MultiScaleCMSConfig()
        
        self.config = config
        self.num_levels = config.num_levels
        
        self._params_list = list(params)
        frequencies = config.get_frequencies()
        level_names = config.get_level_names()
        
        # Create memory bank for each level
        self.banks: List[MemoryBank] = []
        for i, (freq, name) in enumerate(zip(frequencies, level_names)):
            bank = MemoryBank(
                update_frequency=freq,
                decay_rate=config.decay,
                name=name,
            )
            # Initialize with current params
            bank.force_update(self._params_list)
            self.banks.append(bank)
        
        # Importance accumulator for consolidation
        self._importance_accumulator: Dict[int, torch.Tensor] = {}
        
        # Step counter
        self._step_count = 0
    
    def update(self, params: List[nn.Parameter]) -> Dict[str, bool]:
        """Update all memory banks.
        
        Args:
            params: Current model parameters.
        
        Returns:
            Dictionary mapping bank name to whether it updated.
        """
        self._step_count += 1
        result = {}
        
        for bank in self.banks:
            updated = bank.maybe_update(params)
            result[bank.name] = updated
        
        return result
    
    def compute_regularization_loss(
        self,
        params: List[nn.Parameter],
        importance: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute regularization loss from all banks.
        
        The loss penalizes parameter drift from stored memories,
        weighted by importance and bank level.
        
        Args:
            params: Current model parameters.
            importance: Optional importance weights per parameter.
        
        Returns:
            Scalar regularization loss tensor.
        """
        total_loss = torch.tensor(0.0)
        
        # Weight slower banks more heavily (they store more important knowledge)
        for i, bank in enumerate(self.banks):
            # Slower banks get higher weight
            bank_weight = 1.0 + (i / self.num_levels)
            loss = bank.compute_regularization_loss(params, importance)
            total_loss = total_loss + bank_weight * loss
        
        return total_loss * self.config.regularization_strength
    
    def accumulate_importance(self, params: List[nn.Parameter]) -> None:
        """Accumulate importance from gradients.
        
        Uses gradient magnitude as importance signal.
        
        Args:
            params: Model parameters with gradients.
        """
        for i, p in enumerate(params):
            if p.grad is not None:
                grad_importance = p.grad.abs()
                
                if i not in self._importance_accumulator:
                    self._importance_accumulator[i] = torch.zeros_like(p)
                
                self._importance_accumulator[i].add_(grad_importance)
    
    def reset_importance(self) -> None:
        """Reset importance accumulator."""
        self._importance_accumulator = {}
    
    def consolidate(
        self,
        params: List[nn.Parameter],
        importance: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, int]:
        """Consolidate knowledge from fast to slow banks.
        
        Call this at task boundaries to preserve important knowledge.
        
        Args:
            params: Current model parameters.
            importance: Optional pre-computed importance.
        
        Returns:
            Dictionary with consolidation counts per bank.
        """
        if importance is None:
            importance = self._importance_accumulator
        
        result = {}
        
        # Consolidate from fast to slow
        for i in range(self.num_levels - 1):
            source_bank = self.banks[i]
            target_bank = self.banks[i + 1]
            
            count = source_bank.consolidate_to(target_bank, params, importance)
            result[f"{source_bank.name}_to_{target_bank.name}"] = count
        
        return result
    
    def compute_drift(self, params: List[nn.Parameter]) -> Dict[str, Dict[int, float]]:
        """Compute parameter drift from each bank.
        
        Args:
            params: Current model parameters.
        
        Returns:
            Dictionary mapping bank name to drift per parameter.
        """
        result = {}
        
        for bank in self.banks:
            result[bank.name] = bank.compute_drift(params)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all banks.
        
        Returns:
            Dictionary with stats per bank.
        """
        return {
            'step_count': self._step_count,
            'num_levels': self.num_levels,
            'banks': {bank.name: bank.get_stats() for bank in self.banks},
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for serialization.
        
        Returns:
            Dictionary containing CMS state.
        """
        return {
            'step_count': self._step_count,
            'banks': [bank.state_dict() for bank in self.banks],
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from serialization.
        
        Args:
            state_dict: Dictionary containing CMS state.
        """
        self._step_count = state_dict.get('step_count', 0)
        
        if 'banks' in state_dict:
            for bank, bank_state in zip(self.banks, state_dict['banks']):
                bank.load_state_dict(bank_state)


def _test_multi_scale_cms():
    """Test the multi-scale CMS."""
    print("Testing MultiScaleCMS...")
    
    # Create model
    model = nn.Linear(10, 5)
    params = list(model.parameters())
    
    # Test different configurations
    for num_levels in [3, 5, 7]:
        print(f"\n  Testing {num_levels} levels...")
        
        config = MultiScaleCMSConfig(
            num_levels=num_levels,
            freq_ratio=3.0,
            regularization_strength=1.0,
        )
        
        cms = MultiScaleCMS(params, config)
        
        print(f"    Frequencies: {config.get_frequencies()}")
        print(f"    Level names: {config.get_level_names()}")
        
        # Simulate training
        for step in range(100):
            # Fake gradients
            for p in params:
                p.grad = torch.randn_like(p)
            
            # Update CMS
            update_result = cms.update(params)
            
            # Accumulate importance
            cms.accumulate_importance(params)
            
            # Compute regularization
            reg_loss = cms.compute_regularization_loss(params)
        
        # Get stats
        stats = cms.get_stats()
        print(f"    Step count: {stats['step_count']}")
        
        for bank_name, bank_stats in stats['banks'].items():
            print(f"    {bank_name}: updates={bank_stats['total_updates']}")
        
        # Test consolidation
        consolidation = cms.consolidate(params)
        print(f"    Consolidation: {consolidation}")
    
    print("\n✓ MultiScaleCMS test passed!")


if __name__ == "__main__":
    _test_multi_scale_cms()