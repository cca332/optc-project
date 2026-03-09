import pytest
import torch
from omegaconf import OmegaConf
from pathlib import Path

def test_federated_toggle_enabled():
    """Test that federated learning is enabled in the config."""
    config_path = Path(__file__).parents[1] / "config" / "ecar_config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Check the flag
    assert cfg.training.federated_learning is True, "Federated learning should be enabled"

def test_teacher_freeze_logic():
    """Test that teacher parameters can be frozen correctly."""
    # Mock a simple teacher model
    teacher = torch.nn.Linear(10, 2)
    
    # Verify initially requires_grad is True
    for p in teacher.parameters():
        assert p.requires_grad is True
        
    # Apply freeze logic
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    
    # Verify frozen
    for p in teacher.parameters():
        assert p.requires_grad is False
    assert not teacher.training
