
import torch
import torch.nn as nn

def replace_relu(module):
    """Replace ReLU with LeakyReLU to prevent dead neurons"""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(module, name, torch.nn.LeakyReLU(negative_slope=0.1, inplace=True))
        else:
            replace_relu(child)

class LoRALayer(torch.nn.Module):
    def __init__(self, original_linear, rank=4, alpha=8):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.scaling = alpha / rank
        
        # A: [in, r], B: [r, out]
        self.lora_A = torch.nn.Parameter(torch.randn(original_linear.in_features, rank) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, original_linear.out_features))
            
    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora

def apply_lora(module, target_names=["Linear"], rank=4, alpha=8):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, LoRALayer(child, rank=rank, alpha=alpha))
        else:
            apply_lora(child, target_names, rank, alpha)
