"""
nn_forward_gpu.py
-----------------
Detects CUDA GPU, builds a small feed-forward network, moves it to GPU,
and runs one forward pass with a random [16, 4] input.

Network architecture:
    Linear(4 -> 5)  →  ReLU  →  Linear(5 -> 1)
"""

import torch
import torch.nn as nn

# ── 1. Device detection ────────────────────────────────────────────────────────
print("=" * 50)
print("Device Detection")
print("=" * 50)

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected — cannot proceed.")

device = torch.device("cuda")
print(f"CUDA available   : True")
print(f"Device index     : {torch.cuda.current_device()}")
print(f"Device name      : {torch.cuda.get_device_name(0)}")
print(f"CUDA capability  : {torch.cuda.get_device_capability(0)}")
print(f"Total VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"PyTorch version  : {torch.__version__}")
print()

# ── 2. Network definition ──────────────────────────────────────────────────────
print("=" * 50)
print("Network Definition")
print("=" * 50)

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SmallNet()
print(model)
print()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {total_params}")
print(f"  fc1 weight     : {model.fc1.weight.shape}  ({model.fc1.weight.numel()} values)")
print(f"  fc1 bias       : {model.fc1.bias.shape}")
print(f"  fc2 weight     : {model.fc2.weight.shape}  ({model.fc2.weight.numel()} values)")
print(f"  fc2 bias       : {model.fc2.bias.shape}")
print()

# ── 3. Move model to GPU ───────────────────────────────────────────────────────
print("=" * 50)
print("Moving Model to GPU")
print("=" * 50)

model = model.to(device)
print(f"Model device (fc1.weight) : {model.fc1.weight.device}")
print(f"Model device (fc2.weight) : {model.fc2.weight.device}")
print()

# ── 4. Forward pass ────────────────────────────────────────────────────────────
print("=" * 50)
print("Forward Pass")
print("=" * 50)

torch.manual_seed(42)
x = torch.randn(16, 4, device=device)

print(f"Input shape   : {list(x.shape)}")
print(f"Input device  : {x.device}")
print()

model.eval()
with torch.no_grad():
    output = model(x)

print(f"Output shape  : {list(output.shape)}")
print(f"Output device : {output.device}")
print()
print("Output values (16 × 1):")
print(output)
print()

# ── 5. Summary ─────────────────────────────────────────────────────────────────
print("=" * 50)
print("Summary")
print("=" * 50)
print(f"  Input  : {list(x.shape)}  on  {x.device}")
print(f"  Output : {list(output.shape)}  on  {output.device}")
print(f"  Min    : {output.min().item():.6f}")
print(f"  Max    : {output.max().item():.6f}")
print(f"  Mean   : {output.mean().item():.6f}")
print("=" * 50)
