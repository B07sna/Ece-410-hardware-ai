# ResNet-18 Profiling Analysis
**Hussain Alquzweni — ECE 410, Spring 2026**

---

## Top-5 Layers by MAC Count

The torchinfo summary reports Total mult-adds of 1.81 GB for ResNet-18 (batch=1, input 3×224×224, FP32).
MACs per layer are computed from the convolution formula:
`MACs = C_in × C_out × K × K × H_out × W_out`

| Rank | Layer Name | Output Shape | MACs | Params |
|------|------------|-------------|------|--------|
| 1 | Conv2d: 3-42 (layer4.0, 2nd conv) | [1, 512, 7, 7] | 2,359,296 × 49 = ~115.6 M | 2,359,296 |
| 2 | Conv2d: 3-46 (layer4.1, 1st conv) | [1, 512, 7, 7] | ~115.6 M | 2,359,296 |
| 3 | Conv2d: 3-49 (layer4.1, 2nd conv) | [1, 512, 7, 7] | ~115.6 M | 2,359,296 |
| 4 | Conv2d: 3-39 (layer4.0, 1st conv) | [1, 512, 7, 7] | ~57.8 M | 1,179,648 |
| 5 | Conv2d: 3-29 (layer3.0, 2nd conv) | [1, 256, 14, 14] | ~115.6 M | 589,824 |

> Note: torchinfo reports total mult-adds (MACs) per layer grouped by blocks. The top MAC-intensive layers are all 512-channel conv layers in layer4 with 7×7 spatial output.

---

## Arithmetic Intensity — Most MAC-Intensive Layer

**Layer:** `Conv2d: 3-42` — layer4.0, second 3×3 conv  
**Config:** C_in=512, C_out=512, K=3, H_out=7, W_out=7

### Step 1: Compute MACs
```
MACs = C_in × C_out × K × K × H_out × W_out
     = 512 × 512 × 3 × 3 × 7 × 7
     = 512 × 512 × 9 × 49
     = 115,605,504 MACs
```

### Step 2: Compute FLOPs
```
FLOPs = 2 × MACs = 2 × 115,605,504 = 231,211,008 FLOPs
```

### Step 3: Compute Weight Bytes (FP32, no reuse)
```
Weight params = C_in × C_out × K × K = 512 × 512 × 3 × 3 = 2,359,296
Weight bytes  = 2,359,296 × 4 = 9,437,184 bytes
```

### Step 4: Compute Activation Bytes (input + output, FP32, no reuse)
```
Input activation:  C_in  × H_in  × W_in  = 512 × 7 × 7 = 25,088  → × 4 = 100,352 bytes
Output activation: C_out × H_out × W_out = 512 × 7 × 7 = 25,088  → × 4 = 100,352 bytes
Total activation bytes = 100,352 + 100,352 = 200,704 bytes
```

### Step 5: Compute Arithmetic Intensity
```
Arithmetic Intensity = FLOPs / (Weight bytes + Activation bytes)
                     = 231,211,008 / (9,437,184 + 200,704)
                     = 231,211,008 / 9,637,888
                     ≈ 23.99 FLOP/byte
```

**Arithmetic Intensity ≈ 24.0 FLOP/byte**

This is relatively low, meaning this layer is memory-bandwidth bound on most hardware accelerators (which typically have a ridge point of 100–300 FLOP/byte).
