# Arithmetic Intensity Analysis — MobileNetV2 FP32

**Source:** `profiling/project_profile.txt`
**Model:** MobileNetV2, FP32, CPU (PyTorch 2.11.0+cpu)
**Config:** Batch size 1, Input 3×224×224, 10 inference runs

---

## 1. Identifying the Dominant Kernel Layer

The profiling table is sorted by **Self CPU** (time spent in the kernel itself, excluding children).
The top convolution kernel by Self CPU is:

| Field | Value |
|---|---|
| Op | `aten::mkldnn_convolution` |
| Input shape | `[1, 384, 14, 14]` |
| Weight shape | `[384, 1, 3, 3]` |
| # Calls (total) | 40 (= 4 calls/inference × 10 runs) |
| Self CPU (total) | 3.731 ms |
| Self CPU (per call) | ~93.3 µs |
| Profiler KFLOPs | 54,190.080 KFLOPs (total, 40 calls) |

**Layer type:** Depthwise Conv2d — kernel shape `[out_ch, in_ch/groups, kH, kW]` = `[384, 1, 3, 3]`
means `groups = 384`, i.e., each channel is convolved independently.

**MobileNetV2 context:** This is the 3×3 depthwise stage inside the inverted residual blocks
operating at 14×14 spatial resolution with 64 input channels expanded by factor 6 → 384 channels
(blocks 6–9 of MobileNetV2). Four such blocks exist, hence 4 calls per inference.

---

## 2. FLOPs Calculation

For a depthwise Conv2d each of the `C` channels is convolved with its own `Kh×Kw` filter independently.

**Formula:**

```
FLOPs = 2 × C × Kh × Kw × Oh × Ow
```

Where:
- `2` accounts for one multiply and one accumulate per MAC
- `C`  = number of channels = groups = **384**
- `Kh × Kw` = kernel spatial size = **3 × 3 = 9**
- `Oh × Ow` = output spatial size = **14 × 14 = 196**  
  (padding=1, stride=1 → same spatial dims as input)

**Substituting values:**

```
FLOPs = 2 × 384 × 9 × 196
      = 2 × 384 × 1,764
      = 2 × 677,376
      = 1,354,752 FLOPs
```

**Cross-check against profiler:**
```
Profiler total KFLOPs = 54,190,080 FLOPs (40 calls)
Per call              = 54,190,080 / 40 = 1,354,752 FLOPs  ✓
```

---

## 3. Bytes Transferred (No DRAM Reuse)

Assumption: every weight and activation tensor is loaded from DRAM exactly once, with no on-chip
caching or reuse. All values are FP32 → **4 bytes per element**.

### 3a. Weights

```
Shape  : [384, 1, 3, 3]
Elements: 384 × 1 × 3 × 3 = 3,456
Bytes  : 3,456 × 4 = 13,824 bytes
```

### 3b. Input Activation

```
Shape   : [1, 384, 14, 14]
Elements: 384 × 14 × 14 = 75,264
Bytes   : 75,264 × 4 = 301,056 bytes
```

### 3c. Output Activation

```
Shape   : [1, 384, 14, 14]   (same as input — same-padded conv)
Elements: 384 × 14 × 14 = 75,264
Bytes   : 75,264 × 4 = 301,056 bytes
```

### 3d. Total Bytes Transferred

```
Bytes_total = Bytes_weights + Bytes_input + Bytes_output
            = 13,824 + 301,056 + 301,056
            = 615,936 bytes
            ≈ 601.5 KB
```

---

## 4. Arithmetic Intensity

**Formula:**

```
AI = FLOPs / Bytes_transferred   [FLOPs/byte]
```

**Substituting values:**

```
AI = 1,354,752 / 615,936
   ≈ 2.20 FLOPs/byte
```

---

## 5. Summary Table

| Quantity | Formula | Value |
|---|---|---|
| FLOPs | `2 × C × Kh × Kw × Oh × Ow` | **1,354,752 FLOPs** |
| Weight bytes | `C × 1 × Kh × Kw × 4` | 13,824 B |
| Input bytes | `1 × C × Oh × Ow × 4` | 301,056 B |
| Output bytes | `1 × C × Oh × Ow × 4` | 301,056 B |
| Total bytes | weights + input + output | **615,936 B** |
| **Arithmetic Intensity** | FLOPs / Bytes | **≈ 2.20 FLOPs/byte** |

---

## 6. Interpretation

An arithmetic intensity of **~2.20 FLOPs/byte** is extremely low. For reference:

- A100 GPU roofline crossover point: ~208 FLOPs/byte (fp32)
- Typical 1×1 pointwise conv in MobileNetV2: ~12–25 FLOPs/byte
- Standard 3×3 conv (non-depthwise): ~20–300 FLOPs/byte depending on channel width

**Depthwise convolutions are inherently memory-bound.** Because each channel uses only one
filter (no accumulation across channels), the data reuse ratio for weights collapses from
`Cin × Kh × Kw` (standard conv) to just `Kh × Kw = 9`. The output cannot amortize weight
loads over many input channels. On virtually any hardware platform this kernel will be
limited by DRAM bandwidth, not compute throughput.

**Implication for hardware design:** To accelerate this layer, maximizing memory bandwidth
(e.g., wider buses, HBM, higher cache hit rate) is more effective than adding more MACs.
Quantization (FP16/INT8) halves/quarters byte transfers and can push AI into a less
memory-bound regime.
