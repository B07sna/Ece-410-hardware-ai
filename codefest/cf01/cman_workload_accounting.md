# CMAN Workload Accounting
**Hussain Alquzweni — ECE 410, Spring 2026**

Network: [784 → 256 → 128 → 10], batch size 1, FP32, no biases.

---

## (a) Per-Layer MACs

For a fully connected layer: `MACs = input_dim × output_dim`

| Layer | Formula | MACs |
|-------|---------|------|
| Layer 1 (784 → 256) | 784 × 256 | 200,704 |
| Layer 2 (256 → 128) | 256 × 128 | 32,768 |
| Layer 3 (128 → 10)  | 128 × 10  | 1,280  |

---

## (b) Total MACs

```
Total MACs = 200,704 + 32,768 + 1,280 = 234,752 MACs
```

---

## (c) Total Trainable Parameters (weights only, no biases)

```
Layer 1: 784 × 256  = 200,704
Layer 2: 256 × 128  =  32,768
Layer 3: 128 × 10   =   1,280
Total                = 234,752 parameters
```

---

## (d) Weight Memory (FP32 = 4 bytes each)

```
Weight memory = 234,752 × 4 = 939,008 bytes
```

---

## (e) Activation Memory (input + all layer outputs, FP32)

Activations to store simultaneously: input (784) + hidden1 (256) + hidden2 (128) + output (10)

```
Activation memory = (784 + 256 + 128 + 10) × 4
                  = 1,178 × 4
                  = 4,712 bytes
```

---

## (f) Arithmetic Intensity

```
FLOPs = 2 × Total MACs = 2 × 234,752 = 469,504

Arithmetic Intensity = FLOPs / (weight bytes + activation bytes)
                     = 469,504 / (939,008 + 4,712)
                     = 469,504 / 943,720
                     ≈ 0.497 FLOP/byte
```

**Arithmetic Intensity ≈ 0.50 FLOP/byte**

This is very low — the network is heavily memory-bandwidth bound due to the large weight matrices relative to the small activation footprint.
