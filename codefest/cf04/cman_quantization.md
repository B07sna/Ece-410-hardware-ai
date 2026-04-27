# INT8 Symmetric Quantization Analysis

**Matrix:** 4 × 4 FP32 weight matrix  
**Scheme:** Symmetric per-tensor INT8 — one scale factor S for the entire tensor,  
zero-point fixed at 0, representable range **[-128, 127]**.

---

## Input Matrix W (FP32)

$$
W = \begin{bmatrix}
 0.85 & -1.20 &  0.34 &  2.10 \\
-0.07 &  0.91 & -1.88 &  0.12 \\
 1.55 &  0.03 & -0.44 & -2.31 \\
-0.18 &  1.03 &  0.77 &  0.55
\end{bmatrix}
$$

---

## (a) Scale Factor S

Symmetric quantization maps the largest absolute value to 127:

$$
S = \frac{\max(|W|)}{127}
$$

**Absolute-value matrix |W|:**

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| row 0 | 0.85 | 1.20 | 0.34 | **2.10** |
| row 1 | 0.07 | 0.91 | 1.88 | 0.12 |
| row 2 | 1.55 | 0.03 | 0.44 | **2.31** |
| row 3 | 0.18 | 1.03 | 0.77 | 0.55 |

$$
\max(|W|) = 2.31 \quad \text{(position [2,3])}
$$

$$
\boxed{S = \frac{2.31}{127} \approx 0.018189}
$$

Each INT8 step represents exactly S = 0.018189 in FP32.  
The representable FP32 range is [−127·S, +127·S] = [−2.31, +2.31].

---

## (b) Quantized Matrix W_q

$$
W_q = \text{clamp}\!\left(\text{round}\!\left(\frac{W}{S}\right),\,-128,\,127\right)
$$

The scale factor 1/S = 127/2.31 ≈ **54.987**.

### Step-by-step: W / S (before rounding)

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| row 0 | 0.85 × 54.987 = **46.739** | −1.20 × 54.987 = **−65.984** | 0.34 × 54.987 = **18.696** | 2.10 × 54.987 = **115.473** |
| row 1 | −0.07 × 54.987 = **−3.849** | 0.91 × 54.987 = **50.038** | −1.88 × 54.987 = **−103.375** | 0.12 × 54.987 = **6.598** |
| row 2 | 1.55 × 54.987 = **85.230** | 0.03 × 54.987 = **1.649** | −0.44 × 54.987 = **−24.194** | −2.31 × 54.987 = **−127.000** |
| row 3 | −0.18 × 54.987 = **−9.897** | 1.03 × 54.987 = **56.637** | 0.77 × 54.987 = **42.340** | 0.55 × 54.987 = **30.243** |

### After round() and clamp(·, −128, 127)

No value exceeds the INT8 range before clamping, so **no saturation occurs**.

$$
W_q = \begin{bmatrix}
  47 & -66 &  19 &  115 \\
  -4 &  50 & -103 &   7 \\
  85 &   2 &  -24 & -127 \\
 -10 &  57 &   42 &  30
\end{bmatrix}
$$

> Note: element [2,3] = −127 exactly because −2.31 is precisely the maximum  
> absolute value used to derive S.

---

## (c) Dequantized Matrix W_deq

$$
W_{\text{deq}} = W_q \times S \qquad (S = 2.31/127)
$$

Multiply each INT8 value by S = 0.018189:

$$
W_{\text{deq}} = \begin{bmatrix}
 0.854882 & -1.200472 &  0.345591 &  2.091732 \\
-0.072756 &  0.909449 & -1.873465 &  0.127323 \\
 1.546063 &  0.036378 & -0.436535 & -2.310000 \\
-0.181890 &  1.036772 &  0.763937 &  0.545669
\end{bmatrix}
$$

**Derivation for each element** (W_q[i,j] × 2.31 / 127):

| Position | W_q | Computation | W_deq |
|---|---|---|---|
| [0,0] | 47 | 47 × 2.31 / 127 = 108.57 / 127 | **0.854882** |
| [0,1] | −66 | −66 × 2.31 / 127 = −152.46 / 127 | **−1.200472** |
| [0,2] | 19 | 19 × 2.31 / 127 = 43.89 / 127 | **0.345591** |
| [0,3] | 115 | 115 × 2.31 / 127 = 265.65 / 127 | **2.091732** |
| [1,0] | −4 | −4 × 2.31 / 127 = −9.24 / 127 | **−0.072756** |
| [1,1] | 50 | 50 × 2.31 / 127 = 115.50 / 127 | **0.909449** |
| [1,2] | −103 | −103 × 2.31 / 127 = −237.93 / 127 | **−1.873465** |
| [1,3] | 7 | 7 × 2.31 / 127 = 16.17 / 127 | **0.127323** |
| [2,0] | 85 | 85 × 2.31 / 127 = 196.35 / 127 | **1.546063** |
| [2,1] | 2 | 2 × 2.31 / 127 = 4.62 / 127 | **0.036378** |
| [2,2] | −24 | −24 × 2.31 / 127 = −55.44 / 127 | **−0.436535** |
| [2,3] | −127 | −127 × 2.31 / 127 = −2.31 | **−2.310000** |
| [3,0] | −10 | −10 × 2.31 / 127 = −23.10 / 127 | **−0.181890** |
| [3,1] | 57 | 57 × 2.31 / 127 = 131.67 / 127 | **1.036772** |
| [3,2] | 42 | 42 × 2.31 / 127 = 97.02 / 127 | **0.763937** |
| [3,3] | 30 | 30 × 2.31 / 127 = 69.30 / 127 | **0.545669** |

---

## (d) Per-Element Absolute Error

$$
E = |W - W_{\text{deq}}|
$$

The rounding error for each element is bounded by ½ × S = ½ × 0.018189 ≈ **0.009094**  
(the maximum possible rounding error for any single element).

### Error table

| Position | W (orig) | W_deq | \|Error\| | Note |
|---|---|---|---|---|
| [0,0] | 0.8500 | 0.854882 | **0.004882** | |
| [0,1] | −1.2000 | −1.200472 | **0.000472** | |
| [0,2] | 0.3400 | 0.345591 | **0.005591** | |
| [0,3] | 2.1000 | 2.091732 | **0.008268** | ← largest |
| [1,0] | −0.0700 | −0.072756 | **0.002756** | |
| [1,1] | 0.9100 | 0.909449 | **0.000551** | |
| [1,2] | −1.8800 | −1.873465 | **0.006535** | |
| [1,3] | 0.1200 | 0.127323 | **0.007323** | |
| [2,0] | 1.5500 | 1.546063 | **0.003937** | |
| [2,1] | 0.0300 | 0.036378 | **0.006378** | |
| [2,2] | −0.4400 | −0.436535 | **0.003465** | |
| [2,3] | −2.3100 | −2.310000 | **0.000000** | exact (= max\|W\|) |
| [3,0] | −0.1800 | −0.181890 | **0.001890** | |
| [3,1] | 1.0300 | 1.036772 | **0.006772** | |
| [3,2] | 0.7700 | 0.763937 | **0.006063** | |
| [3,3] | 0.5500 | 0.545669 | **0.004331** | |

### Error matrix

$$
E = \begin{bmatrix}
0.004882 & 0.000472 & 0.005591 & 0.008268 \\
0.002756 & 0.000551 & 0.006535 & 0.007323 \\
0.003937 & 0.006378 & 0.003465 & 0.000000 \\
0.001890 & 0.006772 & 0.006063 & 0.004331
\end{bmatrix}
$$

### Largest error element

$$
\max(E) = \mathbf{0.008268} \quad \text{at position [0,3], } W = 2.10
$$

Why is this the largest? 2.10 / S = 115.454, which rounds to 115 — a rounding residual of  
−0.454 steps. The resulting error is 0.454 × S = 0.454 × 0.018189 = **0.008268**.  
(Element [2,3] is exact because −2.31 maps to exactly −127.000 with zero fractional part.)

### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{16}\sum_{i,j} |E_{ij}|
= \frac{0.069214}{16} = \boxed{0.004326}
$$

Sum detail:

```
Row 0: 0.004882 + 0.000472 + 0.005591 + 0.008268 = 0.019213
Row 1: 0.002756 + 0.000551 + 0.006535 + 0.007323 = 0.017165
Row 2: 0.003937 + 0.006378 + 0.003465 + 0.000000 = 0.013780
Row 3: 0.001890 + 0.006772 + 0.006063 + 0.004331 = 0.019056
                                           Total = 0.069214
                                     MAE = 0.069214 / 16 = 0.004326
```

**Sanity check:** MAE = 0.004326 < ½ × S = 0.009094 ✓  
All errors are below the theoretical maximum rounding error, confirming no overflow occurred.

---

## (e) Bad Scale S_bad = 0.01 — What Goes Wrong

### Why S_bad = 0.01 is too small

The representable FP32 range under S_bad = 0.01 is:

$$
[-128 \times 0.01,\ +127 \times 0.01] = [-1.28,\ +1.27]
$$

But **W contains values outside this range** (2.10, −1.88, 1.55, −2.31).  
Those values cannot be represented — they saturate (clamp) to the INT8 boundary,  
permanently destroying information that **cannot be recovered** at dequantization.

The minimum scale that safely covers all values is S_min = 2.31/127 ≈ 0.018189.  
Using S_bad = 0.01 < S_min means the scale is **45% too small**.

---

### Quantization with S_bad = 0.01

$$
W / S_{\text{bad}} \quad\text{(before clamp):}
$$

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| row 0 | 85 | −120 | 34 | **210** ⚠ |
| row 1 | −7 | 91 | **−188** ⚠ | 12 |
| row 2 | **155** ⚠ | 3 | −44 | **−231** ⚠ |
| row 3 | −18 | 103 | 77 | 55 |

After clamp(·, −128, 127) — saturated values shown in **bold**:

$$
W_{q,\text{bad}} = \begin{bmatrix}
  85 & -120 &  34 & \mathbf{127} \\
  -7 &   91 & \mathbf{-128} &  12 \\
\mathbf{127} &    3 &  -44 & \mathbf{-128} \\
 -18 &  103 &   77 &  55
\end{bmatrix}
$$

4 of 16 elements (25%) are saturated.

---

### Dequantization with S_bad

$$
W_{\text{deq,bad}} = W_{q,\text{bad}} \times 0.01
$$

$$
W_{\text{deq,bad}} = \begin{bmatrix}
 0.85 & -1.20 &  0.34 & \mathbf{1.27} \\
-0.07 &  0.91 & \mathbf{-1.28} &  0.12 \\
\mathbf{1.27} &  0.03 & -0.44 & \mathbf{-1.28} \\
-0.18 &  1.03 &  0.77 &  0.55
\end{bmatrix}
$$

---

### Per-element errors under S_bad

| Position | W (orig) | W_deq,bad | \|Error\| | Type |
|---|---|---|---|---|
| [0,0] | 0.85 | 0.85 | **0.0000** | exact (multiple of 0.01) |
| [0,1] | −1.20 | −1.20 | **0.0000** | exact |
| [0,2] | 0.34 | 0.34 | **0.0000** | exact |
| **[0,3]** | **2.10** | **1.27** | **0.8300** | ← SATURATED |
| [1,0] | −0.07 | −0.07 | **0.0000** | exact |
| [1,1] | 0.91 | 0.91 | **0.0000** | exact |
| **[1,2]** | **−1.88** | **−1.28** | **0.6000** | ← SATURATED |
| [1,3] | 0.12 | 0.12 | **0.0000** | exact |
| **[2,0]** | **1.55** | **1.27** | **0.2800** | ← SATURATED |
| [2,1] | 0.03 | 0.03 | **0.0000** | exact |
| [2,2] | −0.44 | −0.44 | **0.0000** | exact |
| **[2,3]** | **−2.31** | **−1.28** | **1.0300** | ← SATURATED (worst) |
| [3,0] | −0.18 | −0.18 | **0.0000** | exact |
| [3,1] | 1.03 | 1.03 | **0.0000** | exact |
| [3,2] | 0.77 | 0.77 | **0.0000** | exact |
| [3,3] | 0.55 | 0.55 | **0.0000** | exact |

> The non-saturated elements are **exact** because all their FP32 values happen to be  
> exact multiples of 0.01 (e.g., 0.85 = 85 × 0.01), so rounding error is zero.  
> In practice with arbitrary weights, non-saturated elements would also have rounding  
> noise — but saturation errors would still dominate by orders of magnitude.

### MAE under S_bad

```
Saturated errors: 0.83 + 0.60 + 0.28 + 1.03 = 2.74
Non-saturated:    0.00 × 12 = 0.00
Total = 2.74
MAE_bad = 2.74 / 16 = 0.17125
```

$$
\boxed{\text{MAE}_{\text{bad}} = 0.17125}
$$

---

## Comparison Summary

| Metric | Good Scale (S = 0.018189) | Bad Scale (S_bad = 0.01) |
|---|---|---|
| Scale factor | 0.018189 | 0.010000 |
| FP32 range covered | [−2.31, +2.31] | [−1.28, +1.27] |
| Saturated elements | **0 / 16** | **4 / 16 (25%)** |
| Max element error | 0.008268 | 1.030000 |
| MAE | **0.004326** | **0.171250** |
| MAE ratio | 1× (baseline) | **~39.6× worse** |

---

## Root-Cause Explanation

### Why saturation is catastrophic

Rounding error is **bounded** — it can never exceed ½ × S regardless of the value.  
Saturation error is **unbounded** — it grows linearly with how far the value exceeds  
the representable range:

$$
\text{saturation error} = |w| - 127 \cdot S_{\text{bad}} \quad \text{for } |w| > 127 \cdot S_{\text{bad}}
$$

For W[2,3] = −2.31:

$$
\text{error} = 2.31 - 127 \times 0.01 = 2.31 - 1.27 = 1.03
$$

This is **125× larger** than the worst rounding error under the good scale (0.008268).

### The scale–precision tradeoff

In symmetric INT8 quantization, S controls a fundamental tradeoff:

- **S too large** → representable range is wide (no saturation) but each INT8 step is  
  coarse → large rounding error for small values.  
- **S too small** → INT8 steps are fine (low rounding error for in-range values) but  
  large values saturate → catastrophic clipping error.

The optimal scale for a symmetric scheme is exactly:

$$
S^* = \frac{\max(|W|)}{127}
$$

This is what part (a) computes. It is the **unique scale** that places the largest absolute  
value exactly at the INT8 boundary, leaving no room for saturation while using the full  
8-bit dynamic range.

### Practical takeaway

Never choose a scale independently of the data. Using S_bad = 0.01 for this matrix  
discards ~25% of weights entirely (they all collapse to ±127), which in a neural network  
would corrupt the layer's output distribution and typically cause significant accuracy  
degradation — often far more than the mild rounding noise introduced by the optimal scale.
