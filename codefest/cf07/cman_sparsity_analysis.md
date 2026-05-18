# Sparsity Breakeven Analysis — Dense vs Sparse MVM
**ECE 410 · CodeFest 7 CMAN · Hussain Alquzweni**

---

## Parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| N | 512 | Matrix dimension (N × N square matrix) |
| s | 0 ≤ s < 1 | Sparsity — fraction of matrix elements that are zero |
| 1 − s | — | Density — fraction of non-zero elements |
| nnz | (1−s)·N² | Number of non-zero entries |
| BW | 320 GB/s | Memory bandwidth (used in §4) |

---

## (a) Dense Matrix–Vector Multiply (MVM)

### Compute

A dense N×N MVM computes:

```
y[i] = Σ_{j=0}^{N-1}  A[i][j] · x[j],    i = 0 … N-1
```

Each output element requires **N multiplications** and **N−1 additions** ≈ **2N FLOPs**.
For N output elements:

$$\text{FLOPs}_\text{dense} = 2N^2$$

$$= 2 \times 512^2 = 2 \times 262\,144 = \boxed{524\,288 \text{ FLOPs}}$$

### Memory footprint

Storing the matrix in FP32 (4 bytes per element):

$$\text{Bytes}_\text{dense} = 4 \cdot N^2$$

$$= 4 \times 262\,144 = \boxed{1\,048\,576 \text{ bytes}} \approx 1 \text{ MB}$$

---

## (b) Sparse CSR Matrix–Vector Multiply

### CSR storage format

Compressed Sparse Row (CSR) stores only the non-zero values:

| Array | Elements | Bytes/element | Total bytes |
|-------|----------|---------------|-------------|
| `values[nnz]` | (1−s)·N² | 4 (FP32) | 4·(1−s)·N² |
| `col_index[nnz]` | (1−s)·N² | 4 (int32) | 4·(1−s)·N² |
| `row_ptr[N+1]` | N+1 | 4 (int32) | 4·(N+1) |

Total:

$$\text{Bytes}_\text{CSR}(s) = 8(1-s)N^2 + 4(N+1)$$

Numerically (N = 512, N+1 = 513):

$$\text{Bytes}_\text{CSR}(s) = 8 \times (1-s) \times 262\,144 + 4 \times 513$$

$$= 2\,097\,152\,(1-s) + 2\,052 \text{ bytes}$$

### Compute

The sparse MVM only processes non-zero entries (one multiply-add per nnz):

$$\text{FLOPs}_\text{CSR}(s) = 2 \cdot (1-s) \cdot N^2$$

$$= 524\,288 \cdot (1-s) \text{ FLOPs}$$

---

## (c) FLOPs Speedup and Compute Breakeven

### Speedup formula

$$\text{Speedup}_\text{FLOPs}(s)
  = \frac{\text{FLOPs}_\text{dense}}{\text{FLOPs}_\text{CSR}(s)}
  = \frac{2N^2}{2N^2(1-s)}
  = \frac{1}{1-s}$$

| Sparsity s | 1 − s | Compute speedup |
|------------|-------|-----------------|
| 0% | 1.00 | 1.00× |
| 25% | 0.75 | 1.33× |
| 50% | 0.50 | **2.00×** |
| 75% | 0.25 | 4.00× |
| 90% | 0.10 | 10.0× |
| 95% | 0.05 | 20.0× |

### Solve for 2× compute speedup

$$\frac{1}{1-s} = 2 \implies 1-s = \tfrac{1}{2} \implies \boxed{s = 0.50}$$

**At 50% sparsity the sparse MVM executes exactly half as many FLOPs as dense.**

---

## (d) Memory Breakeven and End-to-End Speedup

### Memory breakeven derivation

Set sparse memory equal to dense memory and solve for s:

$$8(1-s)N^2 + 4(N+1) = 4N^2$$

$$8(1-s)N^2 = 4N^2 - 4(N+1)$$

$$8(1-s)N^2 = 4\bigl(N^2 - N - 1\bigr)$$

$$(1-s) = \frac{N^2 - N - 1}{2N^2}$$

$$s_\text{mem} = 1 - \frac{N^2 - N - 1}{2N^2} = \frac{2N^2 - N^2 + N + 1}{2N^2}$$

$$\boxed{s_\text{mem} = \frac{N^2 + N + 1}{2N^2}}$$

Substituting N = 512:

$$s_\text{mem} = \frac{262\,144 + 512 + 1}{2 \times 262\,144} = \frac{262\,657}{524\,288} \approx \mathbf{0.5010}$$

**Verification** (plug s = 262657/524288 back in):

```
1 − s = 261631/524288
Sparse bytes = 8 × (261631/524288) × 262144 + 4 × 513
             = 4 × 261631 + 2052
             = 1 046 524 + 2 052
             = 1 048 576  ✓  (= Dense bytes)
```

> **Interpretation.** CSR requires 8 bytes per non-zero (4B value + 4B column index), versus
> 4 bytes per element in dense format. Therefore the storage overhead factor is 2×, and the
> breakeven point must be slightly above 50% to absorb the constant 2,052-byte row-pointer
> overhead. For N = 512 the crossover is at **s ≈ 50.1%** — essentially identical to the
> compute breakeven.

### Comparison of breakeven points

| Criterion | Breakeven sparsity |
|-----------|--------------------|
| Compute (FLOPs) | s = 0.5000 (50.00%) |
| Memory (CSR vs dense) | s ≈ 0.5010 (50.10%) |

The two breakevens are nearly coincident for large N; both round to **50% sparsity**.

---

### End-to-end speedup at s = 0.9, bandwidth-limited regime

At 90% sparsity (nnz = 10% of N²):

**Dense bytes:**
$$\text{Bytes}_\text{dense} = 4 \times 262\,144 = 1\,048\,576 \text{ B}$$

**Sparse bytes:**
$$\text{Bytes}_\text{CSR}(0.9) = 8 \times 0.1 \times 262\,144 + 4 \times 513$$
$$= 209\,715.2 + 2\,052 = 211\,767 \text{ B} \approx 206.8 \text{ KiB}$$

**Transfer times at BW = 320 GB/s:**

$$t_\text{dense} = \frac{1\,048\,576}{320 \times 10^9} \approx 3.28 \text{ µs}$$

$$t_\text{sparse} = \frac{211\,767}{320 \times 10^9} \approx 0.662 \text{ µs}$$

**End-to-end memory speedup:**

$$\text{Speedup}_\text{mem}(0.9)
  = \frac{\text{Bytes}_\text{dense}}{\text{Bytes}_\text{CSR}(0.9)}
  = \frac{1\,048\,576}{211\,767}
  \approx \boxed{4.95\times}$$

**General formula:**

$$\text{Speedup}_\text{mem}(s) = \frac{4N^2}{8(1-s)N^2 + 4(N+1)}
  \;\xrightarrow{N \to \infty}\; \frac{1}{2(1-s)}$$

For s = 0.9: asymptotic limit = 1 / (2 × 0.1) = **5.00×**; exact result is **4.95×**
(the small shortfall comes from the 2,052-byte row-pointer overhead).

---

## Summary Table

| Metric | Dense | Sparse CSR (s = 0.9) | Ratio |
|--------|------:|---------------------:|------:|
| FLOPs | 524,288 | 52,429 | **10.0×** compute savings |
| Memory | 1,048,576 B | 211,767 B | **4.95× memory savings** |
| Transfer time (320 GB/s) | 3.28 µs | 0.662 µs | **4.95×** speedup |

> At 90% sparsity a memory-bandwidth-limited system achieves **~5× end-to-end speedup**.
> The compute savings (10×) exceed the memory savings (5×) because CSR eliminates
> FLOPs proportional to nnz but still stores 8 bytes per non-zero (vs 4 bytes dense),
> halving the memory reduction relative to the FLOP reduction.

---

## Breakeven Summary

| Breakeven condition | Sparsity s |
|---------------------|-----------|
| 2× FLOPs speedup | **50.00%** |
| CSR memory = Dense memory | **50.10%** |
| Any benefit over dense | **> 50.10%** |
