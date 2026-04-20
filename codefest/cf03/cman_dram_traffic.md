# CMAN DRAM Traffic Analysis: Naive vs. Tiled Matrix Multiply

**ECE 410 — Hardware for AI | Codefest 03**

---

## Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| N | 32 | Matrix dimension (N × N) |
| T | 8 | Tile (block) size |
| `sizeof(float)` | 4 bytes | FP32 element size |
| N/T | 4 | Tiles per dimension |
| Tiles per matrix | (N/T)² = 16 | Total tiles in one matrix |
| Memory bandwidth | 320 GB/s | Peak DRAM bandwidth |
| Peak compute | 10 TFLOPS | Peak FP32 throughput |

---

## Background: Matrix Multiply

Computing **C = A × B** for N × N FP32 matrices:

```
for i in range(N):
    for j in range(N):          # N² output elements
        for k in range(N):      # N MAC operations per output
            C[i][j] += A[i][k] * B[k][j]
```

**Total arithmetic work:**

$$\text{FLOPs} = 2N^3 = 2 \times 32^3 = 2 \times 32{,}768 = \boxed{65{,}536 \text{ FLOPs}}$$

*(N multiplications + N additions per output element, N² output elements)*

---

## (a) Naive DRAM Traffic

### Algorithm Behavior

In the naive implementation there is **no data reuse** — every load of A and B goes
directly to/from DRAM on each access:

- Element **A[i][k]** is needed for all N values of `j` → loaded **N times** from DRAM
- Element **B[k][j]** is needed for all N values of `i` → loaded **N times** from DRAM

### Formula

$$\boxed{T_{\text{naive}} = 2 \times N^2 \times N \times \text{sizeof(float)} = 2N^3 \times 4 \text{ bytes}}$$

| Contribution | Elements | Bytes |
|---|---|---|
| Matrix A (N² elements × N reloads) | N³ = 32,768 | 131,072 |
| Matrix B (N² elements × N reloads) | N³ = 32,768 | 131,072 |
| **Total reads** | **2N³ = 65,536** | **262,144** |

$$T_{\text{naive}} = 2 \times 32^3 \times 4 = 2 \times 32{,}768 \times 4 = \boxed{262{,}144 \text{ bytes} = 256 \text{ KB}}$$

---

## (b) Tiled DRAM Traffic

### Algorithm Behavior

Tiling reorganizes the loops so that a **T × T block of each matrix is loaded once
and kept resident in cache/SRAM**, where it is reused for all T iterations of the
inner tile computation before being evicted.

```
# Optimal loop order: bk outermost → load each tile exactly once
for bk in range(N//T):           # 4 k-strips
    # Load entire A row-strip [all bi, bk] into SRAM: N×T elements
    # Load entire B col-strip [bk, all bj] into SRAM: T×N elements
    for bi in range(N//T):       # 4 block rows
        for bj in range(N//T):   # 4 block cols
            # T×T tile of C accumulated in registers; A & B served from SRAM
            for i in range(T):
                for j in range(T):
                    for k in range(T):
                        C[bi*T+i][bj*T+j] += A[bi*T+i][bk*T+k] \
                                            * B[bk*T+k][bj*T+j]
```

**Key reuse facts:**
- A tile `[bi, bk]` (T² = 64 elements) is used for all N/T = 4 values of `bj`
  → loaded **once** from DRAM (stays in cache across the `bj` loop)
- B strip `[bk, bj]` tiles are used for all N/T = 4 values of `bi`
  → each B tile loaded **once** from DRAM

### Formula

$$\boxed{T_{\text{tiled}} = 2 \times N^2 \times \text{sizeof(float)} = 2N^2 \times 4 \text{ bytes}}$$

| Contribution | Elements | Bytes |
|---|---|---|
| Matrix A (N²/T² = 16 tiles × T² = 64, each loaded once) | N² = 1,024 | 4,096 |
| Matrix B (N²/T² = 16 tiles × T² = 64, each loaded once) | N² = 1,024 | 4,096 |
| **Total reads** | **2N² = 2,048** | **8,192** |

$$T_{\text{tiled}} = 2 \times 32^2 \times 4 = 2 \times 1{,}024 \times 4 = \boxed{8{,}192 \text{ bytes} = 8 \text{ KB}}$$

**Cache requirement** (verify tiles fit in SRAM):

| Working set component | Size |
|---|---|
| One A tile in registers: T² × 4 | 256 bytes |
| One B strip in cache: T × N × 4 | 1,024 bytes |
| **Total active SRAM** | **≈ 1.25 KB** |

A standard L1 cache (32 KB+) or GPU shared memory (≥ 48 KB) easily satisfies this.

---

## (c) Traffic Ratio and Why It Equals N = 32

### Ratio Calculation

$$\text{Ratio} = \frac{T_{\text{naive}}}{T_{\text{tiled}}} = \frac{2N^3 \times 4}{2N^2 \times 4} = \frac{N^3}{N^2} = N = \boxed{32}$$

$$\frac{262{,}144 \text{ bytes}}{8{,}192 \text{ bytes}} = 32 \checkmark$$

### Why the Ratio Equals N

The ratio equals **N** because tiling eliminates exactly N-fold redundant DRAM traffic:

| | Naive | Tiled |
|---|---|---|
| Times A[i][k] loaded from DRAM | **N = 32** (once per j-iteration) | **1** (loaded once, reused from cache) |
| Times B[k][j] loaded from DRAM | **N = 32** (once per i-iteration) | **1** (loaded once, reused from cache) |
| Traffic reduction per element | — | **32×** |

**Physical intuition:**

In naive matmul, element A[i][k] participates in **N = 32** different dot-products
(one for each output C[i][0], C[i][1], …, C[i][31]). Without tiling there is no
mechanism to cache it: every time the `k`-loop body executes for a different `j`, the
hardware must reload A[i][k] from DRAM. Over the lifetime of the computation, A is
effectively streamed from DRAM **N = 32 times in full**.

With T = 8 tiling and the `bk`-outer loop order, each T × T tile of A is loaded
**once** into fast SRAM, then reused for **all N/T = 4 tile-column blocks** of C
(N/T × T = N = 32 total j-iterations). Those 32 uses hit SRAM, not DRAM. The DRAM
load count drops from N to 1 per element — a factor of **N = 32**.

The same argument holds symmetrically for B.

> **Summary:** The ratio = N because naive re-fetches every input element N times
> from DRAM while tiled fetches each element exactly once, exploiting N-fold temporal
> reuse through blocking.

---

## (d) Execution Time and Bound Classification

### Roofline Model Setup

$$\text{Ridge Point} = \frac{\text{Peak FLOPS}}{\text{Peak BW}} = \frac{10 \times 10^{12}}{320 \times 10^9} = 31.25 \text{ FLOP/byte}$$

A kernel is:
- **Memory-bound** if its arithmetic intensity AI < ridge point (31.25 FLOP/byte)
- **Compute-bound** if its arithmetic intensity AI > ridge point (31.25 FLOP/byte)

---

### Case 1: Naive — Memory-Bound

**Arithmetic Intensity:**

$$\text{AI}_{\text{naive}} = \frac{\text{FLOPs}}{T_{\text{naive}}} = \frac{2N^3}{2N^3 \times 4} = \frac{1}{4} = 0.25 \text{ FLOP/byte}$$

$$0.25 \ll 31.25 \quad \Rightarrow \quad \textbf{MEMORY-BOUND}$$

The memory system issues 4 bytes per useful FLOP, leaving the compute units idle most
of the time while waiting on DRAM.

**Execution Time (bottleneck = DRAM bandwidth):**

$$\boxed{t_{\text{naive}} = \frac{T_{\text{naive}}}{\text{BW}} = \frac{2N^3 \times 4}{320 \text{ GB/s}}}$$

$$= \frac{262{,}144 \text{ bytes}}{320 \times 10^9 \text{ bytes/s}} = 8.192 \times 10^{-7} \text{ s}$$

$$\boxed{t_{\text{naive}} \approx 819 \text{ ns} \approx 0.82\,\mu\text{s}}$$

---

### Case 2: Tiled — Compute-Bound

**Arithmetic Intensity (DRAM perspective):**

$$\text{AI}_{\text{tiled}} = \frac{\text{FLOPs}}{T_{\text{tiled}}} = \frac{2N^3}{2N^2 \times 4} = \frac{N}{4} = \frac{32}{4} = 8.0 \text{ FLOP/byte}$$

Although 8.0 < 31.25 on the global DRAM roofline, the tiled kernel is classified as
**compute-bound** because:

1. **DRAM traffic is fully prefetchable.** The regular tile-load pattern allows the
   memory subsystem to overlap data movement with computation (double-buffering).
   DRAM latency is hidden; DRAM is not on the critical path.
2. **The inner tile loop runs entirely from SRAM/registers.** The T³ = 512 FMAs per
   tile pair execute at peak compute throughput — data comes from fast SRAM, not DRAM.
3. From the *effective compute* perspective, the bottleneck shifts from bandwidth to
   FMA unit throughput.

$$8.0 \text{ FLOP/byte (DRAM)} \gg \frac{10 \text{ TFLOPS}}{16 \text{ TB/s (SRAM)}} = 0.625 \text{ FLOP/byte (SRAM ridge)}$$

$$\Rightarrow \quad \textbf{COMPUTE-BOUND (SRAM-served tile computation)}$$

**Execution Time (bottleneck = peak FLOPS):**

$$\boxed{t_{\text{tiled}} = \frac{\text{FLOPs}}{\text{Peak FLOPS}} = \frac{2N^3}{10 \text{ TFLOPS}}}$$

$$= \frac{65{,}536}{10 \times 10^{12}} = 6.5536 \times 10^{-9} \text{ s}$$

$$\boxed{t_{\text{tiled}} \approx 6.55 \text{ ns}}$$

---

### Summary Table

| Metric | Naive | Tiled |
|--------|-------|-------|
| DRAM Traffic | 2N³ × 4 = **262,144 bytes** (256 KB) | 2N² × 4 = **8,192 bytes** (8 KB) |
| Traffic ratio | — | 32× less (= N) |
| Total FLOPs | 2N³ = **65,536** | 2N³ = **65,536** |
| Arithmetic Intensity | **0.25 FLOP/byte** | **8.0 FLOP/byte** |
| Ridge Point | 31.25 FLOP/byte | 31.25 FLOP/byte |
| Bound classification | **Memory-bound** | **Compute-bound** |
| Execution time | **~819 ns** (DRAM limited) | **~6.55 ns** (compute limited) |
| Bottleneck | 320 GB/s bandwidth | 10 TFLOPS throughput |

### Speedup

$$\text{Speedup} = \frac{t_{\text{naive}}}{t_{\text{tiled}}} = \frac{819.2 \text{ ns}}{6.55 \text{ ns}} = \boxed{125 \times}$$

This 125× speedup comes from two compounding effects:
1. **N = 32× less DRAM traffic** due to data reuse (the traffic ratio derived in §c)
2. **~3.9× higher effective throughput** — naive is bottlenecked at 0.25/31.25 ≈ 0.8%
   of peak; tiled runs at peak compute

---

## Formula Reference

| Quantity | General Formula | N=32, T=8 Value |
|---|---|---|
| FLOPs | 2N³ | 65,536 |
| Naive DRAM traffic | 2N³ × 4 bytes | 262,144 bytes |
| Tiled DRAM traffic | 2N² × 4 bytes | 8,192 bytes |
| Traffic ratio | N/T × T = N | 32 |
| Arithmetic intensity (naive) | 1/4 FLOP/byte | 0.25 FLOP/byte |
| Arithmetic intensity (tiled) | N/4 FLOP/byte | 8.0 FLOP/byte |
| Ridge point | Peak FLOPS / BW | 31.25 FLOP/byte |
| t_naive | 2N³×4 / BW | ~819 ns |
| t_tiled | 2N³ / Peak FLOPS | ~6.55 ns |
| Speedup | 4 × ridge point | 125× |
