# GEMM Roofline Analysis — RTX 3070 Ti

## Hardware Roofline (RTX 3070 Ti)
| Bound | Peak |
|---|---|
| Compute | 21,750 GFLOPS (FP32) |
| Memory bandwidth | 608 GB/s |
| Ridge point | 35.8 FLOP/byte |

## Measured Results
| Kernel | AI (FLOP/byte) | GFLOPS | % of roof |
|---|---|---|---|
| gemm_naive | 0.25 | 1,315 | 865% of memory slope |
| gemm_tiled | 256.0 | 1,325 | 6.1% of compute peak |

## Analysis

**Why naive is memory-bound in theory.** The naive kernel provides no data reuse: each of the N² output elements reloads its entire A-row and B-column from DRAM, yielding arithmetic intensity AI = 1/4 = 0.25 FLOP/byte — 143× below the RTX 3070 Ti's ridge point of 35.8 FLOP/byte. At this intensity, peak achievable performance is capped at 0.25 × 608 = 152 GFLOPS, far below the 21,750 GFLOPS compute ceiling. The roofline model firmly classifies naive as memory-bound.

**How tiling reduces DRAM traffic.** With T = 8 shared-memory tiling, each element of A and B is loaded from DRAM exactly once and reused N = 1024 times from fast SRAM registers. This collapses DRAM traffic from 2N³ × 4 B (8 GB) to 2N² × 4 B (8 MB) — a 1024× reduction — pushing AI to N/4 = 256 FLOP/byte, well into the compute-bound region.

**Why both kernels show similar performance (~1,320 GFLOPS).** Despite the 1024× DRAM traffic gap, the RTX 3070 Ti's 6 MB L2 cache absorbs naive's repeated element loads at N = 1024 (working set ≈ 12 MB, partially resident). Effective bandwidth reads as ~5,261 GB/s — L2 cache bandwidth, not DRAM. Both kernels are therefore L2/compute-bound at this matrix size; the DRAM savings from tiling only become decisive when N exceeds L2 capacity (~N ≥ 4096), where naive would thrash DRAM at 608 GB/s while tiled remains compute-limited.
