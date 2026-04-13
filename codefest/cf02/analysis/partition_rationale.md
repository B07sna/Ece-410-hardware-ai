# HW/SW Partition Proposal — MobileNetV2 Quantization Accelerator

**Based on:** Roofline analysis (`ai_calculation.md`) and profiling (`project_profile.txt`)

---

## (a) Kernels to Accelerate in Hardware

The primary hardware target is the **depthwise Conv2d family** — specifically the
`[1, 384, 14, 14] × [384, 1, 3, 3]` kernel and its structural siblings at other spatial
resolutions (192@28×28, 960@7×7). Together these account for over 30% of total inference
Self CPU time. The roofline confirms they operate at **AI ≈ 2.20 FLOP/byte**, nearly 5×
below the i7-12700KF ridge point of 10 FLOP/byte — leaving compute utilization below 25%
of peak. Dedicated silicon can couple a wide on-chip SRAM array directly to a
depthwise MAC array, eliminating repeated DRAM round-trips and raising effective AI.
The high-FLOP pointwise (1×1) convolutions, which approach AI ~12–25 FLOP/byte, are
secondary accelerator candidates once the memory bottleneck is resolved.

## (b) Software Baseline

The CPU retains all non-bottleneck work: batch normalization, ReLU6, the final
1×1 expansion conv to 1280 channels, global average pooling, and the linear classifier.
These collectively consume under 15% of runtime and are not memory-bandwidth-constrained
at this batch size. The full PyTorch inference graph and quantization calibration pipeline
remain software-managed.

## (c) Required Interface Bandwidth

At AI = 2.20 FLOP/byte the kernel is throughput-limited by memory bandwidth, not compute.
To avoid the host-accelerator interface becoming the new bottleneck, it must sustain at
least **76.8 GB/s** — matching DRAM bandwidth. PCIe 4.0 ×16 (~64 GB/s) is insufficient;
the design should use on-chip HBM or a wide AXI bus to local SRAM.

## (d) Compute-Bound vs Memory-Bound — Before and After

On the i7-12700KF the kernel is **firmly memory-bound** (AI 2.20 vs ridge 10 FLOP/byte).
The proposed accelerator, operating at 2000 GFLOPS with on-chip weight caching, raises
effective AI toward ~50 FLOP/byte by reusing weights across the output spatial map,
shifting the operating point past the ridge into the **compute-bound regime**. This is
confirmed by the roofline plot: the accelerator point sits above the i7 compute ceiling,
meaning the bottleneck transitions from DRAM bandwidth to MAC throughput — exactly the
regime where added compute resources yield proportional throughput gains.
