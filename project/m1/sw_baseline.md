# Software Baseline Benchmark — MobileNetV2 FP32

**Project:** Workload Analysis and Hardware Efficiency of Quantized MobileNetV2
**Author:** Hussain Alquzweni — ECE 410, Spring 2026
**Milestone:** M1 — Software Baseline
**Date:** 2026-04-12

---

## 1. Platform Specifications

| Field | Value |
|---|---|
| **CPU** | 12th Gen Intel Core i7-12700KF |
| **Physical cores** | 12 (8 P-cores + 4 E-cores) |
| **Logical threads** | 20 |
| **Base / Boost clock** | 3.6 GHz / 5.0 GHz (P-core) |
| **L3 cache** | 25 MB |
| **RAM** | 15.8 GB DDR4 |
| **Peak compute (FP32)** | ~768 GFLOPS (AVX-512, all cores) |
| **Peak memory bandwidth** | 76.8 GB/s (DDR4-4800 dual-channel) |
| **GPU** | None used (CPU-only inference) |
| **OS** | Windows 11 (10.0.26200) |
| **Python** | 3.12.3 |
| **PyTorch** | 2.11.0+cpu |
| **Torchvision** | 0.26.0+cpu |
| **Backend** | oneDNN (MKL-DNN) via `aten::mkldnn_convolution` |

---

## 2. Benchmark Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV2 (torchvision, random weights) |
| Precision | FP32 |
| Batch size | 1 |
| Input shape | `[1, 3, 224, 224]` |
| Warmup runs | 5 (excluded from timing) |
| Timed runs | 10 |
| Timing method | `time.perf_counter()` wall-clock, per-inference |
| Memory method | `psutil.Process.memory_info().rss` (peak over all timed runs) |
| `torch.no_grad()` | Yes |

---

## 3. Latency Results

| Metric | Value |
|---|---|
| **Median latency** | **7.990 ms** |
| Mean latency | 7.683 ms |
| Std deviation | 0.639 ms |
| Min latency | 6.385 ms |
| Max latency | 8.151 ms |
| P90 latency | 8.134 ms |

### Per-Run Breakdown

| Run | Latency (ms) |
|---|---|
| 01 | 8.134 |
| 02 | 8.151 |
| 03 | 6.385 |
| 04 | 6.682 |
| 05 | 8.063 |
| 06 | 7.562 |
| 07 | 8.099 |
| 08 | 7.731 |
| 09 | 8.109 |
| 10 | 7.918 |

The bimodal distribution (runs 3–4 noticeably faster) reflects OS scheduler variability
and LLC eviction behavior on the first access of the activation buffer. The median is the
appropriate central estimate; mean is pulled down by those two fast runs.

---

## 4. Throughput

```
Throughput = Batch Size / Median Latency
           = 1 / (7.990 ms / 1000)
           = 1 / 0.007990 s
           = 125.1 samples / sec
```

| Metric | Value |
|---|---|
| **Throughput** | **125.1 samples/sec** |
| Batch size | 1 |
| Median latency | 7.990 ms |

---

## 5. Memory Usage

| Metric | Value |
|---|---|
| Model parameters | 3,504,872 |
| Model weight size (FP32) | 13.37 MB |
| Process RSS before timed runs | 310.3 MB |
| Peak RSS during timed runs | 310.3 MB |
| Delta RSS (activation overhead) | ~0 MB |

The near-zero delta RSS indicates activations are computed in-place or fit within
already-allocated buffers after warmup. The 310 MB baseline RSS includes the Python
runtime, PyTorch framework, and model weights loaded into memory.

---

## 6. Roofline Context

From the Codefest 2 analysis, the dominant kernel (depthwise Conv2d `[1,384,14,14]`)
has an arithmetic intensity of **2.20 FLOP/byte**, placing it firmly in the
**memory-bound** region of the i7-12700KF roofline (ridge point = 10 FLOP/byte).

| Metric | Value |
|---|---|
| Dominant kernel AI | 2.20 FLOP/byte |
| Ridge point | 10.0 FLOP/byte |
| Peak memory bandwidth | 76.8 GB/s |
| Attainable perf (dominant kernel) | ~169 GFLOPS |
| Peak compute ceiling | 768 GFLOPS |
| Compute utilization (dominant kernel) | ~22% |

The 125.1 samples/sec baseline is the FP32 reference that INT8 quantization will be
benchmarked against in M2. Expected outcome: INT8 should reduce weight and activation
data volume by 4×, raising effective AI and throughput significantly.

---

## 7. Files

| File | Description |
|---|---|
| `run_baseline.py` | Benchmark script (timing + memory measurement) |
| `baseline_results.json` | Raw numeric results (machine-readable) |
| `sw_baseline.md` | This document |
