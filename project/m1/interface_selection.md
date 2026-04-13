# Interface Selection — MobileNetV2 Depthwise Conv Accelerator Chiplet

**Project:** Workload Analysis and Hardware Efficiency of Quantized MobileNetV2
**Author:** Hussain Alquzweni — ECE 410, Spring 2026
**Milestone:** M1 — Interface Selection
**Date:** 2026-04-12

---

## 1. Overview

This document selects and justifies the on-chip communication interface between the
host platform and the MobileNetV2 depthwise convolution accelerator chiplet. The
selection is driven by the measured bandwidth demand of the dominant kernel (depthwise
Conv2d `[1, 384, 14, 14] × [384, 1, 3, 3]`) at the software baseline throughput of
125 inferences/sec.

**Selected interface:** AXI4-Stream
**Host platform:** FPGA SoC (e.g., Xilinx Zynq UltraScale+ ZU9EG)

---

## 2. Dominant Kernel Data Profile

From the Codefest 2 roofline analysis (`cf02/analysis/ai_calculation.md`):

| Parameter | Value | Source |
|---|---|---|
| FLOPs per inference | 1,354,752 FLOPs | 2 × 384 × 9 × 196 |
| Bytes per inference | 615,936 bytes | weights + input + output activation |
| Arithmetic intensity | 2.20 FLOP/byte | FLOPs / bytes |
| Software baseline throughput | 125 inferences/sec | `sw_baseline.md` median |

### Byte Breakdown (FP32, no DRAM reuse)

```
Weights  : [384, 1, 3, 3]    = 3,456 elem  x 4 B =  13,824 B
Input    : [1, 384, 14, 14]  = 75,264 elem x 4 B = 301,056 B
Output   : [1, 384, 14, 14]  = 75,264 elem x 4 B = 301,056 B
                                               Total = 615,936 B
```

---

## 3. Required Interface Bandwidth

### 3.1 Sustained Bandwidth (throughput-driven)

The interface must sustain data delivery at the target inference rate.

```
Formula:
    BW_required = Bytes_per_inference x Inferences_per_sec

Substituting:
    BW_required = 615,936 B  x  125 inferences/sec
                = 76,992,000 B/s
                = 73.5 MB/s
                = 0.588 Gbps
```

### 3.2 Peak Burst Bandwidth (latency-driven)

For a fully pipelined accelerator, all input data (weights + input activation) must
be delivered before the output can be returned. Allocating 20% of the 8 ms inference
budget (1.6 ms) to data transfer sets the burst requirement:

```
Input bytes   = weights + input activation = 13,824 + 301,056 = 314,880 B
Transfer time = 20% x (1 / 125 s) = 0.20 x 8 ms = 1.6 ms

BW_burst = 314,880 B / 0.0016 s
         = 196,800,000 B/s
         = 187.8 MB/s
         = 1.50 Gbps
```

### 3.3 Summary of Bandwidth Requirements

| Mode | Required BW |
|---|---|
| Sustained (average) | 73.5 MB/s  (0.588 Gbps) |
| Peak burst (1.6 ms window) | 187.8 MB/s (1.504 Gbps) |

---

## 4. AXI4-Stream Specification

AXI4-Stream (Advanced eXtensible Interface 4, Streaming) is an ARM AMBA protocol
for unidirectional, burst-optimized data transfer with no addressing overhead.

### 4.1 Protocol Characteristics

| Property | Value / Description |
|---|---|
| Direction | Unidirectional (one channel per direction) |
| Addressing | None — purely streaming, zero address-phase overhead |
| Handshaking | TVALID / TREADY flow control |
| Data width | Configurable: 8, 16, 32, 64, 128, 256, 512 bits |
| Burst length | Unlimited (back-to-back beats) |
| Side-channel signals | TKEEP (byte enables), TLAST (frame boundary), TUSER |
| Clock | Source-synchronous, single clock domain per link |
| Latency | 1 clock cycle per beat (pipelined) |

### 4.2 Target Configuration (Zynq UltraScale+ HP AXI-Stream DMA)

| Parameter | Value | Rationale |
|---|---|---|
| Data bus width | 128 bits (16 bytes) | Standard HP port width on ZU9EG |
| Clock frequency | 250 MHz | Conservative PL fabric target |
| Peak throughput | 128 bits x 250 MHz = **4.0 GB/s = 32.0 Gbps** | |
| DMA engine | Xilinx AXI DMA (scatter-gather mode) | Zero-copy host transfers |
| Back-pressure | TREADY stall | Accelerator signals readiness |

### 4.3 Rated vs Required Bandwidth Comparison

```
AXI4-Stream rated BW  =  128 bits x 250 MHz
                       =  16 B x 250,000,000
                       =  4,000,000,000 B/s
                       =  4.0 GB/s  =  32.0 Gbps

Sustained requirement  =     73.5 MB/s  =   0.588 Gbps
Peak burst requirement =    187.8 MB/s  =   1.504 Gbps
```

| Mode | Required | Available | Headroom | Utilization |
|---|---|---|---|---|
| Sustained | 0.588 Gbps | 32.0 Gbps | 31.4 Gbps | **1.84%** |
| Peak burst | 1.504 Gbps | 32.0 Gbps | 30.5 Gbps | **4.70%** |

The AXI4-Stream interface at 128-bit / 250 MHz provides **54× headroom** over sustained
demand and **21× headroom** over peak burst demand. The interface will not be the
bottleneck under any realistic operating condition for this kernel.

---

## 5. Interface Selection Rationale

### 5.1 Why AXI4-Stream Over Alternatives

| Interface | Peak BW | Addressing | Latency | Verdict |
|---|---|---|---|---|
| **AXI4-Stream (128b/250MHz)** | **4.0 GB/s** | None (streaming) | 1 cycle/beat | **Selected** |
| AXI4 Full (memory-mapped) | 4.0 GB/s | Full address phase | 2–5 cycle overhead/txn | Overkill for streaming |
| AXI4-Lite | ~200 MB/s | 32-bit registers | High per-word overhead | Too slow for bulk data |
| PCIe Gen3 x4 | 3.94 GB/s | Packet-based | uS-level TLP latency | Wrong domain (off-chip) |
| APB | ~50 MB/s | Slow register bus | High overhead | Far too slow |

**AXI4-Stream is selected** because:

1. **Zero address overhead.** The depthwise conv kernel transfers three contiguous
   tensors (weights, input, output) in three sequential bursts. No random access is
   needed — AXI4 Full's address phase would add latency with no benefit.

2. **Back-pressure support.** The TREADY signal allows the accelerator to assert
   flow control when its internal SRAM buffer is full, preventing data loss without
   additional buffering logic.

3. **Native FPGA SoC support.** Xilinx Zynq UltraScale+ exposes AXI4-Stream via
   the AXI DMA IP core, connecting the PS (ARM Cortex-A53 host) to PL (accelerator)
   with scatter-gather DMA and zero-copy operation from DDR4.

4. **Scalability.** If INT8 quantization halves the data width (bytes per tensor
   cut in half), throughput can double on the same interface by narrowing the transfer
   window — no interface redesign required.

5. **Sufficient bandwidth with large margin.** At 1.84% sustained utilization, the
   interface supports 54× scale-up before becoming a bottleneck (e.g., batching,
   multi-kernel pipelining, or higher frame rates).

### 5.2 Host Platform: FPGA SoC

| Component | Specification |
|---|---|
| Platform | Xilinx Zynq UltraScale+ ZU9EG (representative) |
| Host processor | ARM Cortex-A53 (quad-core, 1.3 GHz) in Processing System (PS) |
| Accelerator fabric | Programmable Logic (PL) — FPGA fabric |
| PS-PL link | AXI4-Stream via AXI DMA over HP (High-Performance) port |
| DDR4 controller | In PS, shared between host and DMA engine |
| Host DDR4 BW | Up to 19.2 GB/s (PS-side) |

The FPGA SoC is the appropriate host because:
- The PL fabric can implement the depthwise MAC array and on-chip SRAM buffer directly
- The PS-PL AXI4-Stream HP ports are rated for 4+ GB/s, far exceeding the 187.8 MB/s
  burst requirement
- The ARM host runs the PyTorch-based software stack (BN, ReLU6, classifier) that
  remains on the CPU per the HW/SW partition plan

---

## 6. Interface-Bound Analysis

The accelerator is **not interface-bound** at any operating point in the design envelope.

```
Interface bound condition:
    BW_required >= BW_rated
    187.8 MB/s  << 4,000 MB/s          (not interface-bound)

Throughput to become interface-bound:
    Max inferences/sec = BW_rated / Bytes_per_inference
                       = 4,000,000,000 / 615,936
                       = 6,494 inferences/sec

    Margin = 6,494 / 125 = 51.9x       (51.9x above baseline before interface limits)
```

The interface becomes the bottleneck only if throughput exceeds **6,494 inferences/sec**
— 52× the software baseline. At that point, wider data bus (256-bit) or dual AXI4-Stream
channels would be the appropriate upgrade path.

---

## 7. Signal-Level Summary

### Host-to-Accelerator (input stream)

| Signal | Width | Description |
|---|---|---|
| `s_axis_tdata` | 128 bits | Weight / input activation data |
| `s_axis_tvalid` | 1 bit | Data valid (asserted by DMA) |
| `s_axis_tready` | 1 bit | Accelerator ready to accept |
| `s_axis_tlast` | 1 bit | End of tensor frame |
| `s_axis_tkeep` | 16 bits | Byte-lane enables |

### Accelerator-to-Host (output stream)

| Signal | Width | Description |
|---|---|---|
| `m_axis_tdata` | 128 bits | Output activation data |
| `m_axis_tvalid` | 1 bit | Data valid (asserted by accelerator) |
| `m_axis_tready` | 1 bit | DMA ready to accept |
| `m_axis_tlast` | 1 bit | End of output tensor |
| `m_axis_tkeep` | 16 bits | Byte-lane enables |

---

## 8. Conclusion

| Decision | Selection |
|---|---|
| **Interface** | AXI4-Stream |
| **Data bus width** | 128 bits |
| **Clock** | 250 MHz |
| **Rated bandwidth** | 4.0 GB/s (32.0 Gbps) |
| **Required sustained BW** | 73.5 MB/s (0.588 Gbps) |
| **Required peak burst BW** | 187.8 MB/s (1.504 Gbps) |
| **Interface utilization** | 1.84% sustained / 4.70% burst |
| **Interface-bound margin** | 51.9x above baseline throughput |
| **Host platform** | FPGA SoC (Xilinx Zynq UltraScale+) |
| **Bottleneck** | Compute / on-chip SRAM bandwidth — NOT the chiplet interface |

AXI4-Stream provides a clean, low-overhead, back-pressure-capable streaming link that
fully covers the bandwidth envelope of the depthwise conv accelerator while leaving
ample headroom for quantization scaling, batch size increases, or multi-kernel
pipelining in future design iterations.
