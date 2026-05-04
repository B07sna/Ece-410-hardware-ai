# M2 — MobileNetV2 INT8 Depthwise Conv Accelerator

**Course:** ECE 410, Spring 2026  
**Author:** Hussain Alquzweni  
**Milestone:** M2 — RTL Design and Simulation  

---

## Overview

This milestone delivers a synthesizable SystemVerilog RTL implementation of a
weight-stationary INT8 depthwise convolution accelerator targeting the dominant
MobileNetV2 kernel identified in M1 (depthwise Conv2d `[1, 384, 14, 14]`,
arithmetic intensity 2.20 FLOP/byte). The design includes two simulation-verified
RTL modules, two self-checking testbenches, a cycle-accurate waveform, and a
quantization precision analysis.

---

## Repository Layout

```
project/m2/
├── rtl/
│   ├── compute_core.sv      # Pipelined 3×3 INT8 depthwise conv engine
│   └── interface.sv         # AXI4-Lite control + AXI4-Stream wrapper
├── tb/
│   ├── tb_compute_core.sv   # Testbench: compute_core (16-output 6×6 image)
│   └── tb_interface.sv      # Testbench: axi_stream_ctrl end-to-end + reg checks
├── sim/
│   ├── compute_core_tb.vvp  # Compiled sim binary (compute_core)
│   ├── compute_core_run.log # Simulation output log (compute_core)
│   ├── interface_tb.vvp     # Compiled sim binary (interface)
│   ├── interface_run.log    # Simulation output log (interface)
│   ├── gen_waveform.py      # Waveform generator script
│   └── waveform.png         # Cycle-accurate AXI4-Stream waveform
└── precision.md             # INT8 quantization error analysis
```

---

## Prerequisites

### Simulator — Icarus Verilog 12.0

Installed via MSYS2 MinGW-w64 package manager.

| Item | Value |
|------|-------|
| Tool | Icarus Verilog |
| Version | **12.0 (stable)** |
| Distribution | MSYS2 MinGW-w64 |
| Install path | `C:\msys64\mingw64\bin\` |
| Binaries | `iverilog.exe`, `vvp.exe` |

Install (MSYS2 MinGW64 shell):
```
pacman -S mingw-w64-x86_64-iverilog
```

All commands below prepend `C:\msys64\mingw64\bin` to `PATH`. From a Git Bash
or MSYS2 shell, set the path once per session:

```bash
export PATH="/c/msys64/mingw64/bin:$PATH"
```

Verify:
```bash
iverilog -V
# Icarus Verilog version 12.0 (stable)
```

### Python — 3.12.3

Required only for waveform generation (`gen_waveform.py`).

| Package | Version |
|---------|---------|
| Python | **3.12.3** |
| matplotlib | **3.10.8** |
| numpy | **2.4.4** |

Install packages:
```
py -m pip install matplotlib numpy
```

---

## Reproduction Steps

All commands assume the working directory is `project/m2/`.

### 1. Compile and run `tb_compute_core`

This testbench instantiates `compute_core` directly, streams a 6×6 INT8
ramp image through the 3×3 Gaussian kernel (weights {1,2,1,2,4,2,1,2,1},
QUANT_SHIFT=4), and compares all 16 output pixels against an independent
integer reference model.

**Compile:**
```bash
iverilog -g2012 \
  -o sim/compute_core_tb.vvp \
  rtl/compute_core.sv \
  tb/tb_compute_core.sv
```

**Run:**
```bash
vvp sim/compute_core_tb.vvp
```

**Expected terminal output:**
```
=== tb_compute_core ===
PASS 16 / 16
FAIL 0 / 16
STATUS: PASS
```

**Log written to:** `sim/compute_core_run.log`

Expected log excerpt:
```
=== Summary ===
Total outputs expected : 16
PASS                  : 16
FAIL                  : 0
STATUS: PASS
```

---

### 2. Compile and run `tb_interface`

This testbench instantiates `axi_stream_ctrl` (which internally wraps
`compute_core`) and exercises seven categories of checks:

| # | Check |
|---|-------|
| 1 | AXI4-Lite write — QUANT_CFG (0x10) = 4; verify BRESP = OKAY |
| 2 | AXI4-Lite read  — STATUS (0x04) after reset = 0x00000001 (IDLE) |
| 3 | AXI4-Lite read  — IMG_CFG (0x08) after reset = 0x00060006 ({H=6, W=6}) |
| 4 | Full 6×6 inference via AXI4-Stream — 16 output pixels vs reference |
| 5 | STATUS after inference = 0x00000004 (DONE, sticky) |
| 6 | IRQ asserts after writing IRQ_EN[0] = 1 (DONE_EN) |
| 7 | IRQ_STAT W1C clear → 0x00000000; IRQ deasserts |

**Compile:**
```bash
iverilog -g2012 \
  -o sim/interface_tb.vvp \
  rtl/compute_core.sv \
  rtl/interface.sv \
  tb/tb_interface.sv
```

**Run:**
```bash
vvp sim/interface_tb.vvp
```

**Expected terminal output:**
```
=== tb_interface ===
PASS 23  FAIL 0
STATUS: PASS
```

**Log written to:** `sim/interface_run.log`

Expected log excerpt:
```
=== Summary ===
Total checks : 23
PASS         : 23
FAIL         : 0
STATUS: PASS
```

---

### 3. Regenerate waveform PNG

Reconstructs the cycle-accurate AXI4-Stream waveform from the known
stimulus timeline of `tb_compute_core` and saves it as `sim/waveform.png`.

```bash
py sim/gen_waveform.py
# Saved waveform.png
```

The PNG shows 68 clock cycles covering: synchronous reset, 9-weight load,
36-pixel stream (6 rows × 6 columns, ramp 1–36), and the 16 pipelined
outputs in four 4-beat bursts separated by 2-cycle inter-row gaps. The
pipeline latency from first `win_valid` to first `result_tvalid` is
6 cycles (5 adder-tree stages + 1 output register stage).

---

## Design Summary

### `compute_core` (rtl/compute_core.sv)

| Parameter | Value |
|-----------|-------|
| Architecture | Weight-stationary, single-channel depthwise 3×3 |
| Precision | INT8 weights and activations (signed 8-bit) |
| Accumulator | 20-bit (ceil_log2(9) = 4 extra bits over 16-bit product) |
| Re-quantization | Arithmetic right-shift by `QUANT_SHIFT`; INT8 saturation |
| Pipeline depth | 5 register stages + 1 output register = 6-cycle latency |
| Interface | AXI4-Stream pixel slave, weight slave, result master |
| Backpressure | Full: `out_stall` freezes all pipeline stages simultaneously |

### `axi_stream_ctrl` (rtl/interface.sv)

| Feature | Detail |
|---------|--------|
| Register slave | AXI4-Lite, 8 × 32-bit registers at offsets 0x00–0x1C |
| Control FSM | IDLE → RUNNING → DONE → IDLE; START bit auto-clears |
| Status | IDLE[0], RUNNING[1], DONE[2], ERROR[3]; DONE is sticky |
| Interrupt | Level-sensitive; DONE_IRQ and ERR_IRQ; W1C clear via IRQ_STAT |
| Pixel gating | Pixel AXI-Stream forwarded to core only in FSM_RUNNING |
| Weight routing | Weight AXI-Stream always forwarded; core controls its own tready |

---

## Alignment with M1 Plan

M1 established two binding decisions that M2 implements without deviation:

**1. AXI4-Stream interface (from `m1/interface_selection.md`)**

M1 selected AXI4-Stream at 128-bit / 250 MHz for its zero-address-phase
overhead, native TREADY back-pressure, and 54× bandwidth headroom over the
187.8 MB/s peak burst requirement at 125 inferences/sec. M2 implements
AXI4-Stream pixel and weight slave ports and a result master port with
complete TVALID/TREADY/TLAST handshaking, consistent with the M1 protocol
specification. An AXI4-Lite register slave is added for software-controlled
configuration (START, image dimensions, quantization shift, interrupt
enable/status), which does not conflict with M1 — the streaming data path is
unchanged.

**2. INT8 quantization (motivated by M1 arithmetic intensity = 2.20 FLOP/byte)**

M1 measured the dominant depthwise Conv2d kernel at 2.20 FLOP/byte, firmly
in the memory-bandwidth-bound region of the i7-12700KF roofline (ridge point
10 FLOP/byte, compute utilization ≈ 22%). M1 projected that INT8 would raise
effective arithmetic intensity by 4× (to ≈ 8.80 FLOP/byte-equivalent) and
increase throughput proportionally. M2 implements INT8 exactly: 8-bit signed
weights and activations, a 20-bit accumulator, and arithmetic right-shift
re-quantization. The `precision.md` error analysis confirms the implementation
is correct — worst-case error vs FP32 is 1 LSB across 1,600 test outputs,
with MAE 0.478 LSB and SNR ≈ 33.8 dB.

**No parameters, interfaces, or architectural decisions were altered from the
M1 specification.**

---

## Simulation Results at a Glance

| Testbench | Checks | Pass | Fail |
|-----------|-------:|-----:|-----:|
| `tb_compute_core` | 16 | 16 | 0 |
| `tb_interface` | 23 | 23 | 0 |
| **Total** | **39** | **39** | **0** |
