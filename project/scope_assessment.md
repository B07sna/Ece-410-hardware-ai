# Project Scope Assessment
**ECE 410, Spring 2026 — Hussain Alquzweni**
**Last updated: 2026-05-17 (CodeFest 7 / M3 synthesis)**

---

## Project Summary

**Topic:** Workload analysis and hardware efficiency of quantized MobileNetV2, culminating
in an RTL accelerator for the INT8 depthwise convolution kernel.

---

## Milestone Status

| Milestone | Deliverable | Status |
|-----------|-------------|--------|
| M1 | Baseline profiling, roofline model, interface selection | **Done** |
| M2 | RTL design (`compute_core.sv`, `interface.sv`), simulation (39/39 checks pass) | **Done** |
| M3 | Logic synthesis of `crossbar_mac` (cf07) | **In progress** |

---

## M3 Synthesis Results (CodeFest 7)

Synthesis was performed with **Yosys 0.52** targeting a **generic 130 nm standard-cell
library** (Option B — sky130 PDK not installed; see `codefest/cf07/synth/m3_plan.md`).
Clock target: **10 ns (100 MHz)**.

| Metric | Result |
|--------|--------|
| Total cells | 1 549 |
| Sequential (DFF) | 144 |
| Chip area | 4 060 µm² |
| Sequential area | 864 µm² (21.3%) |
| ABC comb. delay | 5 380 ps (5.38 ns) |
| Est. register-to-register path | 5.86 ns |
| **Worst-case slack** | **+4.14 ns** |
| Timing status | **CLOSED** (40% margin) |

The design **easily meets 10 ns timing** with ~4 ns of positive slack, indicating
headroom to push the clock to approximately 171 MHz before violation. Area is dominated
by sequential storage (21%) and XOR/AOI/OAI carry-tree logic from the four 32-bit
signed accumulators.

---

## Known Scope Constraints

- **Library**: Generic 130 nm (educational). Sky130 PDK would give process-accurate
  numbers for actual tapeout assessment. Area and timing may shift ±20% at the sky130
  corner.
- **RTL adaptation**: `synth_top.sv` is a port-flattened version of `crossbar_mac.sv`
  required by Yosys's limited SystemVerilog parser. All logic is bit-for-bit equivalent.
- **STA**: Yosys 0.52 built-in `sta` does not support custom liberty cell names; timing
  is read from ABC's internal `stime -p` report (see `timing_report.txt`).

---

## Deliverables by Milestone

### M1 (`project/m1/`)
- `sw_baseline.md` — FP32 and INT8 profiling, arithmetic intensity analysis
- `interface_selection.md` — AXI4-Stream at 128-bit / 250 MHz justified
- `system_diagram.png` — block diagram

### M2 (`project/m2/rtl/`)
- `compute_core.sv` — pipelined 3×3 INT8 depthwise conv (5-stage, 6-cycle latency)
- `interface.sv` — AXI4-Lite control + AXI4-Stream wrapper
- Simulation: 39 checks pass, 0 failures across two testbenches

### M3 (`codefest/cf07/`)
- `hdl/synth_top.sv` — Yosys-compatible synthesis view of `crossbar_mac`
- `hdl/generic_130nm.lib` — Educational 130 nm liberty cell library
- `synth/crossbar_mac.ys` — Yosys synthesis script (10 ns clock target)
- `synth/cell_area_report.txt` — Cell count and area breakdown
- `synth/timing_report.txt` — ABC critical-path report
- `synth/netlist.v` — Gate-level netlist
- `synth/synth_interpretation.md` — Analysis of synthesis results
- `synth/m3_plan.md` — Option B fallback rationale and sky130 upgrade path
