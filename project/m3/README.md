# M3 — Integration, Co-Simulation, and Synthesis

**ECE 410 MobileNetV2 INT8 Depthwise Conv Accelerator**

## Directory Map

### RTL

| File | Description |
|------|-------------|
| `rtl/top.sv` | Top-level integration wrapper; instantiates `axi_stream_ctrl` (M2) which internally contains `compute_core` (M2); exposes all host-facing ports — clk/rst/irq, full AXI4-Lite slave, and three AXI4-Stream channels (pixel in, weight in, result out) |

### Testbench

| File | Description |
|------|-------------|
| `tb/tb_top.sv` | End-to-end SystemVerilog testbench; communicates with `top` exclusively through host-facing AXI4-Lite and AXI4-Stream ports; loads 3×3 all-ones kernel, streams 3×3 pixel patch (values 1–9), verifies output = 0 (sum 45 >> QUANT_SHIFT 8), confirms STATUS DONE bit via AXI4-Lite read |

### Simulation Outputs

| File | Description |
|------|-------------|
| `sim/tb_top.vvp` | Icarus Verilog compiled simulation binary |
| `sim/tb_top.vcd` | Value change dump — all signals in tb_top hierarchy |
| `sim/cosim_run.log` | Simulation log; records phase-by-phase progress and PASS/FAIL per check |
| `sim/cosim_waveform.png` | Matplotlib waveform rendered from VCD; annotates Region A (host write + weight load), Region B (pixel streaming / compute), Region C (result valid) |
| `sim/gen_waveform.py` | Python script that parses the VCD and generates the PNG using matplotlib |

### Synthesis

| File | Description |
|------|-------------|
| `synth/area_report.txt` | Yosys cell and area statistics (copied from cf07 M2 baseline: 1620 cells, 4131 µm²) |
| `synth/timing_report.txt` | ABC stime timing report; critical path 5.86 ns, slack +4.14 ns at 100 MHz |
| `synth/openlane_run.log` | Yosys synthesis log (cf07 baseline); OpenLane 2 was unavailable (see note below) |
| `synth/config.json` | Synthesis configuration: design name `top`, clock period 10 ns, source file paths, tool/library metadata |
| `synth/power_report.txt` | Power estimation note; OpenLane power analysis unavailable; academic-model estimate ~2.63 mW @ 100 MHz provided |
| `synth/critical_path.md` | Prose analysis of the 21-stage critical path through the INT8 MAC tree; start/end registers, logic stages, and two shortening strategies |

### Documentation

| File | Description |
|------|-------------|
| `synthesis_notes.md` | Full integration and synthesis write-up (≥600 words): modules integrated, interface-to-core connection, co-sim results, Yosys fallback rationale, synthesis numbers, scope adjustment, M4 plans |
| `README.md` | This file |

## Simulator

- **Tool**: Icarus Verilog 12.0 (stable)
- **Environment**: MSYS2 MinGW64 on Windows 11 Home (10.0.26200)
- **Compile command**:
  ```
  iverilog -g2012 -o project/m3/sim/tb_top.vvp \
    project/m2/rtl/compute_core.sv \
    project/m2/rtl/interface.sv \
    project/m3/rtl/top.sv \
    project/m3/tb/tb_top.sv
  ```
- **Run command**: `vvp project/m3/sim/tb_top.vvp`
- **Result**: `PASS 3  FAIL 0  STATUS: PASS`

## Synthesis Tool

- **Tool**: Yosys 0.52 + ABC
- **Library**: generic_130nm (130 nm educational liberty)
- **Key results**: 1549 cells, 4060 µm², critical path 5.86 ns, slack +4.14 ns

## OpenLane 2 Note

OpenLane 2 installation failed on the host due to the `klayout` Python binding
not supporting Python 3.14, and WSL2 kernel restrictions prevented the Docker
image approach.  Yosys 0.52 was used as a synthesis fallback.  Full physical
design (GDS, DRC, LVS, signoff power) is deferred to M4 on a Linux host.
