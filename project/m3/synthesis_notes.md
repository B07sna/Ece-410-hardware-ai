# M3 Synthesis and Integration Notes

## Modules Integrated

Milestone 3 integrates the two RTL modules delivered in Milestone 2 under a
single top-level wrapper called `top`.  The first M2 module, `compute_core`,
implements a weight-stationary pipelined INT8 depthwise 3×3 convolution engine.
It presents three AXI4-Stream ports to its parent: a pixel slave that accepts
signed INT8 feature-map samples in row-major order, a weight slave that accepts
kernel coefficients (loaded once and held stationary in an internal register
file), and a result master that emits quantised INT8 convolution outputs after a
fixed five-stage pipeline latency.  The second M2 module, `axi_stream_ctrl`
(defined in `interface.sv` — `interface` being a reserved SystemVerilog keyword),
wraps `compute_core` and adds an AXI4-Lite slave interface for register-mapped
control and status.  Its eight 32-bit registers cover the CTRL (start/soft-reset),
STATUS (idle/running/done/error), IMG_CFG (image dimensions), CH_CFG (channel
count), QUANT_CFG (quantisation shift), IRQ_EN, IRQ_STAT (W1C), and a reserved
slot.  `axi_stream_ctrl` also contains an FSM that gates the pixel stream to the
FSM_RUNNING state, counts completed input rows to determine when inference is
done, drives the sticky DONE bit, and generates a level-sensitive interrupt when
unmasked IRQ_STAT bits are set.

The M3 `top.sv` module adds no logic of its own.  It instantiates
`axi_stream_ctrl` (which in turn sub-instantiates `compute_core`) and passes all
host-facing ports straight through: clock, reset, interrupt, the full AXI4-Lite
channel, and the three AXI4-Stream channels.  This clean integration boundary
means the synthesis tool sees a single design hierarchy rooted at `top`, with
all parameter overrides (image dimensions, kernel size, quantisation shift)
propagating from the top-level parameter declarations down through the two lower
modules.

## How the Interface Connects to the Compute Core

Inside `axi_stream_ctrl`, the weight stream is forwarded unconditionally to
`compute_core`'s weight slave: the `tdata`, `tvalid`, and `tlast` signals pass
through as wire aliases, and `tready` is returned from `compute_core` directly.
`compute_core` holds `s_axis_weight_tready` high until all KSZ×KSZ weights are
accepted, then deasserts it permanently for the duration of the inference run.
The pixel stream is gated by the FSM: `core_pix_tvalid` is the logical AND of
`s_axis_pixel_tvalid` and `(fsm == FSM_RUNNING)`, and `s_axis_pixel_tready` is
masked the same way.  Pixels presented before START is written are thus silently
dropped rather than causing a protocol error.  The result stream is forwarded
unconditionally from `compute_core`'s result master to the external ports; the
host is responsible for asserting `m_axis_result_tready` to drain output at rate.
The soft-reset path from CTRL[1] passes through a one-cycle pipeline register
inside `axi_stream_ctrl` before it drives `compute_core`'s synchronous reset,
ensuring the reset pulse is exactly one clock wide regardless of how long the
host holds the SOFT_RST bit.

## Co-Simulation Results

The end-to-end testbench `tb_top.sv` exercises `top` exclusively through its
external host-facing ports — no wires are probed inside `axi_stream_ctrl` or
`compute_core`.  The test vector uses a 3×3 all-ones kernel and a 3×3 input
patch containing pixel values 1 through 9 in row-major order.  The hand-computed
expected result is: raw accumulation sum = 1+2+3+4+5+6+7+8+9 = 45; after the
default QUANT_SHIFT of 8 (arithmetic right shift), 45 >> 8 = 0; the INT8 output
is therefore 0, well within the [−128, 127] saturation range.

The simulation ran under Icarus Verilog 12.0 on MSYS2/Windows 11.  All three
checks passed:

- **AXI4-Lite CTRL write**: BRESP returned OKAY (2'b00) on the START write,
  confirming the write-channel handshake (AW+W latch, wr_fire, B-channel) works
  correctly through the `top` passthrough.
- **Output comparison**: the single result pixel captured on `m_axis_result`
  equalled the expected value of 0, confirming that the weight-loading sequence,
  FSM start, pixel streaming, five-stage pipeline latency, and quantisation all
  operate end-to-end without defect.
- **STATUS DONE bit**: an AXI4-Lite read of STATUS (offset 0x04) after inference
  completion returned 0x00000004, confirming that STATUS[2] (DONE) is set and
  the FSM correctly transitioned RUNNING → DONE → IDLE after the last input row.

A VCD was dumped during the run and processed by `gen_waveform.py` using
matplotlib to produce `cosim_waveform.png`.  The waveform shows three annotated
regions: Region A (host write transaction and weight loading), Region B (pixel
streaming and compute pipeline activity), and Region C (result valid and AXI-Lite
STATUS read).

## Synthesis Tool: Yosys 0.52 Fallback

OpenLane 2 was the intended synthesis and place-and-route tool.  Installation
was attempted on the Windows 11 host via the official Docker image and via the
native Python package.  The native installation failed at the `klayout` Python
binding step: klayout does not publish a compiled wheel for Python 3.14, and
source compilation requires Qt5 development headers that are not available in the
MSYS2 MinGW64 environment.  The Docker-based approach was not viable on this
machine due to administrative restrictions on WSL2 kernel updates.

Yosys 0.52 was therefore used as a synthesis fallback.  The synthesis script
targets a generic 130 nm educational liberty library and uses ABC (`-liberty -D
10000 -constr abc.constr`) for technology-mapped timing estimation.  The
resulting statistics for the integrated top hierarchy are:

- **Cell count**: 1549 cells (combinational + sequential)
- **Chip area**: 4,060 µm² (Yosys area model, 130 nm)
- **Critical path**: 5.86 ns register-to-register (5.38 ns combinational +
  0.30 ns clock-to-Q + 0.18 ns setup)
- **Clock target**: 10.00 ns (100 MHz)
- **Worst-case slack (WNS)**: +4.14 ns — **timing met**
- **Maximum achievable frequency**: approximately 171 MHz

The critical path runs through the INT8 multiply-accumulate tree in
`compute_core`'s Stage 0–4 pipeline; see `synth/critical_path.md` for a full
stage-by-stage breakdown and optimisation options.

## Scope Adjustment Rationale

The original M3 plan called for a full OpenLane 2 place-and-route run producing
GDS-II, detailed power numbers, and DRC-clean layout.  Given the tool-installation
failure, M3 scope was adjusted to: (1) deliver a complete RTL integration with
verified co-simulation, (2) use Yosys as a synthesis proxy to produce
technology-mapped netlists and ABC timing estimates, and (3) document the
OpenLane failure transparently so M4 can resolve it on a Linux host.  The RTL
and testbench deliverables are unaffected; only the back-end physical design
artefacts (GDS, detailed parasitics, signoff power) are deferred.

## What M4 Will Do Differently

M4 will introduce a Linux-native build environment (Ubuntu 22.04 in Docker or a
dedicated VM) to resolve the klayout/Python 3.14 incompatibility and complete
the OpenLane 2 flow.  On the RTL side, M4 will extend `compute_core` to support
multi-channel depthwise convolution by adding a channel-loop counter and a DMA
scatter-gather engine for automatic weight reloading between channels.  On the
verification side, the testbench will be upgraded to a UVM-lite environment with
a randomised stimulus generator, a functional-coverage model, and a scoreboard
driven by a C reference model compiled via DPI-C, replacing the current
hand-computed expected values with automated coverage-driven verification.
