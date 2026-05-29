# Synthesis Interpretation — crossbar_mac
**ECE 410 · CodeFest 7 · Hussain Alquzweni**

**Tool:** Yosys 0.52 · generic 130 nm library | **Clock:** 10 ns (100 MHz)

---

## Clock Period and Worst-Case Slack

The synthesis targeted a **10 ns clock**. The register-to-register critical path is **5.86 ns**
(ABC combinational delay 5.38 ns + DFFR Clk→Q 0.30 ns + DFF setup 0.18 ns), giving a
**worst-case slack of +4.14 ns** — timing met with 41% margin. Maximum achievable frequency
is ~171 MHz (1 / 5.86 ns).

## Critical Path

Source register: `in0[1]` (bit 1 of activation input, held in a DFFR).
Sink register: D input of the output-register DFF for column 0.
The 21-stage path traverses a Brent-Kung carry prefix tree; dominant cell types are
**XNOR2** (sum bits), **AOI21 / OAI21** (carry generate / propagate), and **MUX2**
(weight-gated add/subtract select).

## Total Cell Area and Top Three Contributors

Total mapped: **1,549 cells** over **4,060 µm²**.

| Rank | Cell | Count | Area (µm²) | % of total |
|------|------|------:|----------:|----------:|
| 1 | DFF | 144 | 864 | 21.3% |
| 2 | XNOR2 | 159 | 557 | 13.7% |
| 3 | AOI21 | 217 | 543 | 13.4% |

DFFs store 16-bit weights and four 32-bit output registers. XNOR2 and AOI21 dominate
because each 32-bit add produces one XOR per sum bit and an AOI/OAI carry tree.

## Warnings

Two ABC warnings: (1) **"Templates are not defined"** — no wire-load model, so delay
estimates are slightly optimistic for long interconnects. (2) **Sequential cells skipped by
abc** — DFFs are mapped by `dfflibmap`; the Yosys `stat` totals (1,549 cells, 4,060 µm²)
are authoritative, not the abc gate count (1,476 gates, 3,267 µm²). OpenLane 2 was not run
due to a KLayout Python 3.14 incompatibility; see `synth/metrics.csv`.
