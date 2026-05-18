# Synthesis Interpretation — crossbar_mac
**ECE 410 · CodeFest 7 · Hussain Alquzweni**

---

## Tool and Library

Yosys 0.52 · generic 130 nm standard-cell library (`generic_130nm.lib`, educational,
typical corner 1.8 V / 25 °C). Sky130 PDK is not installed in this WSL environment;
the generic library uses representative 130 nm timing values (see `m3_plan.md` for the
sky130 Option A path). The design was read as Verilog-2005 (`synth_top.sv`), which is a
port-flattened equivalent of `cf06/hdl/crossbar_mac.sv`.

---

## Clock Period and Worst-Case Slack

The synthesis targeted a **10 ns clock (100 MHz)**. ABC (`&nf -D 10000`) reported a
worst-case combinational delay of **5380 ps (5.38 ns)** with timing closed and no
upsizing required. Adding the DFFR clock-to-Q (0.30 ns) and an estimated setup time
(0.18 ns), the full register-to-register path is **5.86 ns**, giving a
**worst-case slack of +4.14 ns**. The design meets timing with more than 40% margin.
The maximum achievable clock (at this library) is approximately **1 / 5.86 ns ≈ 171 MHz**.

---

## Critical Path

The 21-stage critical path runs from primary input `in0[1]` (bit 1 of activation input 0)
through the combinational accumulator for column 0. The dominant cell types on the path
are **XNOR2** (XOR-based sum), **AOI21/OAI21** (carry-generate/propagate in a Brent-Kung
prefix tree), and **MUX2** (weight-gated add/subtract select). The path ends at the D
input of an output-register DFF. ABC chose an AOI/OAI prefix-tree adder style to minimize
the 32-bit add depth; this cuts the adder critical path to roughly 13 logic levels rather
than the 32+ levels of a ripple-carry chain.

---

## Cell Area

| Metric | Value |
|--------|-------|
| Total cells | 1 549 |
| Sequential (DFF) | 144 (21.3% of area) |
| Combinational | 1 405 |
| **Chip area** | **4 060 µm²** |

---

## Top Contributors by Area (µm²)

| Cell | Count | Unit area | Total area | % of total |
|------|------:|----------:|-----------:|----------:|
| XNOR2 | 159 | 3.5 | 556.5 | 13.7% |
| AOI21 | 217 | 2.5 | 542.5 | 13.4% |
| MUX2 | 127 | 3.0 | 381.0 | 9.4% |
| OAI21 | 176 | 2.5 | 440.0 | 10.8% |
| AND2 | 178 | 2.0 | 356.0 | 8.8% |
| DFF | 144 | 6.0 | 864.0 | 21.3% |

XNOR/XNOR2 dominates because each 32-bit add/subtract produces one XOR per sum bit.
AOI21/OAI21 pairs implement the Brent-Kung carry tree. DFFs account for the weight
storage (16 bits) and all four 32-bit output registers (128 bits = 144 total).

---

## Warnings

Two ABC warnings were logged:
1. **"Templates are not defined"** — the library lacks wire-load model groups; load
   capacitance is treated as zero. Area and delay estimates are slightly optimistic for
   long interconnects, typical for early-stage educational synthesis.
2. **Sequential cells skipped by abc** — DFF/DFFR cells are mapped by `dfflibmap`, not
   abc; the abc gate count (1 476 gates, 3 267 µm²) excludes the 144 DFFs. The stat
   command reports the complete 1 549-cell, 4 060 µm² total.
