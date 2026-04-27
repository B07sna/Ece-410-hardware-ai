# Code Review: INT8 MAC Unit — LLM-Generated SystemVerilog

**Reviewer:** ECE 410 Hardware AI  
**Date:** 2026-04-26  
**Files reviewed:** `mac_llm_A.v` (Claude Sonnet 4.6), `mac_llm_B.v` (GPT-4o)  
**Reference:** `mac_correct.v` (corrected implementation)

---

## Overview

Both LLMs were asked to produce a synthesizable SystemVerilog MAC unit:
- Module `mac`, inputs `clk` (1-bit), `rst` (1-bit active-high synchronous),
  `a` (8-bit signed), `b` (8-bit signed), output `out` (32-bit signed accumulator).
- Behavior: on rising edge, if `rst` → clear to 0; else `out <= out + a*b`.
- Constraints: `always_ff` only, no `initial` blocks, no `$display`, no delays.

---

## Simulation Environment

**Tool:** Icarus Verilog 12.0 (stable), installed via MSYS2 `mingw-w64-x86_64-iverilog`.  
**Compile flags:** `iverilog -g2012 -Wall`  
**Testbench:** `mac_tb.v` — stimulus: `a=3, b=4` × 3 cycles → `rst` → `a=-5, b=2` × 2 cycles.

---

## LLM A — Claude Sonnet 4.6 (`mac_llm_A.v`)

### Source

```systemverilog
module mac (
    input  logic        clk,
    input  logic        rst,
    input  logic signed [7:0]  a,
    input  logic signed [7:0]  b,
    output logic signed [31:0] out
);

    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else
            out <= out + (a * b);
    end

endmodule
```

### Compilation

```
$ iverilog -g2012 -Wall -o mac_a.vvp mac_llm_A.v
(no warnings, no errors)
Exit: 0
```

### Issues Found

#### Issue A-1 — Missing `timescale` directive

**Quoted line:** *(absent — no `timescale` at top of file)*

**Problem:** When compiled alongside modules that carry a `` `timescale `` directive,
iverilog warns:

```
warning: Some design elements have no explicit time unit and/or time precision.
         This may cause confusing timing results.
```

Synthesis tools are unaffected (timescale is ignored for synthesis), but simulation
time resolution becomes tool-dependent and the file is not self-describing.

**Correction:** Add `` `timescale 1ns/1ps `` as the first line.

---

#### Issue A-2 — Implicit sign-extension of the multiply product

**Quoted line:**
```systemverilog
            out <= out + (a * b);
```

**Problem:** In SystemVerilog, `a * b` where both operands are `signed [7:0]`
produces a **16-bit signed** intermediate result (the width of the wider operand).
Adding a 16-bit signed value to a 32-bit signed `out` triggers **implicit
sign-extension**, which is correct under IEEE 1800-2012 §11.6.1 — but it relies on
the tool correctly propagating the `signed` attribute through the multiply.
Some synthesis tools and older lint checkers treat the product width as
implementation-defined, generating width-mismatch warnings or silently truncating.

The intent is unambiguous but the code is not maximally portable.

**Correction:** Cast explicitly to make the sign-extension visible to all tools:
```systemverilog
out <= out + 32'(signed'(a * b));
```
This explicitly sign-extends the 16-bit product to 32 bits before the addition,
removing any tool-dependency on implicit widening rules.

---

### Verdict: **PASS with minor style issues**

LLM A correctly implements the MAC. Both issues are quality-of-life concerns
rather than functional bugs. The module produces correct results in simulation.

---

## LLM B — GPT-4o (`mac_llm_B.v`)

### Source

```systemverilog
module mac (
    input  logic        clk,
    input  logic        rst,
    input  logic [7:0]  a,          // Bug 2: missing signed
    input  logic [7:0]  b,          // Bug 2: missing signed
    output logic [31:0] out         // Bug 3: missing signed
);

    always @(posedge clk) begin     // Bug 1: should be always_ff
        if (rst)
            out <= 32'h0;
        else
            out <= out + (a * b);   // unsigned multiply; no sign extension
    end

endmodule
```

### Compilation

```
$ iverilog -g2012 -Wall -o mac_b.vvp mac_llm_B.v
(no warnings, no errors)
Exit: 0
```

> **Note:** Icarus Verilog does not warn on the unsigned-multiply or missing-`always_ff`
> issues at `-Wall`. This demonstrates why simulation alone is insufficient — the bugs
> are semantic, not syntactic.

### Issues Found

#### Issue B-1 — `always @(posedge clk)` instead of `always_ff`

**Quoted line:**
```systemverilog
    always @(posedge clk) begin     // Bug 1: should be always_ff
```

**Problem:** `always @(posedge clk)` is valid Verilog-2001 syntax and will synthesize
correctly in most cases, but it is **not restricted** to sequential logic. The
SystemVerilog `always_ff` construct:

1. **Enforces** that the block contains only flip-flop inferrable statements; any
   combinational or latch-inducing code inside `always_ff` is a **compile error**.
2. **Documents intent** to synthesis tools and formal verification tools, enabling
   stricter checks and better error messages.
3. Is explicitly required by many RTL coding guidelines (e.g., Synopsys, Cadence).

Using plain `always` silently permits mistakes such as non-clocked assignments or
missing `else` branches (which would infer latches) without any tooling feedback.

**Correction:**
```systemverilog
    always_ff @(posedge clk) begin
```

---

#### Issue B-2 — Missing `signed` on inputs causes unsigned multiply (functional bug)

**Quoted lines:**
```systemverilog
    input  logic [7:0]  a,          // Bug 2: missing signed
    input  logic [7:0]  b,          // Bug 2: missing signed
```
and
```systemverilog
            out <= out + (a * b);   // unsigned multiply; no sign extension
```

**Problem:** Because `a` and `b` are declared as `logic [7:0]` (unsigned), the
expression `a * b` is an **unsigned 8×8 multiply**. When `a = -5` (stored as
`8'b11111011 = 8'd251`) and `b = 2`:

| Interpretation | Computation | Result |
|---|---|---|
| **Correct** (signed) | −5 × 2 | **−10** |
| **Buggy** (unsigned) | 251 × 2 | **502** |

The accumulator diverges catastrophically for any negative input. This is a
**functional correctness bug** — the module computes the wrong mathematical operation.

Simulation confirms: after `rst`, applying `a=-5, b=2` gives `out_B = 502`
instead of the expected `out_A = -10`.

**Correction:** Add the `signed` keyword to both input declarations:
```systemverilog
    input  logic signed [7:0]  a,
    input  logic signed [7:0]  b,
```

---

#### Issue B-3 — Missing `signed` on output port

**Quoted line:**
```systemverilog
    output logic [31:0] out         // Bug 3: missing signed
```

**Problem:** The output port is declared unsigned. While the register bits
are physically identical whether declared signed or unsigned, the missing
keyword has two consequences:

1. **Downstream misinterpretation:** Any module that connects to `out` and
   reads it as unsigned will see a large positive number instead of a
   negative accumulator value (e.g., `4294967286` instead of `−10`).
2. **Broken `$signed()` comparisons:** Testbench code using `out > -1` or
   similar comparisons will compare unsigned values, silently giving wrong
   results without any warning from the simulator.

**Correction:**
```systemverilog
    output logic signed [31:0] out
```

---

### Verdict: **FAIL — functional bug (Issue B-2)**

LLM B's module produces incorrect results for any negative input operand.
Issue B-2 is a silent functional bug that compiles and elaborates without
any warning under `iverilog -g2012 -Wall`, making it particularly dangerous.

---

## Simulation Results

### mac_llm_A vs mac_llm_B

```
$ iverilog -g2012 -Wall -o mac_sim.vvp mac_llm_A.v mac_B.v mac_tb.v
warning: Some design elements have no explicit time unit and/or time precision.
Exit: 0

$ vvp mac_sim.vvp

=======================================================
 mac_tb: comparing mac_llm_A (correct) vs mac_llm_B (buggy)
=======================================================
time     | a    b    rst | out_A    | out_B
-------------------------------------------------------
t=6000 ns | rst=1 a=   0 b=   0 | out_A=     0 | out_B=     0
t=16000 ns | rst=0 a=   3 b=   4 | out_A=    12 | out_B=    12
t=26000 ns | rst=0 a=   3 b=   4 | out_A=    24 | out_B=    24
t=36000 ns | rst=0 a=   3 b=   4 | out_A=    36 | out_B=    36
t=46000 ns | rst=1 a=   3 b=   4 | out_A=     0 | out_B=     0
t=56000 ns | rst=0 a=  -5 b=   2 | out_A=   -10 | out_B=   502  <-- MISMATCH
t=66000 ns | rst=0 a=  -5 b=   2 | out_A=   -20 | out_B=  1004  <-- MISMATCH
-------------------------------------------------------
Expected out_A after phase 2: -10, then -20
Expected out_B after phase 2:  502, then 1004 (wrong!)
=======================================================
```

**Observation:** Both modules agree for positive inputs (a=3, b=4 → accumulates
correctly to 12, 24, 36). The unsigned-multiply bug in LLM B is latent until a
negative operand is applied, at which point the outputs diverge by a factor of ~50.

---

### mac_correct.v — Reference Implementation

```systemverilog
`timescale 1ns/1ps

module mac (
    input  logic                 clk,
    input  logic                 rst,
    input  logic signed [7:0]    a,
    input  logic signed [7:0]    b,
    output logic signed [31:0]   out
);

    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else
            out <= out + 32'(signed'(a * b));
    end

endmodule
```

**Fixes applied vs LLM B:**
- `always_ff` (Issue B-1)
- `signed` on inputs `a` and `b` (Issue B-2)
- `signed` on output `out` (Issue B-3)
- Explicit cast `32'(signed'(a * b))` for portable sign-extension (Issue A-2)
- `` `timescale `` directive (Issue A-1)

```
$ iverilog -g2012 -Wall -o mac_correct_sim.vvp mac_correct.v mac_B.v mac_tb.v
C:/…/mac_B.v:18: warning: timescale for mac_B inherited from another file.
Exit: 0

$ vvp mac_correct_sim.vvp

=======================================================
 mac_tb: comparing mac_llm_A (correct) vs mac_llm_B (buggy)
=======================================================
time     | a    b    rst | out_A    | out_B
-------------------------------------------------------
t=6000 ns | rst=1 a=   0 b=   0 | out_A=     0 | out_B=     0
t=16000 ns | rst=0 a=   3 b=   4 | out_A=    12 | out_B=    12
t=26000 ns | rst=0 a=   3 b=   4 | out_A=    24 | out_B=    24
t=36000 ns | rst=0 a=   3 b=   4 | out_A=    36 | out_B=    36
t=46000 ns | rst=1 a=   3 b=   4 | out_A=     0 | out_B=     0
t=56000 ns | rst=0 a=  -5 b=   2 | out_A=   -10 | out_B=   502  <-- MISMATCH
t=66000 ns | rst=0 a=  -5 b=   2 | out_A=   -20 | out_B=  1004  <-- MISMATCH
-------------------------------------------------------
```

`mac_correct.v` produces `out_A = -10` and `out_A = -20` — matching the expected
signed accumulator values exactly. The MISMATCH flags remain because `mac_B`
(GPT-4o's buggy version) is still in the simulation; `mac_correct` (the `out_A`
column) is correct on every cycle.

---

## Summary Table

| Issue | File | Severity | Root Cause | Effect in Simulation |
|---|---|---|---|---|
| A-1: no `timescale` | `mac_llm_A.v` | Style | Omission | Tool warning on mixed compilation |
| A-2: implicit sign-extension | `mac_llm_A.v` | Minor | Implicit SV rule | Correct but tool-dependent |
| B-1: `always` not `always_ff` | `mac_llm_B.v` | Style/Risk | Wrong keyword | No synthesis guard against latches |
| **B-2: unsigned multiply** | **`mac_llm_B.v`** | **Critical** | **Missing `signed`** | **502 instead of −10 for `a=-5, b=2`** |
| B-3: unsigned output port | `mac_llm_B.v` | Moderate | Missing `signed` | Downstream misinterpretation of negative values |

---

## Key Takeaway

The most dangerous LLM error in this review is **silent semantic incorrectness**.
Issue B-2 compiles and simulates without a single warning at `iverilog -g2012 -Wall`,
yet produces results that are mathematically wrong for the core intended use case
(signed integer multiply-accumulate). This class of bug — **type-system omissions
that are syntactically valid but semantically broken** — is a consistent failure mode
of LLM-generated hardware description code and requires explicit human review or
formal property checking to catch.
