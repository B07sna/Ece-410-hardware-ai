# Sneak Path Analysis — 2×2 Resistive Crossbar
**ECE 410 · CodeFest 6**

---

## 1. Array Specification

```
           col0 (j=0)       col1 (j=1)
              │                 │
row0 (i=0) ──┤ R[0][0]=1 kΩ ──┤ R[0][1]=2 kΩ ──
              │                 │
row1 (i=1) ──┤ R[1][0]=2 kΩ ──┤ R[1][1]=1 kΩ ──
              │                 │
```

Each crosspoint resistance R[i][j] connects **row wire i** to **column wire j**.  
Conductance matrix (in mS):

| | col0 | col1 |
|---|---|---|
| **row0** | G[0][0] = 1.000 mS | G[0][1] = 0.500 mS |
| **row1** | G[1][0] = 0.500 mS | G[1][1] = 1.000 mS |

---

## 2. Part (a) — Ideal Read of col0

### Bias Conditions

| Node | Value |
|---|---|
| V\_row0 | 1 V (driven) |
| V\_col0 | 0 V (grounded — sense node) |
| V\_row1 | 0 V (grounded) |
| V\_col1 | 0 V (grounded) |

All unselected nodes are actively held at 0 V, eliminating any floating nodes.

### Current Computation

$$I_{\text{col0}} = \sum_i \frac{V_{\text{row}i} - V_{\text{col0}}}{R[i][0]}$$

| Path | Numerator | Denominator | Current |
|---|---|---|---|
| row0 → R[0][0] → col0 | 1 V − 0 V = 1 V | 1 kΩ | **1.000 mA** |
| row1 → R[1][0] → col0 | 0 V − 0 V = 0 V | 2 kΩ | **0.000 mA** |

$$\boxed{I_{\text{col0}}^{\text{ideal}} = 1.000 \text{ mA}}$$

### Verification

Apparent resistance seen at col0:

$$R_{\text{app}} = \frac{V_{\text{row0}}}{I_{\text{col0}}} = \frac{1\text{ V}}{1\text{ mA}} = 1\text{ k}\Omega = R[0][0] \checkmark$$

The ideal read correctly resolves R[0][0] with zero error.

---

## 3. Part (b) — Sneak Path Read of col0 (row1, col1 Floating)

### Bias Conditions

| Node | Value |
|---|---|
| V\_row0 | 1 V (driven) |
| V\_col0 | 0 V (grounded) |
| V\_row1 | **unknown V_a** (floating) |
| V\_col1 | **unknown V_b** (floating) |

No external current is injected into or extracted from floating nodes.

---

### 3.1 KCL System Setup

#### KCL at row1 (V_a, floating → net external current = 0)

Currents **leaving** row1:

$$\frac{V_a - V_{\text{col0}}}{R[1][0]} + \frac{V_a - V_b}{R[1][1]} = 0$$

$$\frac{V_a}{2\text{ k}\Omega} + \frac{V_a - V_b}{1\text{ k}\Omega} = 0$$

Multiply through by 2 kΩ:

$$V_a + 2(V_a - V_b) = 0$$

$$\boxed{3V_a - 2V_b = 0} \tag{1}$$

#### KCL at col1 (V_b, floating → net external current = 0)

Currents **leaving** col1:

$$\frac{V_b - V_{\text{row0}}}{R[0][1]} + \frac{V_b - V_a}{R[1][1]} = 0$$

$$\frac{V_b - 1}{2\text{ k}\Omega} + \frac{V_b - V_a}{1\text{ k}\Omega} = 0$$

Multiply through by 2 kΩ:

$$(V_b - 1) + 2(V_b - V_a) = 0$$

$$\boxed{-2V_a + 3V_b = 1} \tag{2}$$

---

### 3.2 Solving the Linear System

$$\begin{bmatrix} 3 & -2 \\ -2 & 3 \end{bmatrix} \begin{bmatrix} V_a \\ V_b \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Determinant: $\Delta = (3)(3) - (-2)(-2) = 9 - 4 = 5$

$$V_a = \frac{(0)(3) - (1)(-2)}{5} = \frac{2}{5} = 0.4 \text{ V}$$

$$V_b = \frac{(3)(1) - (-2)(0)}{5} = \frac{3}{5} = 0.6 \text{ V}$$

$$\boxed{V_{\text{row1}} = 0.4 \text{ V}, \quad V_{\text{col1}} = 0.6 \text{ V}}$$

---

### 3.3 Sneak Path Verification

The sneak path is the parasitic loop:  
**row0 → R[0][1] → col1 → R[1][1] → row1 → R[1][0] → col0**

| Segment | Voltage drop | Resistance | Current |
|---|---|---|---|
| R[0][1]: row0 → col1 | 1.0 V − 0.6 V = 0.4 V | 2 kΩ | 0.200 mA |
| R[1][1]: col1 → row1 | 0.6 V − 0.4 V = 0.2 V | 1 kΩ | 0.200 mA |
| R[1][0]: row1 → col0 | 0.4 V − 0.0 V = 0.4 V | 2 kΩ | 0.200 mA |

Current is **conserved at 0.200 mA** through every segment of the sneak loop. ✓

---

### 3.4 Total I\_col0 with Sneak Path

$$I_{\text{col0}} = \frac{V_{\text{row0}} - V_{\text{col0}}}{R[0][0]} + \frac{V_{\text{row1}} - V_{\text{col0}}}{R[1][0]}$$

| Contribution | Calculation | Current |
|---|---|---|
| Direct path: R[0][0] | (1 V − 0 V) / 1 kΩ | 1.000 mA |
| **Sneak path: R[1][0]** | **(0.4 V − 0 V) / 2 kΩ** | **0.200 mA** |

$$\boxed{I_{\text{col0}}^{\text{sneak}} = 1.200 \text{ mA}}$$

---

### 3.5 Error Quantification

| Metric | Value |
|---|---|
| Ideal current | 1.000 mA |
| Measured current (sneak) | 1.200 mA |
| Sneak current contribution | +0.200 mA |
| Relative error | +20.0 % |
| Apparent R[0][0] from measurement | 1 V / 1.2 mA ≈ **833 Ω** (true: 1 kΩ) |

---

## 4. Part (c) — How Sneak Paths Corrupt MVM Results

In a resistive crossbar performing matrix-vector multiplication (MVM), each column output current should equal the dot product of the input voltage vector with the conductance column **for that column alone**. When unselected row and column lines are left floating, however, parasitic sneak-path loops form through adjacent low-resistance crosspoints, injecting extra current into the sense column that belongs to no valid dot-product term. This corrupts every element of the output current vector simultaneously — not just the cell being read — because the floating node voltages (V\_row1, V\_col1 in the 2×2 case) depend on the **entire** conductance landscape, coupling all weights into a single spurious contribution. The result is a systematic, weight-dependent bias that cannot be corrected by a simple scalar offset, making sneak paths one of the fundamental accuracy limiters of passive crossbar analog compute.

---

## 5. Summary Table

| Condition | V\_row1 | V\_col1 | I\_col0 | Error |
|---|---|---|---|---|
| Ideal (all grounded) | 0 V | 0 V | 1.000 mA | 0 % |
| Sneak path (row1, col1 float) | 0.4 V | 0.6 V | 1.200 mA | +20 % |

> **Key takeaway:** A 20 % current error in a 2×2 array illustrates why practical crossbar designs use 1T1R (one transistor, one resistor) cells, active column-biasing schemes, or voltage-mode half-select techniques to suppress sneak paths during inference.
