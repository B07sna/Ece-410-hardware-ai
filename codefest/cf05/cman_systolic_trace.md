# Codefest 5 — Systolic Array Trace

**Problem:** 2×2 weight-stationary systolic array computing **C = A × B**

```
A = [[1, 2],    B = [[5, 6],    Expected C = [[19, 22],
     [3, 4]]         [7, 8]]                  [43, 50]]
```

Verification:
- C[0][0] = 1·5 + 2·7 = 5 + 14 = **19** ✓
- C[0][1] = 1·6 + 2·8 = 6 + 16 = **22** ✓
- C[1][0] = 3·5 + 4·7 = 15 + 28 = **43** ✓
- C[1][1] = 3·6 + 4·8 = 18 + 32 = **50** ✓

---

## (a) PE Array Diagram — Preloaded Weights

Architecture conventions:
- **Weights** (matrix B) are preloaded at startup and never move: `PE[k][j]` holds `B[k][j]`
- **Activations** (rows of A) enter from the **left edge** of each row and flow **right** one column per cycle
- **Partial sums** flow **downward** one row per cycle (psum_in for row-0 PEs is always 0)
- **Row 1 inputs are injected 1 cycle later** than row 0 so psums and activations arrive at PE[1][j] in the same cycle

```
   Activation injection (left edges, staggered by row index)
   ──────────────────────────────────────────────────────────
   Row 0 left:  A[0][0]=1  @ cycle 1,  A[1][0]=3  @ cycle 2
   Row 1 left:  A[0][1]=2  @ cycle 2,  A[1][1]=4  @ cycle 3   ← 1-cycle delay

   col 0                   col 1
    │                       │
    │  act flows right ─►   │  act flows right ─►
    │                       │
  ──┼────┌─────────────┐    ┼────┌─────────────┐
  R │    │  PE[0][0]   │    │    │  PE[0][1]   │
  o ├───►│  weight = 5 ├───►│───►│  weight = 6 │
  w │    │             │    │    │             │
  0 │    └──────┬──────┘    │    └──────┬──────┘
    │           │ psum ↓    │           │ psum ↓
  ──┼────┌──────▼──────┐    ┼────┌──────▼──────┐
  R │    │  PE[1][0]   │    │    │  PE[1][1]   │
  o ├───►│  weight = 7 ├───►│───►│  weight = 8 │
  w │    │             │    │    │             │
  1 │    └──────┬──────┘    │    └──────┬──────┘
               │                        │
               ▼                        ▼
          C[0][0]=19                C[0][1]=22
          C[1][0]=43                C[1][1]=50

  Each PE operation per cycle:
    psum_out = psum_in  +  activation × weight
              (from PE above,          (fixed,
               1 cycle ago)            preloaded)
```

---

## (b) Cycle-by-Cycle Trace

**Column key:**
- `Row N in` = activation value injected at the left edge of row N this cycle
- `PE[r][c]` columns show `act | psum_out` — the activation the PE sees and its outgoing partial sum
- `psum_in` for PE[1][j] = `psum_out` of PE[0][j] from the *previous* cycle
- `psum_in` for PE[0][j] = 0 always (no row above)

| Cycle | Row 0 in | Row 1 in | PE[0][0] act\|psum | PE[0][1] act\|psum | PE[1][0] act\|psum | PE[1][1] act\|psum | C output |
|:-----:|:--------:|:--------:|:------------------:|:------------------:|:------------------:|:------------------:|:--------:|
| **1** | 1        | 0        | 1 \| **5**          | 0 \| 0             | 0 \| 0             | 0 \| 0             | —        |
| **2** | 3        | 2        | 3 \| **15**         | 1 \| **6**         | 2 \| **19**        | 0 \| 0             | C[0][0] = **19** |
| **3** | 0        | 4        | 0 \| 0             | 3 \| **18**        | 4 \| **43**        | 2 \| **22**        | C[1][0] = **43**, C[0][1] = **22** |
| **4** | 0        | 0        | 0 \| 0             | 0 \| 0             | 0 \| 0             | 4 \| **50**        | C[1][1] = **50** |

### Annotated MAC arithmetic per cycle

**Cycle 1**
```
PE[0][0]:  psum_in=0,  act=1,  0 + 1×5 = 5       → psum_out = 5
PE[0][1]:  psum_in=0,  act=0,  0 + 0×6 = 0       → psum_out = 0
PE[1][0]:  psum_in=0*  act=0,  0 + 0×7 = 0       → psum_out = 0
PE[1][1]:  psum_in=0*  act=0,  0 + 0×8 = 0       → psum_out = 0
           * psum_in = PE[0][col] output from cycle 0 = 0
```

**Cycle 2**
```
PE[0][0]:  psum_in=0,  act=3,  0 + 3×5 = 15      → psum_out = 15
PE[0][1]:  psum_in=0,  act=1,  0 + 1×6 = 6       → psum_out = 6
PE[1][0]:  psum_in=5,  act=2,  5 + 2×7 = 19      → psum_out = 19  ✓ C[0][0]
PE[1][1]:  psum_in=0,  act=0,  0 + 0×8 = 0       → psum_out = 0
```

**Cycle 3**
```
PE[0][0]:  psum_in=0,  act=0,  0 + 0×5 = 0       → psum_out = 0
PE[0][1]:  psum_in=0,  act=3,  0 + 3×6 = 18      → psum_out = 18
PE[1][0]:  psum_in=15, act=4,  15 + 4×7 = 43     → psum_out = 43  ✓ C[1][0]
PE[1][1]:  psum_in=6,  act=2,  6  + 2×8 = 22     → psum_out = 22  ✓ C[0][1]
```

**Cycle 4**
```
PE[0][0]:  psum_in=0,  act=0,  0 + 0×5 = 0       → psum_out = 0
PE[0][1]:  psum_in=0,  act=0,  0 + 0×6 = 0       → psum_out = 0
PE[1][0]:  psum_in=0,  act=0,  0 + 0×7 = 0       → psum_out = 0   (done)
PE[1][1]:  psum_in=18, act=4,  18 + 4×8 = 50     → psum_out = 50  ✓ C[1][1]
```

All four outputs drain by cycle 4: C = [[19, 22], [43, 50]] ✓

---

## (c) Hardware Metrics

### Total MAC Count

Each output element C[i][j] = Σ_k A[i][k]·B[k][j] requires k=2 multiply-accumulate operations.
With 4 output elements: **4 × 2 = 8 total MACs**.

| PE       | MACs performed | Cycles active |
|----------|:--------------:|:-------------:|
| PE[0][0] | 2              | 1, 2          |
| PE[0][1] | 2              | 2, 3          |
| PE[1][0] | 2              | 2, 3          |
| PE[1][1] | 2              | 3, 4          |
| **Total**| **8**          |               |

### Input Reuse Count

**Weight reuse (matrix B):**
Each of the 4 weight values is preloaded once and used once per row of A (M=2 rows).
Each weight fires 2 MACs, so reuse factor = 2 (loaded once, used twice).
```
Weight reuses = 4 weights × (2 uses − 1 load) = 4 weight reuses
```

**Activation reuse (matrix A):**
Each A[i][k] enters row k and flows rightward through N=2 columns, firing one MAC per PE.
Each activation fires 2 MACs, so reuse factor = 2 (loaded once, used twice).
```
Activation reuses = 4 A elements × (2 uses − 1 load) = 4 activation reuses
```

**Total input reuses = 4 (weights) + 4 (activations) = 8**

### Off-Chip Memory Access Count

| Matrix | Direction | Count | Notes |
|--------|-----------|:-----:|-------|
| A      | Read      | **4** | Each of the 4 elements streamed in once; no re-fetch |
| B      | Read      | **4** | All 4 weights preloaded once at setup |
| C      | Write     | **4** | Each of the 4 output values written out once |
| **Total** |        | **12** | |

Breakdown: 8 reads (4 A + 4 B) + 4 writes (C) = **12 off-chip memory accesses**.

---

## (d) Output-Stationary Dataflow

In output-stationary dataflow, the partial sum (accumulator) for each output element remains fixed inside its assigned PE while both input activations and weights are streamed in from off-chip each cycle.
