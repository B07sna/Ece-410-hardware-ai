# M3 Synthesis Plan — Option B (Generic Library Fallback)
**ECE 410 · CodeFest 7 · Hussain Alquzweni**

---

## Why Option B

Option A targets the **Sky130 PDK** (`sky130_fd_sc_hd__tt_025C_1v80.lib`), which provides
process-accurate timing, area, and power for a real open-source 130 nm node. Sky130 is
not currently installed in the project WSL environment. Rather than block M3 progress,
Option B substitutes a **hand-crafted generic 130 nm liberty file** (`generic_130nm.lib`)
with representative scalar timing arcs derived from published 130 nm bulk-CMOS corner
data. Results are plausible for educational estimation but not tape-out accurate.

---

## Option B Flow (Executed)

1. `wsl yosys -s crossbar_mac.ys` — Yosys 0.52 with generic liberty
2. `dfflegalize` → convert sync-reset DFFs to async-reset (library limitation)
3. `dfflibmap` → map DFF/DFFR; `abc -D 10000` → combinational mapping at 10 ns
4. `stat -liberty` → cell count, area; ABC `stime -p` → critical-path delay

## Path to Option A

Install the Sky130 PDK in WSL:
```bash
pip install sky130  # or openlane, which bundles the PDK
```
Then replace `generic_130nm.lib` with
`$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib`
and re-run `crossbar_mac.ys` unchanged. No script edits are needed.
