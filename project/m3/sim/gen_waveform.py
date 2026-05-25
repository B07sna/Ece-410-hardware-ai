"""
gen_waveform.py — Parse tb_top.vcd and render cosim_waveform.png.

Three annotated regions:
  Region A: Host write transaction  (AXI4-Lite CTRL write + weight loading)
  Region B: Compute activity        (pixel stream accepted by compute_core)
  Region C: Host read result        (m_axis_result_tvalid fires, STATUS read)
"""

import re
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

BASE = "C:/Users/Husai/Ece-410-hardware-ai/project/m3/sim"
VCD_PATH  = os.path.join(BASE, "tb_top.vcd")
PNG_PATH  = os.path.join(BASE, "cosim_waveform.png")

# ---------------------------------------------------------------------------
# Minimal VCD parser
# ---------------------------------------------------------------------------
def parse_vcd(path, wanted_names):
    """Return dict name -> [(time_ps, value_int)] for each wanted signal."""
    id_to_name = {}   # VCD id char(s) -> signal name
    traces = {n: [] for n in wanted_names}
    current_time = 0

    with open(path, "r") as f:
        lines = f.read().splitlines()

    in_def = True
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # parse $var lines: $var wire <bits> <id> <name> $end
        if in_def and line.startswith("$var"):
            m = re.match(r'\$var\s+\S+\s+\d+\s+(\S+)\s+(\S+)', line)
            if m:
                vid, vname = m.group(1), m.group(2)
                if vname in wanted_names:
                    id_to_name[vid] = vname

        if "$enddefinitions" in line:
            in_def = False

        if not in_def:
            # time marker
            if line.startswith("#"):
                try:
                    current_time = int(line[1:])
                except ValueError:
                    pass
            # scalar value change: 0/1/x/z followed by id
            elif len(line) >= 2 and line[0] in "01xzXZ" and not line.startswith("b"):
                val_ch = line[0]
                vid    = line[1:]
                if vid in id_to_name:
                    v = 1 if val_ch == "1" else 0
                    traces[id_to_name[vid]].append((current_time, v))
            # vector value change: b<binary> <id>
            elif line.startswith("b"):
                parts = line.split()
                if len(parts) == 2:
                    bval, vid = parts
                    if vid in id_to_name:
                        try:
                            v = int(bval[1:], 2)
                        except ValueError:
                            v = 0
                        traces[id_to_name[vid]].append((current_time, v))
        i += 1

    return traces

# ---------------------------------------------------------------------------
# Build step waveform arrays from (time, value) list
# ---------------------------------------------------------------------------
def to_steps(events, t_end):
    """Convert [(t, v), ...] to (times[], values[]) for step plotting."""
    if not events:
        return [0, t_end], [0, 0]
    times  = []
    values = []
    cur_v  = 0
    for t, v in events:
        if times:
            times.append(t)
            values.append(cur_v)
        times.append(t)
        values.append(v)
        cur_v = v
    times.append(t_end)
    values.append(cur_v)
    return times, values

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
wanted = [
    "clk",
    "s_axil_awvalid", "s_axil_awready",
    "s_axis_weight_tvalid", "s_axis_weight_tready",
    "s_axis_pixel_tvalid",  "s_axis_pixel_tready",
    "m_axis_result_tvalid",
]

print(f"Parsing {VCD_PATH} ...")
traces = parse_vcd(VCD_PATH, wanted)

# Determine time axis in ns (timescale is 1ns/1ps → 1 unit = 1 ps)
all_times = []
for evts in traces.values():
    for t, _ in evts:
        all_times.append(t)
t_end_ps = max(all_times) if all_times else 1000000
t_end_ns = t_end_ps / 1000.0

def ns(evts):
    return [(t / 1000.0, v) for t, v in evts]

# ---------------------------------------------------------------------------
# Identify annotation boundaries (in ns) from first rising edges
# ---------------------------------------------------------------------------
def first_rise(evts_ns):
    for t, v in evts_ns:
        if v == 1:
            return t
    return None

def last_fall(evts_ns):
    result = None
    for t, v in evts_ns:
        if v == 0:
            result = t
    return result

wt_ns   = ns(traces["s_axis_weight_tvalid"])
pix_ns  = ns(traces["s_axis_pixel_tvalid"])
res_ns  = ns(traces["m_axis_result_tvalid"])
aw_ns   = ns(traces["s_axil_awvalid"])

t_weight_start = first_rise(wt_ns)   or 30
t_weight_end   = last_fall(wt_ns)    or 120
t_axil_start   = first_rise(aw_ns)   or 130
t_axil_end     = (last_fall(ns(traces["s_axil_awready"])) or (t_axil_start + 60))
t_pix_start    = first_rise(pix_ns)  or (t_axil_end + 20)
t_pix_end      = last_fall(pix_ns)   or (t_pix_start + 100)
t_res_start    = first_rise(res_ns)  or (t_pix_end + 60)
t_res_end      = (t_res_start + 30)  if t_res_start else (t_pix_end + 90)

# Region A: weight load + AXI-Lite write
rA_s = min(t_weight_start, t_axil_start) - 10
rA_e = max(t_weight_end,   t_axil_end)   + 10
# Region B: pixel streaming / compute
rB_s = t_pix_start - 10
rB_e = t_pix_end   + 10
# Region C: result valid
rC_s = t_res_start - 10
rC_e = min(t_res_end + 30, t_end_ns)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
SIGNALS = [
    ("clk",                    "clk"),
    ("s_axil_awvalid",         "axil_awvalid"),
    ("s_axil_awready",         "axil_awready"),
    ("s_axis_weight_tvalid",   "wt_tvalid"),
    ("s_axis_pixel_tvalid",    "pix_tvalid"),
    ("s_axis_pixel_tready",    "pix_tready"),
    ("m_axis_result_tvalid",   "res_tvalid"),
]

n_sig  = len(SIGNALS)
fig_h  = 1.2 * n_sig + 1.5
fig, ax = plt.subplots(figsize=(14, fig_h))

COLORS = {
    "clk":          "#4a90d9",
    "axil_awvalid": "#e67e22",
    "axil_awready": "#f39c12",
    "wt_tvalid":    "#27ae60",
    "pix_tvalid":   "#8e44ad",
    "pix_tready":   "#16a085",
    "res_tvalid":   "#c0392b",
}

row_h  = 1.0
y_top  = n_sig * row_h

for idx, (sig_key, label) in enumerate(SIGNALS):
    y_base = (n_sig - 1 - idx) * row_h
    evts   = ns(traces.get(sig_key, []))
    times, values = to_steps(evts, t_end_ns)
    scaled = [v * 0.7 + y_base + 0.1 for v in values]
    color  = COLORS.get(label, "#555555")
    ax.plot(times, scaled, color=color, linewidth=1.4, drawstyle="steps-post")
    ax.text(-5, y_base + 0.4, label, ha="right", va="center",
            fontsize=8, color=color, fontweight="bold")

# Y-axis ticks off
ax.set_yticks([])

# Region shading
def shade(xs, xe, label, color, y_label):
    ax.axvspan(xs, xe, alpha=0.10, color=color, zorder=0)
    ax.annotate("", xy=(xe, y_top + 0.35), xytext=(xs, y_top + 0.35),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
    ax.text((xs + xe) / 2, y_top + 0.55, label,
            ha="center", va="bottom", fontsize=8.5,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8))

shade(rA_s, rA_e, "Region A\nHost Write +\nWeight Load", "#e67e22", y_top)
shade(rB_s, rB_e, "Region B\nCompute\nActivity",         "#8e44ad", y_top)
shade(rC_s, rC_e, "Region C\nHost Read\nResult",         "#c0392b", y_top)

ax.set_xlabel("Simulation time (ns)", fontsize=9)
ax.set_xlim(-20, t_end_ns * 1.02)
ax.set_ylim(-0.3, y_top + 1.2)
ax.set_title(
    "tb_top Co-Simulation Waveform — M3 Integration\n"
    "top → axi_stream_ctrl → compute_core  |  3×3 all-ones kernel, pixels 1–9",
    fontsize=10, pad=6)
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {PNG_PATH}")
