"""
Generate a waveform PNG for the tb_compute_core simulation.
Signals: clk, rst, weight AXI4-Stream, pixel AXI4-Stream, result AXI4-Stream.
Timeline is reconstructed cycle-accurately from the testbench stimulus.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Timing constants  (x-axis = clock cycles; each cycle = 10 ns)
# ---------------------------------------------------------------------------
# After reset (4 posedges = cycles 1-4, rst deasserts at 3.6 cycles):
#   Cycle 4.6 : weight[0] driven; sampled at cycle 5 posedge
#   Cycle 13.6: weight[8] driven; tvalid deasserted at cycle 13.6+1
#   Cycle 14.6: pixel[0][0]=1 driven; sampled at cycle 15
#   ...
#   Cycle 50.6: last pixel (pixel[5][5]=36) driven; tvalid deasserted at 50.6+1
#
# Pipeline latency = 6 cycles (5 adder stages + 1 output register)
# win_valid first fires at pixel[2][2] (index 14) → sample cycle 29
#   → result_tvalid rises at cycle 29+6=35 (one cycle later the output reg fires,
#     making result visible at cycle 36 in the output checker)
# Outputs appear in 4 bursts of 4 (one burst per output row, separated by 2-cycle
# gaps while the window crosses columns 0-1 of the next input row):
#   output[0][0..3]: fire cycles 35, 36, 37, 38
#   output[1][0..3]: fire cycles 41, 42, 43, 44
#   output[2][0..3]: fire cycles 47, 48, 49, 50
#   output[3][0..3]: fire cycles 53, 54, 55, 56

TMAX = 68  # cycles shown

# ---------------------------------------------------------------------------
# Signal construction helpers
# ---------------------------------------------------------------------------

def events_to_xy(events, t_max, epsilon=0.02):
    """
    Convert a list of (time, value) change-events into x/y arrays for
    matplotlib step-function plotting.  epsilon offsets the first sample
    so transitions appear on the correct side of the clock edge.
    """
    events = sorted(events, key=lambda e: e[0])
    xs, ys = [], []
    for i, (t, v) in enumerate(events):
        t_next = events[i+1][0] if i+1 < len(events) else t_max
        xs += [t + epsilon, t_next + epsilon]
        ys += [v, v]
    return np.array(xs), np.array(ys)


def draw_bus(ax, events, t_max, y_mid, height, color, label_fmt="{}", fontsize=6):
    """
    Draw a bus (multi-valued) signal as coloured bands with value annotations.
    events : list of (time, value); value==0 → don't annotate (invalid / X)
    """
    events = sorted(events, key=lambda e: e[0])
    for i, (t, v) in enumerate(events):
        t_next = events[i+1][0] if i+1 < len(events) else t_max
        if t_next <= t:
            continue
        if v != 0:
            ax.fill_between([t, t_next], [y_mid - height/2]*2, [y_mid + height/2]*2,
                            color=color, alpha=0.30, linewidth=0)
            ax.plot([t, t_next], [y_mid - height/2]*2, color=color, linewidth=0.8)
            ax.plot([t, t_next], [y_mid + height/2]*2, color=color, linewidth=0.8)
            # transition ticks
            ax.plot([t, t], [y_mid - height/2, y_mid + height/2], color=color, linewidth=0.8)
            ax.plot([t_next, t_next], [y_mid - height/2, y_mid + height/2],
                    color=color, linewidth=0.8)
            mid = (t + t_next) / 2
            ax.text(mid, y_mid, label_fmt.format(v),
                    ha='center', va='center', fontsize=fontsize,
                    color=color, fontweight='bold')
        else:
            # show as flat low line
            ax.plot([t, t_next], [y_mid - height/2]*2, color='#aaaaaa', linewidth=0.5)
            ax.plot([t, t_next], [y_mid + height/2]*2, color='#aaaaaa', linewidth=0.5)


# ---------------------------------------------------------------------------
# Build event lists
# ---------------------------------------------------------------------------

# ── clk ──────────────────────────────────────────────────────────────────────
clk_evts = []
for i in range(TMAX * 2 + 2):
    clk_evts.append((i * 0.5, 1 if i % 2 == 1 else 0))

# ── rst ──────────────────────────────────────────────────────────────────────
rst_evts = [(0, 1), (3.6, 0)]

# ── s_axis_weight_tvalid ─────────────────────────────────────────────────────
wt_valid_evts = [(0, 0), (4.6, 1), (13.6, 0)]

# ── s_axis_weight_tdata ──────────────────────────────────────────────────────
weights = [1, 2, 1, 2, 4, 2, 1, 2, 1]
wt_data_evts = [(0, 0)] + [(4.6 + i, w) for i, w in enumerate(weights)] + [(13.6, 0)]

# ── s_axis_weight_tlast ──────────────────────────────────────────────────────
wt_last_evts = [(0, 0), (12.6, 1), (13.6, 0)]

# ── s_axis_pixel_tvalid ──────────────────────────────────────────────────────
px_valid_evts = [(0, 0), (14.6, 1), (50.6, 0)]

# ── s_axis_pixel_tdata ───────────────────────────────────────────────────────
px_data_evts = [(0, 0)] + [(14.6 + i, i + 1) for i in range(36)] + [(50.6, 0)]

# ── s_axis_pixel_tlast ───────────────────────────────────────────────────────
px_last_evts = [(0, 0)]
for row in range(6):
    t_on = 14.6 + row * 6 + 5
    px_last_evts += [(t_on, 1), (t_on + 1.0, 0)]

# ── m_axis_result_tvalid ─────────────────────────────────────────────────────
# Outputs in 4 bursts, each 4 wide, separated by 2-cycle gaps
result_fire_cycles = (
    [35, 36, 37, 38] +
    [41, 42, 43, 44] +
    [47, 48, 49, 50] +
    [53, 54, 55, 56]
)
res_valid_evts = [(0, 0)]
prev = None
for fc in sorted(result_fire_cycles):
    if prev is None or fc > prev + 1:
        if prev is not None:
            res_valid_evts.append((prev + 1, 0))
        res_valid_evts.append((fc, 1))
    prev = fc
res_valid_evts.append((prev + 1, 0))

# ── m_axis_result_tdata ──────────────────────────────────────────────────────
result_outputs = [8, 9, 10, 11,  14, 15, 16, 17,  20, 21, 22, 23,  26, 27, 28, 29]
res_data_evts = [(0, 0)]
for i, fc in enumerate(result_fire_cycles):
    res_data_evts.append((fc, result_outputs[i]))
res_data_evts.append((TMAX, 0))

# ── m_axis_result_tlast ──────────────────────────────────────────────────────
res_last_evts = [(0, 0)]
for row in range(4):
    t_on = result_fire_cycles[row * 4 + 3]
    res_last_evts += [(t_on, 1), (t_on + 1.0, 0)]

# ---------------------------------------------------------------------------
# Plot layout
# ---------------------------------------------------------------------------
SIGNALS = [
    ("clk",                  "digital", clk_evts,        '#1a6faf', 0.75),
    ("rst",                  "digital", rst_evts,         '#e05c5c', 0.75),
    ("wt_tvalid",            "digital", wt_valid_evts,    '#2ca02c', 0.65),
    ("wt_tdata",             "bus",     wt_data_evts,     '#2ca02c', 0.65),
    ("wt_tlast",             "digital", wt_last_evts,     '#2ca02c', 0.55),
    ("px_tvalid",            "digital", px_valid_evts,    '#ff7f0e', 0.65),
    ("px_tdata",             "bus",     px_data_evts,     '#ff7f0e', 0.65),
    ("px_tlast",             "digital", px_last_evts,     '#ff7f0e', 0.55),
    ("result_tvalid",        "digital", res_valid_evts,   '#9467bd', 0.65),
    ("result_tdata",         "bus",     res_data_evts,    '#9467bd', 0.65),
    ("result_tlast",         "digital", res_last_evts,    '#9467bd', 0.55),
]

ROW_HEIGHT = 1.0      # vertical space per signal row
VGAP      = 0.25      # gap between rows
SIG_H     = 0.55      # actual drawn height of each signal

fig_h = len(SIGNALS) * (ROW_HEIGHT + VGAP) + 1.0
fig, ax = plt.subplots(figsize=(18, fig_h), dpi=150)

LABEL_X = -1.8  # x position for signal labels

for idx, (name, kind, evts, color, amp) in enumerate(SIGNALS):
    y_base = -(idx * (ROW_HEIGHT + VGAP))
    y_mid  = y_base + SIG_H / 2

    # Signal name label
    ax.text(LABEL_X, y_mid, name, ha='right', va='center',
            fontsize=8, fontfamily='monospace', color='#222222')

    if kind == "digital":
        xs, ys = events_to_xy(evts, TMAX)
        ys_scaled = y_base + ys * amp
        ax.plot(xs, ys_scaled, color=color, linewidth=1.0)
        ax.fill_between(xs, y_base, ys_scaled, color=color, alpha=0.18, step=None)

    elif kind == "bus":
        draw_bus(ax, evts, TMAX, y_mid=y_mid,
                 height=SIG_H * 0.85, color=color,
                 label_fmt="{}", fontsize=6)

# ---------------------------------------------------------------------------
# Annotations: phase labels
# ---------------------------------------------------------------------------
phase_y = -(len(SIGNALS) * (ROW_HEIGHT + VGAP)) - 0.4

def phase_bracket(ax, t0, t1, label, color, py):
    ax.annotate("", xy=(t1, py), xytext=(t0, py),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
    ax.text((t0+t1)/2, py - 0.35, label, ha='center', va='top',
            fontsize=7, color=color)

phase_bracket(ax,  0,   3.6, "reset",           '#e05c5c', phase_y)
phase_bracket(ax,  4.6, 13.6,"weight load",     '#2ca02c', phase_y)
phase_bracket(ax, 14.6, 50.6,"pixel stream →",  '#ff7f0e', phase_y)
phase_bracket(ax, 35,   56,  "← outputs",       '#9467bd', phase_y - 0.8)

# ---------------------------------------------------------------------------
# Grid lines at every cycle
# ---------------------------------------------------------------------------
for cyc in range(0, TMAX + 1, 5):
    ax.axvline(x=cyc, color='#cccccc', linewidth=0.5, zorder=0)
    ax.text(cyc, 1.2, str(cyc), ha='center', va='bottom',
            fontsize=7, color='#666666')

ax.text(TMAX / 2, 1.7, 'Clock cycle', ha='center', va='bottom',
        fontsize=9, color='#444444')

# ---------------------------------------------------------------------------
# Decoration
# ---------------------------------------------------------------------------
ax.set_xlim(LABEL_X - 0.5, TMAX + 0.5)
y_top = 1.5
y_bot = phase_y - 1.4
ax.set_ylim(y_bot, y_top)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])

fig.suptitle("compute_core simulation — AXI4-Stream waveform\n"
             "6×6 image  ·  3×3 Gaussian kernel  ·  QUANT_SHIFT=4",
             fontsize=11, y=0.99, va='top')

# Legend patches
legend_items = [
    mpatches.Patch(color='#1a6faf', label='clock / reset'),
    mpatches.Patch(color='#2ca02c', label='weight stream'),
    mpatches.Patch(color='#ff7f0e', label='pixel stream'),
    mpatches.Patch(color='#9467bd', label='result stream'),
]
ax.legend(handles=legend_items, loc='upper right',
          fontsize=8, framealpha=0.7, ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(
    r"C:\Users\Husai\Ece-410-hardware-ai\project\m2\sim\waveform.png",
    dpi=150, bbox_inches='tight', facecolor='white'
)
print("Saved waveform.png")
