"""
Roofline plot — RTX 3070 Ti  |  gemm_naive vs gemm_tiled (N=1024, FP32)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Hardware specs ─────────────────────────────────────────────────────────────
PEAK_FLOPS_GFLOPS = 21_750.0   # RTX 3070 Ti  FP32  GFLOPS
PEAK_BW_GBS       = 608.0      # RTX 3070 Ti  DRAM  GB/s
RIDGE_AI          = PEAK_FLOPS_GFLOPS / PEAK_BW_GBS   # 35.76 FLOP/byte

# ── Measured kernel results ────────────────────────────────────────────────────
kernels = {
    "gemm_naive\n(AI = 0.25 F/B)": {
        "ai":      0.25,          # 1/(4 bytes/FLOP)  = 0.25 FLOP/byte
        "gflops":  1315.0,
        "color":   "#e74c3c",     # red
        "marker":  "o",
        "zorder":  5,
    },
    "gemm_tiled\n(AI = 256 F/B)": {
        "ai":      256.0,         # N/4 = 1024/4 = 256 FLOP/byte
        "gflops":  1325.0,
        "color":   "#2ecc71",     # green
        "marker":  "s",
        "zorder":  5,
    },
}

# ── Roofline curve ─────────────────────────────────────────────────────────────
ai_range = np.logspace(-2, 4, 2000)   # 0.01 … 10 000 FLOP/byte
perf_roof = np.minimum(ai_range * PEAK_BW_GBS, PEAK_FLOPS_GFLOPS)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# Grid
ax.grid(True, which="both", color="#2a2f3a", linewidth=0.6, linestyle="--", zorder=0)

# Roofline — memory slope + compute flat
ax.plot(ai_range, perf_roof,
        color="#f0c040", linewidth=2.8, zorder=3, label="Roofline")

# Shade memory-bound region
mem_ai   = ai_range[ai_range <= RIDGE_AI]
mem_perf = mem_ai * PEAK_BW_GBS
ax.fill_between(mem_ai, mem_perf, PEAK_FLOPS_GFLOPS,
                alpha=0.07, color="#3498db", zorder=1)
ax.fill_between(ai_range[ai_range >= RIDGE_AI], PEAK_FLOPS_GFLOPS,
                alpha=0.07, color="#9b59b6", zorder=1)

# Dashed lines: peak compute & ridge point
ax.axhline(PEAK_FLOPS_GFLOPS, color="#f0c040", linewidth=1.0,
           linestyle=":", alpha=0.5, zorder=2)
ax.axvline(RIDGE_AI, color="#f0c040", linewidth=1.0,
           linestyle=":", alpha=0.5, zorder=2)

# Ridge-point annotation
ax.annotate(f"Ridge point\n{RIDGE_AI:.1f} FLOP/byte",
            xy=(RIDGE_AI, PEAK_FLOPS_GFLOPS),
            xytext=(RIDGE_AI * 2.5, PEAK_FLOPS_GFLOPS * 0.62),
            color="#f0c040", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#f0c040", lw=1.2),
            zorder=6)

# Peak-compute label on the flat roof
ax.text(ai_range[-1] * 0.55, PEAK_FLOPS_GFLOPS * 1.04,
        f"Peak compute: {PEAK_FLOPS_GFLOPS:,.0f} GFLOPS",
        color="#f0c040", fontsize=9, ha="right", va="bottom", zorder=6)

# Memory slope label
slope_ai   = 0.18
slope_perf = slope_ai * PEAK_BW_GBS * 0.55
ax.text(slope_ai, slope_perf,
        f"Memory slope\n{PEAK_BW_GBS:.0f} GB/s",
        color="#3498db", fontsize=9, rotation=38,
        ha="center", va="center", zorder=6)

# Region labels
ax.text(0.025, PEAK_FLOPS_GFLOPS * 0.18, "Memory-\nBound",
        color="#3498db", fontsize=10, fontweight="bold",
        alpha=0.7, ha="left", va="center", zorder=6)
ax.text(RIDGE_AI * 9, PEAK_FLOPS_GFLOPS * 0.18, "Compute-\nBound",
        color="#9b59b6", fontsize=10, fontweight="bold",
        alpha=0.7, ha="center", va="center", zorder=6)

# ── Kernel data points ─────────────────────────────────────────────────────────
for label, k in kernels.items():
    ai, gf = k["ai"], k["gflops"]

    # Vertical dashed line from x-axis up to the data point
    ax.plot([ai, ai], [1e0, gf],
            color=k["color"], linewidth=0.9, linestyle=":", alpha=0.55, zorder=4)

    # Horizontal dashed line from y-axis to the data point
    ax.plot([ai_range[0], ai], [gf, gf],
            color=k["color"], linewidth=0.9, linestyle=":", alpha=0.55, zorder=4)

    # Marker
    ax.scatter(ai, gf,
               s=160, color=k["color"], marker=k["marker"],
               edgecolors="white", linewidths=0.8,
               zorder=k["zorder"])

    # Utilisation vs roofline
    roof_at_ai = min(ai * PEAK_BW_GBS, PEAK_FLOPS_GFLOPS)
    util_pct   = gf / roof_at_ai * 100

    # Label box
    offset_x = 3.0 if "tiled" in label else 1.6
    offset_y = 0.72 if "tiled" in label else 1.55
    ax.annotate(
        f"{label}\n{gf:,.0f} GFLOPS\n({util_pct:.1f}% of roof)",
        xy=(ai, gf),
        xytext=(ai * offset_x, gf * offset_y),
        color=k["color"], fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#1a1f2e",
                  ec=k["color"], lw=1.0, alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=k["color"], lw=1.1),
        zorder=7,
    )

# ── Axes ───────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.02, 5_000)
ax.set_ylim(1, PEAK_FLOPS_GFLOPS * 3.5)

ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)",
              color="white", fontsize=12, labelpad=8)
ax.set_ylabel("Performance  (GFLOPS)",
              color="white", fontsize=12, labelpad=8)

ax.tick_params(colors="white", which="both", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

# ── Title ──────────────────────────────────────────────────────────────────────
ax.set_title(
    "Roofline Model — RTX 3070 Ti  |  1024×1024 FP32 GEMM",
    color="white", fontsize=13, fontweight="bold", pad=14,
)
ax.text(0.5, 1.005,
        "gemm_naive vs gemm_tiled  (N=1024, T=8, measured with cudaEvent)",
        transform=ax.transAxes, color="#aaaaaa",
        fontsize=8.5, ha="center", va="bottom")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color="#f0c040", lw=2.5, label="Roofline"),
    mpatches.Patch(facecolor="#3498db", alpha=0.35, label="Memory-bound region"),
    mpatches.Patch(facecolor="#9b59b6", alpha=0.35, label="Compute-bound region"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
           markersize=9, label="gemm_naive  (1,315 GFLOPS)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ecc71",
           markersize=9, label="gemm_tiled  (1,325 GFLOPS)"),
]
leg = ax.legend(handles=legend_elements, loc="lower right",
                framealpha=0.85, facecolor="#1a1f2e",
                edgecolor="#444", labelcolor="white", fontsize=9)

plt.tight_layout()
out_path = r"C:\Users\Husai\Ece-410-hardware-ai\codefest\cf03\profiling\gemm_roofline.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
