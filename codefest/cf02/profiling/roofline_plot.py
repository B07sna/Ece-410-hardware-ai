import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Hardware parameters ─────────────────────────────────────────────────────
PEAK_COMPUTE   = 768.0    # GFLOPS  (i7-12700KF)
PEAK_BW        = 76.8     # GB/s
RIDGE_POINT    = PEAK_COMPUTE / PEAK_BW   # = 10.0 FLOP/byte

# ── Roofline x-axis ─────────────────────────────────────────────────────────
ai = np.logspace(-1, 3, 2000)   # 0.1 … 1000 FLOP/byte

roofline = np.minimum(PEAK_BW * ai, PEAK_COMPUTE)   # GFLOPS

# ── Kernel points ────────────────────────────────────────────────────────────
# MobileNetV2 dominant kernel: depthwise conv [1,384,14,14] × [384,1,3,3]
kernel_ai    = 2.20     # FLOP/byte
kernel_perf  = PEAK_BW * kernel_ai   # memory-bound → BW × AI
kernel_label = "MobileNetV2\nDepthwise Conv\n(AI = 2.20 FLOP/B)"

# Hypothetical accelerator operating point
accel_ai    = 50.0      # FLOP/byte
accel_peak  = 2000.0    # GFLOPS
accel_perf  = min(accel_peak, accel_ai * PEAK_BW)   # still BW-bound if below ridge
# For the accelerator we treat it as running on the SAME memory system but
# showing its compute ceiling separately. Plot its achieved/attainable perf.
# Since 50 FLOP/B × 76.8 GB/s = 3840 GFLOPS > 2000 GFLOPS → compute-bound.
accel_attainable = min(accel_peak, accel_ai * PEAK_BW)
accel_label = "HW Accelerator\n(AI = 50 FLOP/B,\nPeak = 2000 GFLOPS)"

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# Grid
ax.grid(True, which="both", color="#2a2d3e", linewidth=0.6, linestyle="--")

# ── Roofline (i7-12700KF) ────────────────────────────────────────────────────
ax.loglog(ai, roofline, color="#00c8ff", linewidth=2.5, label="i7-12700KF Roofline")

# Shade memory-bound vs compute-bound regions
ax.axvline(x=RIDGE_POINT, color="#00c8ff", linewidth=1.2, linestyle=":", alpha=0.6)
ax.fill_betweenx([0.1, PEAK_COMPUTE], 0.1, RIDGE_POINT,
                 alpha=0.07, color="#00c8ff")
ax.fill_betweenx([0.1, PEAK_COMPUTE], RIDGE_POINT, 1000,
                 alpha=0.07, color="#ff9900")

# ── Accelerator ceiling line ─────────────────────────────────────────────────
accel_roof = np.minimum(PEAK_BW * ai, accel_peak)
ax.loglog(ai, accel_roof, color="#ff6b35", linewidth=2.0, linestyle="--",
          label="HW Accelerator Roofline (2000 GFLOPS)")

# ── Ridge point annotation ───────────────────────────────────────────────────
ax.plot(RIDGE_POINT, PEAK_COMPUTE, marker="D", markersize=9,
        color="#00c8ff", zorder=5)
ax.annotate(f"Ridge Point\n({RIDGE_POINT:.0f} FLOP/B, {PEAK_COMPUTE:.0f} GFLOPS)",
            xy=(RIDGE_POINT, PEAK_COMPUTE),
            xytext=(RIDGE_POINT * 1.6, PEAK_COMPUTE * 0.55),
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2),
            color="#aaaaaa", fontsize=9, ha="left")

# ── MobileNetV2 kernel point ─────────────────────────────────────────────────
mv2_perf = PEAK_BW * kernel_ai   # memory-bound: 76.8 × 2.20 = 168.96 GFLOPS
ax.plot(kernel_ai, mv2_perf, marker="o", markersize=12,
        color="#ff4d6d", zorder=6, label="MobileNetV2 Dominant Kernel")
ax.annotate(kernel_label,
            xy=(kernel_ai, mv2_perf),
            xytext=(kernel_ai * 2.8, mv2_perf * 0.35),
            arrowprops=dict(arrowstyle="->", color="#ff4d6d", lw=1.4),
            color="#ff4d6d", fontsize=9.5, ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1e1e2e", ec="#ff4d6d", lw=1))

# ── Accelerator operating point ──────────────────────────────────────────────
# At AI=50, BW roof = 76.8×50=3840 GFLOPS > 2000 → compute-bound → perf=2000
ax.plot(accel_ai, accel_peak, marker="^", markersize=13,
        color="#a8ff78", zorder=6, label="HW Accelerator Operating Point")
ax.annotate(accel_label,
            xy=(accel_ai, accel_peak),
            xytext=(accel_ai * 2.2, accel_peak * 1.35),
            arrowprops=dict(arrowstyle="->", color="#a8ff78", lw=1.4),
            color="#a8ff78", fontsize=9.5, ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1e1e2e", ec="#a8ff78", lw=1))

# ── Ceiling labels ───────────────────────────────────────────────────────────
ax.axhline(y=PEAK_COMPUTE, color="#00c8ff", linewidth=0.8, linestyle=":",
           alpha=0.5, xmin=0.55)
ax.text(800, PEAK_COMPUTE * 1.08, f"i7-12700KF Peak: {PEAK_COMPUTE:.0f} GFLOPS",
        color="#00c8ff", fontsize=8.5, ha="right")

ax.axhline(y=accel_peak, color="#ff6b35", linewidth=0.8, linestyle=":",
           alpha=0.5, xmin=0.55)
ax.text(800, accel_peak * 1.08, f"Accelerator Peak: {accel_peak:.0f} GFLOPS",
        color="#ff6b35", fontsize=8.5, ha="right")

# Memory-bound / Compute-bound region labels
ax.text(0.15, 1.5, "Memory\nBound", color="#00c8ff", fontsize=9,
        alpha=0.7, style="italic")
ax.text(18, 1.5, "Compute\nBound", color="#ff9900", fontsize=9,
        alpha=0.7, style="italic")

# BW slope label
mid_ai   = 1.5
mid_perf = PEAK_BW * mid_ai
ax.text(mid_ai, mid_perf * 2.2,
        f"BW = {PEAK_BW:.1f} GB/s",
        color="#00c8ff", fontsize=8.5, rotation=38, alpha=0.85)

# ── Axes formatting ──────────────────────────────────────────────────────────
ax.set_xlim(0.1, 1000)
ax.set_ylim(0.5, 8000)
ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", color="#cccccc", fontsize=12)
ax.set_ylabel("Performance (GFLOPS)", color="#cccccc", fontsize=12)
ax.set_title("Roofline Model — Intel i7-12700KF\nMobileNetV2 FP32 Dominant Kernel vs HW Accelerator",
             color="white", fontsize=13, pad=14)

ax.tick_params(colors="#aaaaaa", which="both")
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")

legend = ax.legend(loc="upper left", framealpha=0.25, facecolor="#1a1a2e",
                   edgecolor="#444466", labelcolor="#cccccc", fontsize=9)

plt.tight_layout()
out = r"C:\Users\Husai\Ece-410-hardware-ai\codefest\cf02\profiling\roofline_project.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
