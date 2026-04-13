import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Hardware parameters ──────────────────────────────────────────────────────
PEAK_COMPUTE = 10_000.0   # GFLOPS  (10 TFLOPS)
PEAK_BW      = 320.0      # GB/s
RIDGE        = PEAK_COMPUTE / PEAK_BW   # = 31.25 FLOP/byte

# ── Roofline curve ───────────────────────────────────────────────────────────
ai = np.logspace(-2, 4, 3000)
roofline = np.minimum(PEAK_BW * ai, PEAK_COMPUTE)

# ── Kernel points ────────────────────────────────────────────────────────────
ka_ai, ka_perf = 170.7,  10_000.0   # GEMM  — compute-bound
kb_ai, kb_perf =   0.083,    26.6   # Vec add — memory-bound

# ────────────────────────────────────────────────────────────────────────────
with plt.xkcd(scale=0.9, length=100, randomness=2):

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("#fffef5")
    ax.set_facecolor("#fffef5")

    # ── Roofline ─────────────────────────────────────────────────────────────
    ax.loglog(ai, roofline, color="black", linewidth=2.8, zorder=4,
              path_effects=[pe.withStroke(linewidth=4.5, foreground="#fffef5")])

    # ── Slope label on memory-bound segment ──────────────────────────────────
    slope_ai   = 0.55
    slope_perf = PEAK_BW * slope_ai
    ax.text(slope_ai, slope_perf * 2.6,
            f"slope = {PEAK_BW:.0f} GB/s",
            fontsize=10, rotation=44, ha="center", color="#333333",
            fontfamily="monospace")

    # ── Peak compute ceiling ──────────────────────────────────────────────────
    ax.axhline(y=PEAK_COMPUTE, color="black", linewidth=1.5,
               linestyle=(0, (6, 3)), alpha=0.55, xmin=0.48)
    ax.text(6000, PEAK_COMPUTE * 1.12,
            "Peak Compute = 10 TFLOPS", fontsize=10,
            ha="right", color="#333333", fontfamily="monospace")

    # ── Ridge point ───────────────────────────────────────────────────────────
    ax.plot(RIDGE, PEAK_COMPUTE, marker="D", markersize=10,
            color="black", zorder=6)
    ax.annotate(f"Ridge Point\n({RIDGE:.2f} FLOP/B, {PEAK_COMPUTE/1000:.0f} TFLOPS)",
                xy=(RIDGE, PEAK_COMPUTE),
                xytext=(RIDGE * 0.28, PEAK_COMPUTE * 0.38),
                arrowprops=dict(arrowstyle="->, head_width=0.35",
                                color="black", lw=1.4,
                                connectionstyle="arc3,rad=-0.25"),
                fontsize=9.5, ha="center", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fffef5",
                          ec="black", lw=1.2))

    # ── Memory-bound / Compute-bound region labels ────────────────────────────
    ax.text(0.018, 4000, "MEMORY\nBOUND", fontsize=11, color="#555555",
            style="italic", ha="left", fontfamily="monospace", alpha=0.75)
    ax.text(180,   38,   "COMPUTE\nBOUND",  fontsize=11, color="#555555",
            style="italic", ha="left", fontfamily="monospace", alpha=0.75)

    # ── Vertical dashed drop lines for both kernels ───────────────────────────
    for x_pt, y_pt in [(ka_ai, ka_perf), (kb_ai, kb_perf)]:
        ax.plot([x_pt, x_pt], [0.5, y_pt],
                color="gray", linewidth=1.0, linestyle=":", zorder=2)
        ax.plot([0.01, x_pt], [y_pt, y_pt],
                color="gray", linewidth=1.0, linestyle=":", zorder=2)

    # ── Kernel A — GEMM (compute-bound) ──────────────────────────────────────
    ax.plot(ka_ai, ka_perf, marker="o", markersize=14,
            color="#d62728", zorder=7,
            path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    ax.annotate(
        "  Kernel A: GEMM\n"
        "  AI = 170.7 FLOP/B\n"
        "  Perf = 10,000 GFLOPS\n"
        "  [COMPUTE-BOUND]",
        xy=(ka_ai, ka_perf),
        xytext=(ka_ai * 0.045, ka_perf * 0.28),
        arrowprops=dict(arrowstyle="->, head_width=0.35",
                        color="#d62728", lw=1.6,
                        connectionstyle="arc3,rad=0.20"),
        fontsize=9.5, ha="left", color="#d62728",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f0",
                  ec="#d62728", lw=1.3))

    # ── Kernel B — Vector Add (memory-bound) ─────────────────────────────────
    ax.plot(kb_ai, kb_perf, marker="^", markersize=14,
            color="#1f77b4", zorder=7,
            path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    ax.annotate(
        "Kernel B: Vector Add\n"
        "AI = 0.083 FLOP/B\n"
        "Perf = 26.6 GFLOPS\n"
        "[MEMORY-BOUND]",
        xy=(kb_ai, kb_perf),
        xytext=(kb_ai * 14, kb_perf * 0.10),
        arrowprops=dict(arrowstyle="->, head_width=0.35",
                        color="#1f77b4", lw=1.6,
                        connectionstyle="arc3,rad=-0.30"),
        fontsize=9.5, ha="left", color="#1f77b4",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4ff",
                  ec="#1f77b4", lw=1.3))

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(0.01, 10_000)
    ax.set_ylim(0.5,  80_000)

    ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)",
                  fontsize=13, fontfamily="monospace", labelpad=8)
    ax.set_ylabel("Attainable Performance  (GFLOPS)",
                  fontsize=13, fontfamily="monospace", labelpad=8)
    ax.set_title("Roofline Model — CMAN Hardware Platform\n"
                 "Peak Compute: 10 TFLOPS  |  Peak BW: 320 GB/s  |  Ridge: 31.25 FLOP/B",
                 fontsize=12, fontfamily="monospace", pad=14)

    # Custom tick labels to keep it readable
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y >= 1 else f"{y:.2f}"))

    ax.tick_params(labelsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black",      lw=2.5,         label="CMAN Roofline"),
        Line2D([0], [0], marker="o", color="#d62728", lw=0,
               markersize=10, label="Kernel A: GEMM (compute-bound)"),
        Line2D([0], [0], marker="^", color="#1f77b4", lw=0,
               markersize=10, label="Kernel B: Vector Add (memory-bound)"),
        Line2D([0], [0], marker="D", color="black",   lw=0,
               markersize=8,  label=f"Ridge Point ({RIDGE:.2f} FLOP/B)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              framealpha=0.85, facecolor="#fffef5",
              edgecolor="black", fontsize=9.5,
              prop={"family": "monospace"})

    plt.tight_layout()
    out = r"C:\Users\Husai\Ece-410-hardware-ai\codefest\cf02\cman_roofline.png"
    plt.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
