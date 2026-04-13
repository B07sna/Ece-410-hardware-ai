import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = r"C:\Users\Husai\Ece-410-hardware-ai\project\m1\system_diagram.png"

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("#F0F4F8")
ax.set_facecolor("#F0F4F8")

# ── Palette ───────────────────────────────────────────────────────────────────
C_HOST    = "#1E3A5F"   # deep navy  — Host CPU
C_AXI     = "#0077B6"   # ocean blue — AXI4-Stream
C_CHIPLET = "#B45309"   # amber      — chiplet boundary (dashed)
C_ENGINE  = "#065F46"   # emerald    — Compute Engine
C_SRAM    = "#4C1D95"   # violet     — SRAM
C_ARROW   = "#374151"   # dark gray  — arrows
C_BG      = "#F0F4F8"   # background
WHITE     = "#FFFFFF"
LTGRAY    = "#E5E7EB"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: rounded box with label
# ─────────────────────────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, color, label, sublabel=None,
             fontsize=11, radius=0.25, alpha=1.0,
             linestyle="-", linewidth=2.0, text_color=WHITE):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=linewidth,
                         edgecolor=color,
                         facecolor=color if alpha == 1.0 else "none",
                         linestyle=linestyle,
                         alpha=1.0,
                         zorder=3)
    ax.add_patch(box)

    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.22, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=text_color, zorder=5,
                fontfamily="DejaVu Sans")
        ax.text(cx, cy - 0.28, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=text_color, alpha=0.88, zorder=5,
                fontfamily="DejaVu Sans")
    else:
        ax.text(cx, cy, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=text_color, zorder=5,
                fontfamily="DejaVu Sans")

def draw_arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=2.2,
               both=True, label=None, label_offset=(0, 0.35)):
    style = "<->" if both else "->"
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                          arrowstyle=f"{style}, head_width=0.18, head_length=0.18",
                          linewidth=lw, color=color, zorder=4,
                          connectionstyle="arc3,rad=0.0")
    ax.add_patch(arr)
    if label:
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8.5, color=color, fontstyle="italic",
                fontfamily="DejaVu Sans", zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# (1) Host CPU / FPGA SoC
# ─────────────────────────────────────────────────────────────────────────────
HOST_X, HOST_Y, HOST_W, HOST_H = 0.6, 2.8, 2.8, 3.4
draw_box(ax, HOST_X, HOST_Y, HOST_W, HOST_H,
         color=C_HOST,
         label="Host CPU /\nFPGA SoC",
         sublabel="ARM Cortex-A53\nZynq UltraScale+",
         fontsize=11.5)

# Number badge
ax.text(HOST_X + 0.22, HOST_Y + HOST_H - 0.28, "1",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_HOST,
        bbox=dict(boxstyle="circle,pad=0.18", fc=WHITE, ec=C_HOST, lw=1.5),
        zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# (3) Chiplet Boundary — dashed box (drawn before internals so it's behind)
# ─────────────────────────────────────────────────────────────────────────────
CHIP_X, CHIP_Y, CHIP_W, CHIP_H = 5.8, 0.6, 9.4, 7.8
chip_box = FancyBboxPatch((CHIP_X, CHIP_Y), CHIP_W, CHIP_H,
                           boxstyle="round,pad=0,rounding_size=0.35",
                           linewidth=2.8,
                           edgecolor=C_CHIPLET,
                           facecolor="#FEF3C7",   # very light amber
                           linestyle=(0, (8, 4)),
                           zorder=2)
ax.add_patch(chip_box)

# Chiplet label at top
ax.text(CHIP_X + CHIP_W / 2, CHIP_Y + CHIP_H - 0.52,
        "Chiplet Boundary",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=C_CHIPLET, zorder=6, fontfamily="DejaVu Sans")

# Number badge
ax.text(CHIP_X + 0.30, CHIP_Y + CHIP_H - 0.30, "3",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_CHIPLET,
        bbox=dict(boxstyle="circle,pad=0.18", fc=WHITE, ec=C_CHIPLET, lw=1.5),
        zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# (4) Depthwise Conv Compute Engine  (top-right inside chiplet)
# ─────────────────────────────────────────────────────────────────────────────
ENG_X, ENG_Y, ENG_W, ENG_H = 6.6, 5.0, 7.8, 2.7
draw_box(ax, ENG_X, ENG_Y, ENG_W, ENG_H,
         color=C_ENGINE,
         label="Depthwise Conv\nCompute Engine",
         sublabel="384-channel MAC Array\n1354752 FLOPs / inference",
         fontsize=11.5)

ax.text(ENG_X + 0.22, ENG_Y + ENG_H - 0.28, "4",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_ENGINE,
        bbox=dict(boxstyle="circle,pad=0.18", fc=WHITE, ec=C_ENGINE, lw=1.5),
        zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# (5) On-Chip SRAM  (bottom-right inside chiplet)
# ─────────────────────────────────────────────────────────────────────────────
SRAM_X, SRAM_Y, SRAM_W, SRAM_H = 6.6, 1.5, 7.8, 2.7
draw_box(ax, SRAM_X, SRAM_Y, SRAM_W, SRAM_H,
         color=C_SRAM,
         label="On-Chip SRAM",
         sublabel="Weight Buffer + Activation Buffer\n615936 B / inference (FP32)",
         fontsize=11.5)

ax.text(SRAM_X + 0.22, SRAM_Y + SRAM_H - 0.28, "5",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_SRAM,
        bbox=dict(boxstyle="circle,pad=0.18", fc=WHITE, ec=C_SRAM, lw=1.5),
        zorder=6)

# ── Arrow: SRAM <-> Compute Engine (vertical, inside chiplet) ─────────────────
SRAM_CX = SRAM_X + SRAM_W / 2
draw_arrow(ax,
           SRAM_CX - 0.5, SRAM_Y + SRAM_H,
           SRAM_CX - 0.5, ENG_Y,
           color=C_ENGINE, lw=2.2, both=True,
           label="weights &\nactivations",
           label_offset=(-1.15, 0))
draw_arrow(ax,
           SRAM_CX + 0.5, ENG_Y,
           SRAM_CX + 0.5, SRAM_Y + SRAM_H,
           color=C_SRAM, lw=2.2, both=False,
           label="output\nwrite-back",
           label_offset=(1.1, 0))

# ─────────────────────────────────────────────────────────────────────────────
# (2) AXI4-Stream Interface — bus block between host and chiplet
# ─────────────────────────────────────────────────────────────────────────────
AXI_X, AXI_Y, AXI_W, AXI_H = 3.8, 3.15, 1.65, 2.7
draw_box(ax, AXI_X, AXI_Y, AXI_W, AXI_H,
         color=C_AXI,
         label="AXI4-Stream\nInterface",
         sublabel="128-bit @ 250 MHz\n4.0 GB/s  |  32 Gbps",
         fontsize=10)

ax.text(AXI_X + 0.22, AXI_Y + AXI_H - 0.28, "2",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_AXI,
        bbox=dict(boxstyle="circle,pad=0.18", fc=WHITE, ec=C_AXI, lw=1.5),
        zorder=6)

# ── Arrow: Host -> AXI block ──────────────────────────────────────────────────
draw_arrow(ax,
           HOST_X + HOST_W, HOST_Y + HOST_H / 2,
           AXI_X,           AXI_Y + AXI_H / 2,
           color=C_HOST, lw=2.4, both=True,
           label="DDR4 DMA", label_offset=(0, 0.35))

# ── Arrow: AXI block -> Chiplet boundary (input stream) ──────────────────────
draw_arrow(ax,
           AXI_X + AXI_W, AXI_Y + AXI_H * 0.68,
           CHIP_X,         AXI_Y + AXI_H * 0.68,
           color=C_AXI, lw=2.4, both=False,
           label="TDATA / TVALID /\nTREADY / TLAST",
           label_offset=(0, 0.42))

# ── Arrow: Chiplet -> AXI block (output stream) ───────────────────────────────
draw_arrow(ax,
           CHIP_X,         AXI_Y + AXI_H * 0.32,
           AXI_X + AXI_W, AXI_Y + AXI_H * 0.32,
           color=C_AXI, lw=2.4, both=False,
           label="output activations",
           label_offset=(0, -0.38))

# ─────────────────────────────────────────────────────────────────────────────
# Data flow annotation inside chiplet (input path Host -> SRAM -> Engine)
# ─────────────────────────────────────────────────────────────────────────────
# Arrow: chiplet left wall -> SRAM
draw_arrow(ax,
           CHIP_X + 0.1,  SRAM_Y + SRAM_H / 2,
           SRAM_X,        SRAM_Y + SRAM_H / 2,
           color=C_SRAM, lw=2.0, both=False,
           label="stream in", label_offset=(0, 0.32))

# Arrow: Engine -> chiplet right wall (output)
draw_arrow(ax,
           ENG_X + ENG_W,       ENG_Y + ENG_H / 2,
           CHIP_X + CHIP_W - 0.1, ENG_Y + ENG_H / 2,
           color=C_ENGINE, lw=2.0, both=False,
           label="stream out", label_offset=(0, 0.32))

# ─────────────────────────────────────────────────────────────────────────────
# Bandwidth callout box (bottom centre)
# ─────────────────────────────────────────────────────────────────────────────
callout_x, callout_y = 8.0, 0.08
callout_text = (
    "Required BW: 73.5 MB/s sustained  |  187.8 MB/s peak burst  "
    "|  Interface headroom: 54x"
)
ax.text(callout_x, callout_y, callout_text,
        ha="center", va="center", fontsize=8.8,
        color="#1F2937", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.35", fc=WHITE,
                  ec=C_AXI, lw=1.2, alpha=0.92),
        zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
ax.text(8.0, 8.65,
        "MobileNetV2 Depthwise Conv Accelerator — System Block Diagram",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="#111827", fontfamily="DejaVu Sans", zorder=7)
ax.text(8.0, 8.25,
        "ECE 410 | Hussain Alquzweni | Host: FPGA SoC (Zynq UltraScale+)  |  Interface: AXI4-Stream 128b @ 250 MHz",
        ha="center", va="center", fontsize=9, color="#6B7280",
        fontfamily="DejaVu Sans", zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_HOST,    edgecolor="white", label="(1) Host CPU / FPGA SoC"),
    mpatches.Patch(facecolor=C_AXI,     edgecolor="white", label="(2) AXI4-Stream Interface"),
    mpatches.Patch(facecolor="none",    edgecolor=C_CHIPLET,
                   linewidth=2, linestyle=(0,(5,3)),   label="(3) Chiplet Boundary"),
    mpatches.Patch(facecolor=C_ENGINE,  edgecolor="white", label="(4) Depthwise Conv Engine"),
    mpatches.Patch(facecolor=C_SRAM,    edgecolor="white", label="(5) On-Chip SRAM"),
]
leg = ax.legend(handles=legend_items, loc="lower left",
                bbox_to_anchor=(0.01, 0.01),
                framealpha=0.95, facecolor=WHITE,
                edgecolor="#D1D5DB", fontsize=9,
                title="Components", title_fontsize=9.5)
leg.get_title().set_fontweight("bold")

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
