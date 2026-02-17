"""
Generate FedDualAtt framework diagram for the MWSCAS 2026 paper.

Produces: docs/figures/feddualatt_framework.png  (and .pdf)

Layout:
  Left panel  – DualAttentionResNet1D model architecture + block detail
  Right panel – Per-round FL communication protocol
"""

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = "#2563EB"   # global params
ORANGE = "#D97706"   # local params
GREEN  = "#059669"   # shared / backbone
GRAY   = "#6B7280"
LIGHT_BLUE   = "#DBEAFE"
LIGHT_ORANGE = "#FEF3C7"
LIGHT_GREEN  = "#D1FAE5"
LIGHT_GRAY   = "#F3F4F6"
WHITE  = "#FFFFFF"


# ── helpers ───────────────────────────────────────────────────────────────────

def rounded_box(ax, xy, width, height, color, text, fontsize=8,
                text_color="white", bold=False, alpha=1.0, lw=1.2,
                edge_color=None):
    """Draw a rounded rectangle with centred text."""
    x, y = xy
    ec = edge_color if edge_color else color
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=2
    )
    ax.add_patch(box)
    fw = "bold" if bold else "normal"
    ax.text(x, y, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=fw, zorder=3,
            wrap=False)
    return box


def arrow(ax, x0, y0, x1, y1, color, lw=1.5, arrowsize=10, style="->",
          zorder=2):
    """Draw an arrow."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color,
                        lw=lw, mutation_scale=arrowsize),
        zorder=zorder
    )


def vline(ax, x, y0, y1, color, lw=1.4, ls="-"):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, ls=ls, zorder=2)


# ═════════════════════════════════════════════════════════════════════════════
# Figure
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(13.5, 6.8))
fig.patch.set_facecolor(WHITE)

# Two panels via explicit axes positions
# left panel: [left, bottom, width, height]
ax_arch  = fig.add_axes([0.01, 0.02, 0.44, 0.96])   # architecture
ax_fl    = fig.add_axes([0.52, 0.02, 0.47, 0.96])   # FL protocol

for ax in [ax_arch, ax_fl]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Architecture
# ═════════════════════════════════════════════════════════════════════════════

ax = ax_arch

# title
ax.text(0.5, 0.975, "DualAttentionResNet1D", ha="center", va="top",
        fontsize=10, fontweight="bold", color="#1F2937")

# ── backbone column (x=0.22, y positions top-to-bottom) ──────────────────────
bx = 0.22
bw, bh = 0.36, 0.055

blocks = [
    # (y_center, color, text, text_color)
    (0.905, LIGHT_GREEN, "Input  (batch × 12 × 5000)", "#1F2937"),
    (0.835, GREEN,       "Conv1d + BN + ReLU + MaxPool", WHITE),
    (0.765, GREEN,       "ResNet Layer 1   ×3 blocks",  WHITE),
    (0.695, GREEN,       "ResNet Layer 2   ×4 blocks",  WHITE),
    (0.625, GREEN,       "ResNet Layer 3   ×6 blocks",  WHITE),
    (0.555, GREEN,       "ResNet Layer 4   ×3 blocks",  WHITE),
    (0.485, GREEN,       "Positional Encoding",         WHITE),
]

for (yc, col, txt, tc) in blocks:
    rounded_box(ax, (bx, yc), bw, bh, col, txt,
                fontsize=7.5, text_color=tc)

# vertical connectors between backbone blocks
for i in range(len(blocks) - 1):
    y_top = blocks[i][0]   - bh / 2
    y_bot = blocks[i+1][0] + bh / 2
    vline(ax, bx, y_bot, y_top, GRAY, lw=1.2)

# gap between positional encoding and first transformer block
pos_enc_y  = blocks[-1][0]
tblock1_y  = 0.390
vline(ax, bx, tblock1_y + bh / 2 + 0.005, pos_enc_y - bh / 2 - 0.002,
      GRAY, lw=1.2, ls="--")
ax.text(bx + 0.22, (pos_enc_y + tblock1_y) / 2,
        "(batch × ~156 × 512)", ha="left", va="center",
        fontsize=6.5, color=GRAY, style="italic")

# ── two DualAtt transformer blocks ───────────────────────────────────────────
t_ys = [0.390, 0.295]
t_label = ["Dual Attention Block 1", "Dual Attention Block 2"]
for (ty, tlabel) in zip(t_ys, t_label):
    # outer frame
    frame = FancyBboxPatch(
        (bx - bw / 2 - 0.01, ty - bh / 2 - 0.005),
        bw + 0.02, bh + 0.01,
        boxstyle="round,pad=0.015",
        facecolor=LIGHT_GRAY, edgecolor="#9CA3AF",
        linewidth=1.2, zorder=1
    )
    ax.add_patch(frame)
    # label above
    ax.text(bx, ty + bh / 2 + 0.022, tlabel,
            ha="center", va="bottom", fontsize=7, color="#374151",
            fontweight="bold")
    # two head boxes side-by-side inside
    head_y = ty
    gx = bx - 0.095
    lx = bx + 0.095
    hw, hh = 0.145, 0.042
    rounded_box(ax, (gx, head_y), hw, hh,
                BLUE, f"Global Attn\n(G heads)",
                fontsize=6.5, text_color=WHITE)
    rounded_box(ax, (lx, head_y), hw, hh,
                ORANGE, f"Local Attn\n(L heads)",
                fontsize=6.5, text_color=WHITE)
    # combine arrow hint
    ax.annotate("", xy=(bx, head_y - hh / 2 - 0.012),
                xytext=(bx - 0.03, head_y),
                arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.8))
    ax.annotate("", xy=(bx, head_y - hh / 2 - 0.012),
                xytext=(bx + 0.03, head_y),
                arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.8))
    ax.text(bx, head_y - hh / 2 - 0.018, "concat + combine",
            ha="center", va="top", fontsize=5.5, color=GRAY, style="italic")

# connector between the two transformer blocks
vline(ax, bx, t_ys[1] + bh / 2 + 0.005,
           t_ys[0] - bh / 2 - 0.005, GRAY, lw=1.2)

# GAP and FC blocks
gap_y = 0.205
fc_y  = 0.140
out_y = 0.072

rounded_box(ax, (bx, gap_y), bw, bh, GREEN, "Global Avg Pool → (batch × 512)",
            fontsize=7.5, text_color=WHITE)
rounded_box(ax, (bx, fc_y),  bw, bh, GREEN, "FC (512 → 20)  + Sigmoid",
            fontsize=7.5, text_color=WHITE)
rounded_box(ax, (bx, out_y), bw, bh, LIGHT_GREEN,
            "Output  (batch × 20 labels)", fontsize=7.5, text_color="#1F2937")

for ya, yb in [(gap_y + bh/2, t_ys[1] - bh/2),
               (fc_y + bh/2, gap_y - bh/2),
               (out_y + bh/2, fc_y - bh/2)]:
    vline(ax, bx, yb, ya, GRAY, lw=1.2)

# ── legend ────────────────────────────────────────────────────────────────────
leg_patches = [
    mpatches.Patch(facecolor=GREEN,  edgecolor="white", label="Global (ResNet / FFN)"),
    mpatches.Patch(facecolor=BLUE,   edgecolor="white", label="Global attention (FedAvg)"),
    mpatches.Patch(facecolor=ORANGE, edgecolor="white", label="Local attention (per-client)"),
]
ax.legend(handles=leg_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.01),
          ncol=3, fontsize=6.2, frameon=True,
          framealpha=0.9, edgecolor="#D1D5DB")


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — FL Communication Protocol
# ═════════════════════════════════════════════════════════════════════════════

ax = ax_fl

ax.text(0.5, 0.975, "Federated Learning Protocol (per round)",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#1F2937")

# ── vertical layout constants ─────────────────────────────────────────────────
srv_x, srv_y = 0.50, 0.885
srv_w, srv_h = 0.85, 0.110

cli_y = 0.640
cli_w, cli_h = 0.210, 0.130

arr_top    = srv_y - srv_h / 2        # bottom edge of server
arr_bottom = cli_y + cli_h / 2        # top edge of clients
mid_y      = (arr_top + arr_bottom) / 2

# ── server box ────────────────────────────────────────────────────────────────
srv_frame = FancyBboxPatch(
    (srv_x - srv_w / 2, srv_y - srv_h / 2),
    srv_w, srv_h,
    boxstyle="round,pad=0.015",
    facecolor="#EFF6FF", edgecolor="#93C5FD",
    linewidth=1.8, zorder=1
)
ax.add_patch(srv_frame)
ax.text(srv_x, srv_y + 0.030, "Server", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#1D4ED8")
ax.text(srv_x - 0.14, srv_y - 0.015,
        "θ_global ← FedAvg(θ₁…θ₄)",
        ha="center", va="center", fontsize=7, color=BLUE)
ax.text(srv_x + 0.21, srv_y - 0.015,
        "φ_k ← φ_k  (not aggregated)",
        ha="center", va="center", fontsize=7, color=ORANGE)
ax.plot([srv_x + 0.04, srv_x + 0.04],
        [srv_y - srv_h / 2 + 0.010, srv_y + srv_h / 2 - 0.008],
        color="#93C5FD", lw=1.0, ls="--")

# ── four client boxes ─────────────────────────────────────────────────────────
client_names = ["Client 1\n(SPH)", "Client 2\n(PTB-XL)",
                "Client 3\n(SXPH)", "Client 4\n(G12EC)"]
client_xs = [0.115, 0.370, 0.630, 0.885]

for cx, cname in zip(client_xs, client_names):
    cframe = FancyBboxPatch(
        (cx - cli_w / 2, cli_y - cli_h / 2),
        cli_w, cli_h,
        boxstyle="round,pad=0.015",
        facecolor="#FFF7ED", edgecolor="#FCD34D",
        linewidth=1.5, zorder=1
    )
    ax.add_patch(cframe)
    ax.text(cx, cli_y + 0.025, cname, ha="center", va="center",
            fontsize=8, fontweight="bold", color="#92400E")
    ax.text(cx, cli_y - 0.020, "Train θ_global + φ_k\n(joint SGD, BCELoss)",
            ha="center", va="center", fontsize=6.5, color="#374151")

# ── arrows (down=blue, up=orange) ─────────────────────────────────────────────
dx = 0.025

for cx in client_xs:
    ax.annotate("", xy=(cx - dx, arr_bottom),
                xytext=(cx - dx, arr_top),
                arrowprops=dict(arrowstyle="-|>", color=BLUE,
                                lw=1.5, mutation_scale=9), zorder=3)
    ax.annotate("", xy=(cx + dx, arr_top),
                xytext=(cx + dx, arr_bottom),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                lw=1.5, mutation_scale=9), zorder=3)

# ── arrow labels (centred between clients 1-2 and 3-4) ───────────────────────
# downlink label — between clients 1 and 2
dl_x = (client_xs[0] + client_xs[1]) / 2
ax.text(dl_x, mid_y + 0.048, "↓ DOWNLINK", ha="center", va="center",
        fontsize=7, color=BLUE, fontweight="bold")
ax.text(dl_x, mid_y + 0.012, "θ_global  (local=0)", ha="center", va="center",
        fontsize=6.5, color=BLUE)
ax.text(dl_x, mid_y - 0.022, "+ φ_k  (own local dict)", ha="center", va="center",
        fontsize=6.5, color=ORANGE)

# uplink label — between clients 3 and 4
ul_x = (client_xs[2] + client_xs[3]) / 2
ax.text(ul_x, mid_y + 0.048, "↑ UPLINK", ha="center", va="center",
        fontsize=7, color=ORANGE, fontweight="bold")
ax.text(ul_x, mid_y + 0.012, "θ_global  (local=0)", ha="center", va="center",
        fontsize=6.5, color=BLUE)
ax.text(ul_x, mid_y - 0.022, "+ φ_k  + n_k", ha="center", va="center",
        fontsize=6.5, color=ORANGE)

# ── parameter partitioning boxes ──────────────────────────────────────────────
ax.text(0.5, 0.498, "Parameter Partitioning",
        ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1F2937")

param_items = [
    (BLUE,   LIGHT_BLUE,
     "θ_global  — Global Parameters  (FedAvg)",
     "ResNet1D-34 backbone  ·  Positional Encoding  ·  global_att / global_proj\n"
     "FFN  ·  LayerNorm  ·  FC classifier  →  aggregated every round"),
    (ORANGE, LIGHT_ORANGE,
     "φ_k  — Local Parameters  (per-client, client k)",
     "local_att  ·  local_proj  (L attention heads only)\n"
     "Stored on server per client  →  never aggregated, fully personalized"),
]

p_row_h = 0.112
for i, (ec, fc, title, body) in enumerate(param_items):
    py = 0.448 - i * (p_row_h + 0.015)
    pbox = FancyBboxPatch(
        (0.03, py - p_row_h / 2), 0.94, p_row_h,
        boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=ec,
        linewidth=1.4, zorder=1
    )
    ax.add_patch(pbox)
    ax.text(0.06, py + p_row_h * 0.18, title,
            ha="left", va="center", fontsize=7.5, fontweight="bold", color=ec)
    ax.text(0.06, py - p_row_h * 0.15, body,
            ha="left", va="center", fontsize=6.5, color="#374151")

# ── client training steps ─────────────────────────────────────────────────────
steps_top = 0.448 - 2 * (p_row_h + 0.015) - 0.020
ax.text(0.5, steps_top, "Client Steps Each Round",
        ha="center", va="top", fontsize=8, fontweight="bold", color="#1F2937")

steps = [
    (BLUE,   "① Load θ_global  (local positions = 0 from server)"),
    (ORANGE, "② Restore φ_k  (overwrite with own local params)"),
    (GREEN,  "③ Train all params jointly  (single BCELoss)"),
    (ORANGE, "④ Extract updated φ_k"),
    (BLUE,   "⑤ Zero local positions → clean θ_global serialization"),
    (GRAY,   "⑥ Upload  [θ_global (local=0),  φ_k,  n_k]"),
]

s_step = 0.060
sy0 = steps_top - 0.040
for i, (col, txt) in enumerate(steps):
    sy = sy0 - i * s_step
    ax.plot([0.035], [sy], 'o', ms=7, color=col, zorder=3)
    ax.text(0.065, sy, txt,
            ha="left", va="center", fontsize=7, color="#1F2937")


# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════

out_dir = Path(__file__).parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

for fmt in ("png", "pdf"):
    out_path = out_dir / f"feddualatt_framework.{fmt}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=WHITE, format=fmt)
    print(f"Saved: {out_path}")

plt.close(fig)
print("Done.")
