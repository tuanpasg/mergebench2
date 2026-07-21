"""
plot_alignment_figures.py

Two paper-style figures derived from a full-model gradient-alignment log:

  Figure 1 (alignment trajectory): mean inter-task gradient alignment ρ̄(t)
  across all weight matrices, showing the descent from cooperative through
  zero crossing to antagonistic saturation. Per-key trajectories form a
  light bundle in the background; median and p10–p90 band are overlaid.

  Figure 2 (zero-crossing distribution): per-key step at which ρ̄ first
  crosses zero, stratified by projection type. Reveals the heterogeneity
  in cooperative-phase duration across MLP vs attention projections.

Expects a CSV with columns: key, step, rho_bar, and optionally total_loss.
The 'key' column must follow the HuggingFace-style naming pattern
model.layers.{N}.{block}.{proj}_proj.weight.

Usage:
    python plot_alignment_figures.py \\
        --csv wudi_loss_by_key.csv \\
        --out-dir figures/ \\
        --model-name "Gemma-2-2b"
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Colors
C_MEDIAN  = "#1f3864"   # navy — main alignment line
C_BAND    = "#7b9dd1"   # lighter navy for p10–p90 band
C_BUNDLE  = "#cfd8e8"   # very light for individual trajectories
C_ZERO    = "#666666"   # neutral grey for ρ̄=0 reference
C_COOP    = "#1d9e75"   # green band — cooperative regime
C_ANTI    = "#e24b4a"   # red band — antagonistic regime
C_PEAK    = "#e24b4a"   # red — crossing marker
C_LOSS    = "#4b4b4b"   # dark grey for total loss overlay

# Projection-type palette (MLP greens, ATTN warm tones)
PROJ_ORDER  = ["down", "gate", "up", "o", "v", "k", "q"]
PROJ_COLORS = {
    "down": "#0d6940",
    "gate": "#1d9e75",
    "up":   "#54c79b",
    "o":    "#f3a522",
    "v":    "#e8763e",
    "k":    "#d24b35",
    "q":    "#9a2e29",
}
PROJ_GROUP = {p: ("MLP" if p in {"down","gate","up"} else "ATTN") for p in PROJ_ORDER}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_LAYER_RE = re.compile(r"layers\.(\d+)\.")

def parse_key(k: str):
    """Extract (layer, projection_type) from a weight key."""
    layer = int(_LAYER_RE.search(k).group(1))
    parts = k.split(".")
    # e.g. ...self_attn.q_proj.weight  or  ...mlp.gate_proj.weight
    ptype = parts[4].replace("_proj", "").replace("weight", "").rstrip(".")
    return layer, ptype


def load_alignment(csv_path):
    df = pd.read_csv(csv_path)
    if "rho_bar" not in df.columns:
        raise ValueError(f"Expected 'rho_bar' column in {csv_path}; "
                         f"got {list(df.columns)}")
    df[["layer", "ptype"]] = df["key"].apply(lambda k: pd.Series(parse_key(k)))
    return df


def first_zero_crossing(kdf):
    """First step where rho_bar drops below 0, with linear interpolation."""
    kdf = kdf.sort_values("step").reset_index(drop=True)
    below = kdf[kdf["rho_bar"] < 0]
    if len(below) == 0:
        return np.nan
    s = int(below["step"].iloc[0])
    if s == 0:
        return 0.0
    prev_row = kdf[kdf["step"] == s - 1]
    if len(prev_row) == 0:
        return float(s)
    prev = prev_row.iloc[0]["rho_bar"]
    curr = kdf[kdf["step"] == s].iloc[0]["rho_bar"]
    if prev == curr:
        return float(s)
    return (s - 1) + prev / (prev - curr)


# ---------------------------------------------------------------------------
# Figure 1: alignment trajectory
# ---------------------------------------------------------------------------
def plot_trajectory(df, out_path, model_name=None, x_max=None):
    set_style()

    steps_all = np.sort(df["step"].unique())
    if x_max is None:
        x_max = int(steps_all.max())

    # Wide table: rows = step, cols = key, values = rho_bar
    pivot = df.pivot(index="step", columns="key", values="rho_bar").sort_index()
    pivot = pivot.loc[pivot.index <= x_max]

    steps = pivot.index.values
    median = pivot.median(axis=1).values
    p10 = pivot.quantile(0.10, axis=1).values
    p90 = pivot.quantile(0.90, axis=1).values
    total_loss = None
    if "total_loss" in df.columns:
        total_loss = (df[df["step"] <= x_max]
                      .groupby("step")["total_loss"]
                      .sum()
                      .reindex(steps)
                      .values)

    # Median zero crossing across keys
    zc = df.groupby("key").apply(first_zero_crossing).dropna()
    median_zc = float(zc.median())

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # --- regime shading (faint) ---
    # Cooperative: ρ̄ > 0
    ax.axhspan(0.0, 0.65, color=C_COOP, alpha=0.05, zorder=0)
    # Antagonistic: ρ̄ < 0
    ax.axhspan(-0.65, 0.0, color=C_ANTI, alpha=0.05, zorder=0)

    # --- individual key trajectories (light bundle) ---
    for col in pivot.columns:
        ax.plot(steps, pivot[col].values,
                color=C_BUNDLE, linewidth=0.55, alpha=0.32, zorder=1)

    # --- p10–p90 band + median ---
    ax.fill_between(steps, p10, p90, color=C_BAND, alpha=0.45,
                    linewidth=0, zorder=2)
    ax.plot(steps, median, color=C_MEDIAN, linewidth=2.0, zorder=3)
    ax2 = None
    if total_loss is not None:
        ax2 = ax.twinx()
        ax2.plot(steps, total_loss, color=C_LOSS, linewidth=1.5,
                 linestyle="--", alpha=0.9, zorder=4)
        ax2.set_ylabel("Total loss across keys")
        ax2.tick_params(axis="y", colors=C_LOSS)
        ax2.yaxis.label.set_color(C_LOSS)
        ax2.spines["right"].set_color(C_LOSS)

    # --- ρ̄ = 0 reference ---
    ax.axhline(0.0, color=C_ZERO, linewidth=0.9, linestyle="--",
               alpha=0.85, zorder=2)

    # --- median zero-crossing marker ---
    ax.axvline(median_zc, color=C_PEAK, linewidth=0.9, linestyle=":",
               alpha=0.7, zorder=2)
    ax.scatter([median_zc], [0.0], color=C_PEAK, s=40,
               edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate(f"median crossing\nstep {median_zc:.1f}",
                xy=(median_zc, 0.0),
                xytext=(median_zc + 0.10 * x_max, 0.15),
                fontsize=9, color="#222",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.8,
                                connectionstyle="arc3,rad=-0.15"))

    # --- regime labels (placed in left half to avoid legend collision) ---
    ax.text(x_max * 0.40, 0.50, r"cooperative regime  $\bar{\rho} > 0$",
            ha="center", va="center", fontsize=9,
            color=C_COOP, style="italic")
    ax.text(x_max * 0.65, -0.50, r"antagonistic regime  $\bar{\rho} < 0$",
            ha="center", va="center", fontsize=9,
            color=C_ANTI, style="italic")

    # --- axes ---
    ax.set_xlabel("Optimization step (Adam iteration)")
    ax.set_ylabel(r"Mean pairwise gradient alignment  $\bar{\rho}(t)$")
    ax.set_xlim(-0.5, x_max + 0.5)
    ax.set_ylim(-0.55, 0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5,
            color="#cccccc", alpha=0.6)

    # --- legend (compact, top-right) ---
    bundle_handle = mpatches.Patch(color=C_BUNDLE, alpha=0.7,
                                   label=f"per-layer (n={pivot.shape[1]})")
    band_handle = mpatches.Patch(color=C_BAND, alpha=0.45,
                                 label="p10–p90 across layers")
    median_handle = plt.Line2D([0], [0], color=C_MEDIAN, linewidth=2.0,
                               label="median across layers")
    legend_handles = [median_handle, band_handle, bundle_handle]
    if total_loss is not None:
        loss_handle = plt.Line2D([0], [0], color=C_LOSS, linewidth=1.5,
                                 linestyle="--",
                                 label="total loss across keys")
        legend_handles.append(loss_handle)
    ax.legend(handles=legend_handles,
              loc="upper right", frameon=True, framealpha=0.92,
              edgecolor="#cccccc", fancybox=False, fontsize=8.5,
              handlelength=1.8)

    if model_name:
        ax.set_title(model_name, fontsize=11, pad=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.png')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: zero-crossing distribution
# ---------------------------------------------------------------------------
def plot_distribution(df, out_path, model_name=None, bin_width=1.0, x_max=None):
    set_style()

    # Per-key zero crossings + projection type
    zc = (df.groupby("key")
            .apply(lambda g: pd.Series({
                "zc":    first_zero_crossing(g),
                "ptype": g["ptype"].iloc[0],
            }))
            .dropna(subset=["zc"]))

    if x_max is None:
        x_max = int(np.ceil(zc["zc"].quantile(0.98))) + 1
        x_max = max(x_max, 10)

    # Stacked histogram bins
    bins = np.arange(0, x_max + bin_width, bin_width)

    # Group keys by projection in plotting order
    by_ptype = {p: zc[zc["ptype"] == p]["zc"].values for p in PROJ_ORDER}

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # --- stacked histogram by projection type ---
    bottom = np.zeros(len(bins) - 1)
    for p in PROJ_ORDER:
        if len(by_ptype[p]) == 0:
            continue
        hist, _ = np.histogram(by_ptype[p], bins=bins)
        ax.bar(bins[:-1], hist, bottom=bottom, width=bin_width,
               align="edge", color=PROJ_COLORS[p],
               edgecolor="white", linewidth=0.4,
               label=f"{p} (n={len(by_ptype[p])}, {PROJ_GROUP[p]})")
        bottom += hist

    # --- summary statistics ---
    med = float(zc["zc"].median())
    p90 = float(zc["zc"].quantile(0.90))
    mean = float(zc["zc"].mean())

    ax.axvline(med, color=C_MEDIAN, linewidth=1.2, linestyle="-",
               alpha=0.85, zorder=4)
    ax.axvline(p90, color=C_PEAK, linewidth=1.2, linestyle="--",
               alpha=0.85, zorder=4)

    # Place a small stats box in the upper-left to avoid label overlap.
    top = bottom.max()
    stat_text = (f"median = {med:.1f}\n"
                 f"mean   = {mean:.1f}\n"
                 f"p90    = {p90:.1f}")
    ax.text(0.97, 0.97, stat_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="#bbbbbb", linewidth=0.7))

    # --- axes ---
    ax.set_xlabel("Per-layer zero-crossing step")
    ax.set_ylabel("Number of layers")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, top * 1.30)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5,
            color="#cccccc", alpha=0.6)

    # Legend (upper-left). Append median/p90 line handles so the user
    # knows what the colored verticals mean.
    handles, labels = ax.get_legend_handles_labels()
    median_line = plt.Line2D([0], [0], color=C_MEDIAN, linewidth=1.2,
                             linestyle="-", label="median")
    p90_line = plt.Line2D([0], [0], color=C_PEAK, linewidth=1.2,
                          linestyle="--", label="p90")
    ax.legend(handles + [median_line, p90_line],
              labels + ["median", "p90"],
              loc="upper left", frameon=False,
              fontsize=8.5, ncol=1, labelspacing=0.3,
              handlelength=1.4, handletextpad=0.6)

    if model_name:
        ax.set_title(f"{model_name}  ·  n={len(zc)} layers",
                     fontsize=11, pad=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.png')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="Per-key alignment CSV (columns: key, step, rho_bar).")
    ap.add_argument("--out-dir", default="figures",
                    help="Directory to write the figures into.")
    ap.add_argument("--model-name", default=None,
                    help="Optional model name to display as title.")
    ap.add_argument("--trajectory-x-max", type=int, default=None,
                    help="Crop trajectory plot at this step (default: full range).")
    ap.add_argument("--distribution-x-max", type=int, default=None,
                    help="Crop distribution histogram at this step "
                         "(default: just past p98).")
    args = ap.parse_args()

    df = load_alignment(args.csv)
    out_dir = Path(args.out_dir)

    plot_trajectory(
        df,
        out_dir / "fig_alignment_trajectory.pdf",
        model_name=args.model_name,
        x_max=args.trajectory_x_max,
    )
    plot_distribution(
        df,
        out_dir / "fig_zero_crossing_distribution.pdf",
        model_name=args.model_name,
        x_max=args.distribution_x_max,
    )


if __name__ == "__main__":
    main()
