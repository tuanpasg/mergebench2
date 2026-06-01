import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

COLD_PATH = Path("wudi_loss_by_key_cold_start.csv")
WARM_PATH = Path("wudi_loss_by_key_warm_start.csv")

OUTDIR = Path("wudi_loss_deep_analysis")
OUTDIR.mkdir(exist_ok=True)

OUTLIER_IQR_FACTOR = 1.5
CONVERGENCE_RATIO = 0.95
# A key is considered to reach convergence when it already achieves
# 95% of its total loss reduction.


# ============================================================
# Utilities
# ============================================================

def load_and_enrich(path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mode"] = mode

    df["layer"] = df["key"].str.extract(r"model\.layers\.(\d+)\.").astype(int)
    df["block"] = df["key"].str.extract(r"\.(self_attn|mlp)\.")[0]
    df["proj"] = df["key"].str.extract(
        r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    )[0]

    return df


def get_loss_cols(df: pd.DataFrame):
    loss_cols = [c for c in df.columns if re.match(r"loss_\d+$", c)]
    loss_cols = sorted(loss_cols, key=lambda x: int(x.split("_")[1]))
    steps = np.array([int(c.split("_")[1]) for c in loss_cols])
    return loss_cols, steps


def add_convergence_metrics(df: pd.DataFrame, loss_cols, steps) -> pd.DataFrame:
    df = df.copy()

    start = df[loss_cols[0]]
    final = df[loss_cols[-1]]

    df["loss_start"] = start
    df["loss_final"] = final
    df["abs_drop"] = start - final
    df["rel_drop_pct"] = np.where(start > 0, 100 * df["abs_drop"] / start, np.nan)

    conv_steps = []

    for _, row in df.iterrows():
        losses = row[loss_cols].to_numpy(dtype=float)

        total_drop = losses[0] - losses[-1]

        if total_drop <= 0:
            conv_steps.append(np.nan)
            continue

        target_loss = losses[0] - CONVERGENCE_RATIO * total_drop

        reached = np.where(losses <= target_loss)[0]
        if len(reached) == 0:
            conv_steps.append(np.nan)
        else:
            conv_steps.append(steps[reached[0]])

    df["conv_step_95pct_drop"] = conv_steps

    return df


def mark_outliers(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()

    q1 = df[value_col].quantile(0.25)
    q3 = df[value_col].quantile(0.75)
    iqr = q3 - q1

    upper = q3 + OUTLIER_IQR_FACTOR * iqr

    df[f"{value_col}_outlier"] = df[value_col] > upper
    df[f"{value_col}_outlier_threshold"] = upper

    return df


def setup_step_axis(steps):
    # Space logged checkpoints evenly while keeping step values as labels.
    plt.xticks(np.arange(len(steps)), steps)


# ============================================================
# Load data
# ============================================================

cold = load_and_enrich(COLD_PATH, "cold_start")
warm = load_and_enrich(WARM_PATH, "warm_start")

loss_cols, steps = get_loss_cols(cold)
step_positions = np.arange(len(steps))

cold = add_convergence_metrics(cold, loss_cols, steps)
warm = add_convergence_metrics(warm, loss_cols, steps)

df = pd.concat([cold, warm], ignore_index=True)
df = mark_outliers(df, "loss_final")


# ============================================================
# 1. Global loss trajectory
# ============================================================

plt.figure(figsize=(8, 5))

for mode, g in df.groupby("mode"):
    mean_loss = [g[c].mean() for c in loss_cols]
    median_loss = [g[c].median() for c in loss_cols]

    plt.plot(step_positions, mean_loss, marker="o", label=f"{mode} mean")
    plt.plot(step_positions, median_loss, marker="x", linestyle="--", label=f"{mode} median")

setup_step_axis(steps)
plt.xlabel("WUDI optimization step")
plt.ylabel("Loss")
plt.title("Global WUDI loss trajectory")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "01_global_loss_trajectory_evenly_spaced_x.png", dpi=200)
plt.close()


# ============================================================
# 2. Projection-wise convergence speed
# ============================================================

proj_summary = (
    df.groupby(["mode", "proj"])
    .agg(
        key_count=("key", "count"),
        mean_start_loss=("loss_start", "mean"),
        mean_final_loss=("loss_final", "mean"),
        mean_rel_drop_pct=("rel_drop_pct", "mean"),
        median_conv_step_95pct_drop=("conv_step_95pct_drop", "median"),
        outlier_count=("loss_final_outlier", "sum"),
    )
    .reset_index()
)

proj_summary.to_csv(OUTDIR / "projection_summary.csv", index=False)

plt.figure(figsize=(9, 5))

for mode, g in df.groupby("mode"):
    proj_curve = (
        g.groupby("proj")[loss_cols]
        .mean()
        .reindex(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    )

    for proj, row in proj_curve.iterrows():
        if pd.isna(row).all():
            continue
        plt.plot(step_positions, row.values, marker="o", alpha=0.7, label=f"{mode}:{proj}")

setup_step_axis(steps)
plt.xlabel("WUDI optimization step")
plt.ylabel("Mean loss")
plt.title("Projection-wise convergence behavior")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUTDIR / "02_projection_convergence_evenly_spaced_x.png", dpi=200)
plt.close()


# ============================================================
# 3. Layer-wise final loss
# ============================================================

layer_summary = (
    df.groupby(["mode", "layer"])
    .agg(
        mean_start_loss=("loss_start", "mean"),
        mean_final_loss=("loss_final", "mean"),
        mean_rel_drop_pct=("rel_drop_pct", "mean"),
        median_conv_step_95pct_drop=("conv_step_95pct_drop", "median"),
        outlier_count=("loss_final_outlier", "sum"),
    )
    .reset_index()
)

layer_summary.to_csv(OUTDIR / "layer_summary.csv", index=False)

plt.figure(figsize=(10, 5))

for mode, g in layer_summary.groupby("mode"):
    plt.plot(g["layer"], g["mean_final_loss"], marker="o", label=mode)

plt.yscale("log")
plt.xlabel("Layer index")
plt.ylabel("Mean final loss")
plt.title("Layer-wise final WUDI loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "03_layerwise_final_loss.png", dpi=200)
plt.close()


# ============================================================
# 4. Layer-wise convergence speed
# ============================================================

plt.figure(figsize=(10, 5))

for mode, g in layer_summary.groupby("mode"):
    plt.plot(
        g["layer"],
        g["median_conv_step_95pct_drop"],
        marker="o",
        label=mode,
    )

plt.xlabel("Layer index")
plt.ylabel("Median step to reach 95% of total loss drop")
plt.title("Layer-wise convergence speed")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "04_layerwise_convergence_speed.png", dpi=200)
plt.close()


# ============================================================
# 5. Outlier report
# ============================================================

outliers = df[df["loss_final_outlier"]].copy()
outliers = outliers.sort_values("loss_final", ascending=False)

outliers[
    [
        "mode",
        "key",
        "layer",
        "block",
        "proj",
        "loss_start",
        "loss_final",
        "abs_drop",
        "rel_drop_pct",
        "conv_step_95pct_drop",
        "loss_final_outlier_threshold",
    ]
].to_csv(OUTDIR / "outlier_keys.csv", index=False)


# ============================================================
# 6. Top high-final-loss keys
# ============================================================

top_final = df.sort_values("loss_final", ascending=False).head(30)

top_final[
    [
        "mode",
        "key",
        "layer",
        "block",
        "proj",
        "loss_start",
        "loss_final",
        "rel_drop_pct",
        "conv_step_95pct_drop",
    ]
].to_csv(OUTDIR / "top_30_high_final_loss_keys.csv", index=False)

plt.figure(figsize=(10, 8))

labels = (
    top_final["mode"]
    + " | L"
    + top_final["layer"].astype(str)
    + " "
    + top_final["proj"].astype(str)
)

plt.barh(labels, top_final["loss_final"])
plt.xscale("log")
plt.xlabel("Final loss")
plt.ylabel("Key")
plt.title("Top 30 highest final-loss keys")
plt.grid(True, axis="x", alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTDIR / "05_top_30_high_final_loss_keys.png", dpi=200)
plt.close()


# ============================================================
# 7. Slow-converging keys
# ============================================================

slow = df.sort_values("conv_step_95pct_drop", ascending=False).head(30)

slow[
    [
        "mode",
        "key",
        "layer",
        "block",
        "proj",
        "loss_start",
        "loss_final",
        "rel_drop_pct",
        "conv_step_95pct_drop",
    ]
].to_csv(OUTDIR / "top_30_slow_converging_keys.csv", index=False)

plt.figure(figsize=(10, 8))

labels = (
    slow["mode"]
    + " | L"
    + slow["layer"].astype(str)
    + " "
    + slow["proj"].astype(str)
)

plt.barh(labels, slow["conv_step_95pct_drop"])
plt.xlabel("Step to reach 95% of total loss drop")
plt.ylabel("Key")
plt.title("Top 30 slow-converging keys")
plt.grid(True, axis="x", alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTDIR / "06_top_30_slow_converging_keys.png", dpi=200)
plt.close()


# ============================================================
# 8. Overall mode summary
# ============================================================

mode_summary = (
    df.groupby("mode")
    .agg(
        key_count=("key", "count"),
        mean_start_loss=("loss_start", "mean"),
        mean_final_loss=("loss_final", "mean"),
        median_start_loss=("loss_start", "median"),
        median_final_loss=("loss_final", "median"),
        mean_rel_drop_pct=("rel_drop_pct", "mean"),
        median_conv_step_95pct_drop=("conv_step_95pct_drop", "median"),
        max_final_loss=("loss_final", "max"),
        outlier_count=("loss_final_outlier", "sum"),
    )
    .reset_index()
)

mode_summary.to_csv(OUTDIR / "mode_summary.csv", index=False)


# ============================================================
# Print concise summary
# ============================================================

print("\n=== Mode summary ===")
print(mode_summary.to_string(index=False))

print("\n=== Projection summary ===")
print(proj_summary.to_string(index=False))

print(f"\nSaved all outputs to: {OUTDIR.resolve()}")
