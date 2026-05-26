#!/usr/bin/env python3
"""
Compute and plot layer-wise pre-WUDI score S_l for Llama-3.2-3B task vectors.

For each 2D projection matrix in layer l:
    tau_m^TA = mean_i tau_i, or merged_model - base_model if --merged is set
    S_key = sum_i ||(tau_m^TA - tau_i) tau_i^T||_F^2 / ||tau_i||_F^2
Then S_l is the sum or mean of S_key over selected modules in that transformer layer.
The script also reports MLP-only and attention-only layer scores to compare
module-family contributions.

Default checkpoints follow MergeBench Llama-3.2-3B:
    base:        meta-llama/Llama-3.2-3B
    finetunes:   instruction, math, coding

Example:
python analyze_wudi_layer_score.py \
  --base meta-llama/Llama-3.2-3B \
  --finetunes MergeBench/Llama-3.2-3B_instruction MergeBench/Llama-3.2-3B_math MergeBench/Llama-3.2-3B_coding \
  --merged path-or-hf-id-for-merged-model \
  --outdir outs/wudi_layer_score \
  --device cuda \
  --dtype float16
"""

import argparse
import gc
import json
import math
import os
import re
import fnmatch
import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM


DEFAULT_INCLUDE = [
    r".*model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
    r".*model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
]

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
MODULE_RE = re.compile(
    r"model\.layers\.\d+\.(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="meta-llama/Llama-3.2-3B")
    p.add_argument(
        "--finetunes",
        nargs="+",
        default=[
            "MergeBench/Llama-3.2-3B_instruction",
            "MergeBench/Llama-3.2-3B_math",
            "MergeBench/Llama-3.2-3B_coding",
        ],
    )
    p.add_argument(
        "--task-names",
        nargs="+",
        default=["instruction", "math", "coding"],
        help="Only used for metadata; length should match --finetunes.",
    )
    p.add_argument(
        "--merged",
        default=None,
        help=(
            "Optional merged checkpoint/model id. When set, tau_m is computed as "
            "merged_model - base_model instead of mean_i tau_i."
        ),
    )
    p.add_argument("--outdir", default="outs/wudi_layer_score")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--chunk-rows", type=int, default=256)
    p.add_argument(
        "--include",
        nargs="*",
        default=DEFAULT_INCLUDE,
        help="Regex of parameter names to analyze. Default: Llama attention+MLP projection weights.",
    )
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument(
        "--layer-reduce",
        default="sum",
        choices=["sum", "mean"],
        help="How to aggregate module scores into a layer score.",
    )
    p.add_argument(
        "--normalize-by-param-count",
        action="store_true",
        help="Also compute score per parameter, useful because MLP matrices are larger than attention matrices.",
    )
    p.add_argument(
        "--load-all-finetunes",
        action="store_true",
        help="Faster but uses much more CPU RAM. Default loads finetunes sequentially and caches selected deltas to disk.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Where to cache task-vector tensors. Default: <outdir>/delta_cache",
    )
    return p.parse_args()


def dtype_from_str(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def match_any(patterns: Iterable[str], name: str) -> bool:
    for pat in patterns:
        try:
            if re.match(pat, name):
                return True
        except re.error:
            if fnmatch.fnmatchcase(name, pat):
                return True
    return False


def selected_key(name: str, tensor: torch.Tensor, include: List[str], exclude: List[str]) -> bool:
    if tensor.ndim != 2:
        return False
    if include and not match_any(include, name):
        return False
    if exclude and match_any(exclude, name):
        return False
    return True


def get_layer_idx(name: str) -> int:
    m = LAYER_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse layer index from parameter name: {name}")
    return int(m.group(1))


def get_module_name(name: str) -> str:
    m = MODULE_RE.search(name)
    return m.group(1) if m else "unknown"


def get_module_family(name: str) -> str:
    module = get_module_name(name)
    if module.startswith("mlp."):
        return "mlp"
    if module.startswith("self_attn."):
        return "attention"
    return "unknown"


def safe_name(name: str) -> str:
    return name.replace("/", "__slash__").replace(".", "__dot__")


def load_model_state(model_id: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    sd = model.state_dict()
    # Clone tensors so they remain valid after model deletion in some loading modes.
    sd = {k: v.detach().cpu() for k, v in sd.items()}
    del model
    gc.collect()
    return sd


@torch.no_grad()
def frob_abt_sq_chunked(A: torch.Tensor, B: torch.Tensor, device: torch.device, chunk_rows: int) -> float:
    """Return ||A B^T||_F^2 without materializing the full product."""
    assert A.ndim == 2 and B.ndim == 2 and A.shape == B.shape
    B_dev = B.to(device=device, dtype=torch.float32)
    total = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, A.shape[0], chunk_rows):
        A_chunk = A[start : start + chunk_rows].to(device=device, dtype=torch.float32)
        prod = A_chunk @ B_dev.T
        total += prod.double().pow(2).sum()
        del A_chunk, prod
    del B_dev
    return float(total.cpu().item())


@torch.no_grad()
def compute_s_key(
    task_vectors: List[torch.Tensor],
    device: torch.device,
    chunk_rows: int,
    tau_m: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Compute S_key = sum_i ||(tau_m - tau_i) tau_i^T||_F^2 / ||tau_i||_F^2.
    If tau_m is not provided, use mean_j tau_j.
    Returns (raw_score, score_per_param).
    """
    tvs = [t.float().cpu() for t in task_vectors]
    tau_m = tau_m.float().cpu() if tau_m is not None else torch.stack(tvs, dim=0).mean(dim=0)

    score = 0.0
    for tau_i in tvs:
        delta = tau_m - tau_i
        num = frob_abt_sq_chunked(delta, tau_i, device=device, chunk_rows=chunk_rows)
        denom = float(tau_i.float().pow(2).sum().item()) + eps
        score += num / denom

    score_per_param = score / float(tau_m.numel())
    return score, score_per_param


def cache_task_vectors(args: argparse.Namespace, base_sd: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, List[str]]:
    """Sequentially load finetunes and save selected deltas to disk. Returns key -> list[path]."""
    cache_dir = args.cache_dir or os.path.join(args.outdir, "delta_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key_to_paths: Dict[str, List[str]] = {k: [] for k in keys}
    dtype = dtype_from_str(args.dtype)

    for task_id, ft in enumerate(args.finetunes):
        print(f"Loading finetune {task_id}: {ft}")
        ft_sd = load_model_state(ft, dtype=dtype)
        task_dir = os.path.join(cache_dir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        for k in tqdm(keys, desc=f"Caching deltas for task {task_id}"):
            if k not in ft_sd:
                raise KeyError(f"Key {k} missing in finetune {ft}")
            if ft_sd[k].shape != base_sd[k].shape:
                raise ValueError(f"Shape mismatch for {k}: base={base_sd[k].shape}, ft={ft_sd[k].shape}")
            delta = (ft_sd[k].float() - base_sd[k].float()).to(torch.float16)
            path = os.path.join(task_dir, safe_name(k) + ".pt")
            torch.save(delta, path)
            key_to_paths[k].append(path)
            del delta

        del ft_sd
        gc.collect()

    return key_to_paths


def cache_merged_task_vectors(
    args: argparse.Namespace,
    merged_sd: Dict[str, torch.Tensor],
    base_sd: Dict[str, torch.Tensor],
    keys: List[str],
) -> Dict[str, str]:
    """Save selected merged-model task vectors tau_m = merged - base. Returns key -> path."""
    cache_dir = args.cache_dir or os.path.join(args.outdir, "delta_cache")
    merged_dir = os.path.join(cache_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    key_to_path = {}
    for k in tqdm(keys, desc="Building merged task vectors"):
        if k not in merged_sd:
            raise KeyError(f"Key {k} missing in merged checkpoint")
        if merged_sd[k].shape != base_sd[k].shape:
            raise ValueError(f"Shape mismatch for {k}: base={base_sd[k].shape}, merged={merged_sd[k].shape}")
        merged_tv = (merged_sd[k].float() - base_sd[k].float()).to(torch.float16)
        path = os.path.join(merged_dir, safe_name(k) + ".pt")
        torch.save(merged_tv, path)
        key_to_path[k] = path
        del merged_tv
    return key_to_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = dtype_from_str(args.dtype)

    if len(args.task_names) != len(args.finetunes):
        args.task_names = [f"task_{i}" for i in range(len(args.finetunes))]

    print(f"Loading base: {args.base}")
    base_sd = load_model_state(args.base, dtype=dtype)
    keys = [k for k, v in base_sd.items() if selected_key(k, v, args.include, args.exclude)]
    keys = sorted(keys, key=lambda x: (get_layer_idx(x), x))
    print(f"Selected {len(keys)} 2D projection matrices.")

    metadata = {
        "base": args.base,
        "finetunes": args.finetunes,
        "merged": args.merged,
        "tau_m_source": "merged_checkpoint" if args.merged else "mean_finetunes",
        "task_names": args.task_names,
        "include": args.include,
        "exclude": args.exclude,
        "layer_reduce": args.layer_reduce,
        "chunk_rows": args.chunk_rows,
    }
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if args.merged:
        print(f"Loading merged checkpoint for tau_m: {args.merged}")
        merged_sd = load_model_state(args.merged, dtype=dtype)
        key_to_merged_path = cache_merged_task_vectors(args, merged_sd, base_sd, keys)
        del merged_sd
        gc.collect()
    else:
        key_to_merged_path = None

    if args.load_all_finetunes:
        print("Loading all finetunes into RAM. This is faster but memory-heavy.")
        ft_sds = [load_model_state(ft, dtype=dtype) for ft in args.finetunes]
        key_to_tvs = {}
        for k in tqdm(keys, desc="Building in-memory task vectors"):
            key_to_tvs[k] = [(sd[k].float() - base_sd[k].float()).to(torch.float16) for sd in ft_sds]
        del ft_sds
        gc.collect()
        key_to_paths = None
    else:
        key_to_tvs = None
        key_to_paths = cache_task_vectors(args, base_sd, keys)

    rows = []
    for k in tqdm(keys, desc="Computing S_key"):
        if key_to_tvs is not None:
            tvs = key_to_tvs[k]
        else:
            assert key_to_paths is not None
            tvs = [torch.load(p, map_location="cpu") for p in key_to_paths[k]]

        tau_m = torch.load(key_to_merged_path[k], map_location="cpu") if key_to_merged_path is not None else None
        raw, per_param = compute_s_key(tvs, device=device, chunk_rows=args.chunk_rows, tau_m=tau_m)
        module_family = get_module_family(k)
        if module_family == "unknown":
            warnings.warn(
                f"Selected parameter has unknown module family; expected Llama attention or MLP projection: {k}",
                RuntimeWarning,
            )

        rows.append(
            {
                "layer": get_layer_idx(k),
                "module": get_module_name(k),
                "module_family": module_family,
                "param_name": k,
                "shape": str(tuple(tvs[0].shape)),
                "num_params": int(tvs[0].numel()),
                "S_key": raw,
                "S_key_per_param": per_param,
        }
        )
        del tau_m
        del tvs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df_key = pd.DataFrame(rows)
    key_csv = os.path.join(args.outdir, "s_key_by_module.csv")
    df_key.to_csv(key_csv, index=False)

    agg_func = "sum" if args.layer_reduce == "sum" else "mean"
    df_layer = df_key.groupby("layer", as_index=False).agg(
        S_layer=("S_key", agg_func),
        S_layer_per_param=("S_key_per_param", agg_func),
        total_params=("num_params", "sum"),
        n_modules=("param_name", "count"),
    )

    df_family = df_key.groupby(["layer", "module_family"], as_index=False).agg(
        S_layer=("S_key", agg_func),
        S_layer_per_param=("S_key_per_param", agg_func),
        total_params=("num_params", "sum"),
        n_modules=("param_name", "count"),
    )
    df_family_wide = df_family.pivot(index="layer", columns="module_family")
    df_family_wide.columns = [f"{metric}_{family}" for metric, family in df_family_wide.columns]
    df_family_wide = df_family_wide.reset_index()
    df_layer = df_layer.merge(df_family_wide, on="layer", how="left")
    for family in ("mlp", "attention"):
        for metric in ("S_layer", "S_layer_per_param", "total_params", "n_modules"):
            col = f"{metric}_{family}"
            if col not in df_layer:
                df_layer[col] = 0.0
            else:
                df_layer[col] = df_layer[col].fillna(0.0)

    # Additional normalized variant: total S divided by total selected params in layer.
    df_sum = df_key.groupby("layer", as_index=False).agg(S_layer_sum=("S_key", "sum"), total_params=("num_params", "sum"))
    df_sum["S_layer_sum_per_param"] = df_sum["S_layer_sum"] / df_sum["total_params"]
    df_layer = df_layer.merge(df_sum[["layer", "S_layer_sum_per_param"]], on="layer", how="left")
    df_family_sum = df_key.groupby(["layer", "module_family"], as_index=False).agg(
        S_layer_sum=("S_key", "sum"),
        total_params=("num_params", "sum"),
    )
    df_family_sum["S_layer_sum_per_param"] = df_family_sum["S_layer_sum"] / df_family_sum["total_params"]
    df_family_sum_wide = df_family_sum.pivot(index="layer", columns="module_family", values="S_layer_sum_per_param")
    df_family_sum_wide = df_family_sum_wide.rename(
        columns={family: f"S_layer_sum_per_param_{family}" for family in df_family_sum_wide.columns}
    ).reset_index()
    df_layer = df_layer.merge(df_family_sum_wide, on="layer", how="left")
    for family in ("mlp", "attention"):
        col = f"S_layer_sum_per_param_{family}"
        if col not in df_layer:
            df_layer[col] = 0.0
        else:
            df_layer[col] = df_layer[col].fillna(0.0)

    layer_csv = os.path.join(args.outdir, "s_layer.csv")
    df_layer.to_csv(layer_csv, index=False)

    # Plot raw layer score.
    plt.figure(figsize=(10, 5))
    plt.plot(df_layer["layer"], df_layer["S_layer"], marker="o", label="total")
    plt.plot(df_layer["layer"], df_layer["S_layer_mlp"], marker="o", label="MLP only")
    plt.plot(df_layer["layer"], df_layer["S_layer_attention"], marker="o", label="attention only")
    plt.xlabel("Layer index")
    plt.ylabel(f"S_l ({args.layer_reduce} over selected modules)")
    plt.title("Layer-wise pre-WUDI objective score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    raw_png = os.path.join(args.outdir, "s_layer.png")
    plt.savefig(raw_png, dpi=200)
    plt.close()

    # Plot parameter-normalized layer score.
    plt.figure(figsize=(10, 5))
    if args.normalize_by_param_count:
        y_total = df_layer["S_layer_sum_per_param"]
        y_mlp = df_layer["S_layer_sum_per_param_mlp"]
        y_attention = df_layer["S_layer_sum_per_param_attention"]
    else:
        y_total = df_layer["S_layer_per_param"]
        y_mlp = df_layer["S_layer_per_param_mlp"]
        y_attention = df_layer["S_layer_per_param_attention"]
    plt.plot(df_layer["layer"], y_total, marker="o", label="total")
    plt.plot(df_layer["layer"], y_mlp, marker="o", label="MLP only")
    plt.plot(df_layer["layer"], y_attention, marker="o", label="attention only")
    plt.xlabel("Layer index")
    ylabel = "sum(S_key) / selected parameter count" if args.normalize_by_param_count else f"mean/sum S_key_per_param ({args.layer_reduce})"
    plt.ylabel(ylabel)
    plt.title("Layer-wise pre-WUDI score, parameter-normalized")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    norm_png = os.path.join(args.outdir, "s_layer_per_param.png")
    plt.savefig(norm_png, dpi=200)
    plt.close()

    print("Done.")
    print(f"Saved module scores: {key_csv}")
    print(f"Saved layer scores:  {layer_csv}")
    print(f"Saved plots:         {raw_png}")
    print(f"                     {norm_png}")


if __name__ == "__main__":
    main()
