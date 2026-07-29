import argparse
import json
from typing import Any, List
import torch
from transformers import AutoTokenizer
import utils
from param import param
from merge import MergingMethod
import time
from pathlib import Path
import os
import re
import gc

BASE = "google/gemma-2-2b"

FT_MODELS = [
    ("instruction", "MergeBench/gemma-2-2b_instruction"),
    ("math",        "MergeBench/gemma-2-2b_math"),
    ("coding",      "MergeBench/gemma-2-2b_coding"),
]

DEFAULT_EXCLUDE = [
    r".*embed_tokens\.weight$",
    r".*lm_head\.weight$",
]

def write_merge_readme(save_path: str, args: Any, ft_ckpts: List[str], runtime_sec: float):
    """
    Create README.md summarizing the merged model and merge settings.

    Args:
        save_path: Directory where the merged model is saved.
        args: argparse Namespace (or any obj with __dict__/vars()) of CLI args.
        kwargs: Extra keyword args passed to the merge function.
        ft_ckpts: List of fine-tuned checkpoint names/paths merged.
        runtime_sec: Merge runtime in seconds.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    readme_path = os.path.join(save_path, "README.md")

    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    content = [
        "# Merged Model",
        f"- Base model: {BASE}",
        f"- Algorithm: `{args_dict.get('merge_method', args_dict.get('algo', 'N/A'))}`",
        f"- Save path: `{save_path}`",
        f"- Fine-tuned checkpoints: {ft_ckpts}",
        f"- Merge runtime (s): {runtime_sec:.3f}",
        "",
        "## Arguments",
        "```json",
        json.dumps(args_dict, indent=2),
        "```",
        "",
    ]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

def parse_args():
    ap = argparse.ArgumentParser("Merge Llama-3.2-3B MergeBench finetunes (merge-only)")
    ap.add_argument("--out", required=True, help="Output dir for merged model")

    ap.add_argument("--scaling", type=float, default=1.0,
                    help="Scaling applied to merged task vector (TA typical: 1/num_tasks; for WUDI also supported)")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device_map", default="cpu", help="HF loading device_map (cpu is safest)")

    ap.add_argument("--merge_method", default="task_arithmetic",
                    help="Implemented in merge.MergingMethod (e.g. task_arithmetic, wudi_merge)")

    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Additional regex patterns to exclude from merging; default excludes embeddings and lm_head")

    # ---- WUDI specific knobs (used by WUDI merge methods) ----
    ap.add_argument("--wudi_variant",
                    choices=[
                        "wudi_all_linear",
                        "wudi_lines_all_linear",
                        "wudi_attention_only",
                        "wudi_mlp_only",
                        "wudi_last_7_layers",
                        "wudi_last_14_layers",
                        "wudi_last_21_layers",
                    ],
                    default="wudi_all_linear",
                    help="Which parameter subset to optimize with WUDI")
    ap.add_argument("--wudi_iter", type=int, default=200)
    ap.add_argument("--wudi_lr", type=float, default=1e-5)
    ap.add_argument("--wudi_weight_decay", type=float, default=0.0)
    ap.add_argument("--wudi_alpha", "--alpha", dest="wudi_alpha", type=float, default=0.0,
                    help="Strength of the WUDI proximal penalty around the cold initialization")
    ap.add_argument("--wudi_device", default="cuda", help="cuda recommended; cpu will be very slow")
    ap.add_argument("--wudi_warm_start", action="store_true",
                    help="Enable closed-form warm start for WUDI optimization")
    ap.add_argument("--wudi_cfs_ridge", type=float, default=0.0,
                    help="Ridge regularization used by WUDI closed-form warm start")
    ap.add_argument("--wudi_fallback", choices=["task_arithmetic", "zero"], default="task_arithmetic",
                    help="How to merge keys not optimized by WUDI")
    ap.add_argument("--wudi_fallback_scaling", type=float, default=1.0,
                    help="Scaling applied only to task_arithmetic fallback vectors")
    ap.add_argument("--wudi_K", type=float, default=0.7,
                    help="Keep fraction for sparsed_wudi_merge; must satisfy 0 < K <= 1")
    ap.add_argument("--wudi_sparsify_variant",
                    choices=["ties_sparsify", "dare_sparsify"],
                    default="ties_sparsify",
                    help="Sparsification variant for sparsed_wudi_merge")
    ap.add_argument("--wudi_loss_log_csv", default=None,
                    help="CSV path for WUDI metric logs. Default: <out>/wudi_loss_by_key.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    args.effective_exclude = DEFAULT_EXCLUDE + args.exclude
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # 1) Load base + finetunes (CausalLM)
    base_model = utils.load_causallm(BASE, dtype=dtype, device_map=args.device_map)
    ft_models = []
    ft_names = []
    for name, mid in FT_MODELS:
        ft_names.append(name)
        ft_models.append(utils.load_causallm(mid, dtype=dtype, device_map=args.device_map))

    # 2) Wrap into param objects
    base_p = param(base_model)
    ft_ps = [param(m) for m in ft_models]
    del ft_models
    gc.collect()

    # 3) Filtering: excluded keys are removed from merging and keep base values.
    if args.effective_exclude:
        def keep_param(n: str, _t: torch.Tensor):
            return not any(re.match(pat, n) for pat in args.effective_exclude)

        base_p.filter(keep_param)
        for p in ft_ps:
            p.filter(keep_param)

    # 4) Merge
    merger = MergingMethod(models_to_merge=ft_ps, models_name=ft_names)
    if not hasattr(merger, args.merge_method):
        raise ValueError(f"Unknown merge_method={args.merge_method}. "
                         f"Pick one implemented in merge.MergingMethod")

    merge_fn = getattr(merger, args.merge_method)

    start = time.perf_counter()
    
    if args.merge_method in {"wudi_merge", "sparsed_wudi_merge", "selective_wudi_merge"}:
        wudi_loss_log_csv = args.wudi_loss_log_csv or os.path.join(args.out, "wudi_loss_by_key.csv")
        merge_kwargs = {
            "base_model": base_p,
            "models_to_merge": ft_ps,
            "scaling": args.scaling,
            "iter_num": args.wudi_iter,
            "lr": args.wudi_lr,
            "weight_decay": args.wudi_weight_decay,
            "alpha": args.wudi_alpha,
            "device": args.wudi_device,
            "warm_start": args.wudi_warm_start,
            "cfs_ridge": args.wudi_cfs_ridge,
            "fallback": args.wudi_fallback,
            "fallback_scaling": args.wudi_fallback_scaling,
            "eps": 1e-12,
            "loss_log_path": wudi_loss_log_csv,
            "verbose": True,
        }
        if args.merge_method == "sparsed_wudi_merge":
            merge_kwargs["K"] = args.wudi_K
            merge_kwargs["sparsify_variant"] = args.wudi_sparsify_variant
        if args.merge_method == "selective_wudi_merge":
            merge_kwargs["variant"] = args.wudi_variant

        merged_p = merge_fn(**merge_kwargs)
    else:
        # default path (e.g., task_arithmetic)
        merged_p = merge_fn(base_model=base_p, models_to_merge=ft_ps, scaling=args.scaling)

    runtime_sec = time.perf_counter() - start
    # 5) Write merged weights into a real HF model + save
    merged_p.assign(base_model)
    base_model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.out)
    
    write_merge_readme(
        save_path=args.out,
        args=args,
        ft_ckpts=[mid for _, mid in FT_MODELS],
        runtime_sec=runtime_sec,
    )

    print("Saved merged model to:", args.out)


if __name__ == "__main__":
    main()
