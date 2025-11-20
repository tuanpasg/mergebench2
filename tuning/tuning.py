import os, json, time, itertools, tempfile, shutil, math, random, importlib, subprocess, sys, re
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config
# -------------------------
RANDOM_SEED = 2025
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparam search grids (unchanged; extend if desired)
GRIDS = {
    "TaskArithmetic": dict(scaling_coef=[0.2, 0.4, 0.6, 0.8, 1.0]),
    "TIES":           dict(K=[0.1, 0.2, 0.3], scaling_coef=[0.4, 0.6, 0.8], merge_func=["sum"]),
    "DARE":           dict(p=[0.1, 0.2, 0.3],  scaling_coef=[0.4, 0.6, 0.8]),
    "Consensus":      dict(k=[2], lamda=[0.3, 0.5, 0.7], scaling_coef=[0.4, 0.6, 0.8]),
    "LocalizeAndStitch": dict(sparsity=[0.1, 0.2, 0.3, 0.4]),
}

# -------------------------
# Small utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

def _dtype_flag():
    if torch.cuda.is_available():
        return "dtype=bfloat16"
    return ""

def _run(cmd: list, cwd: Optional[str]=None, env: Optional[dict]=None) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env or os.environ)
    out, err = p.communicate()
    return p.returncode, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore")

def _parse_json_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {"raw": data}

def _pick_first_metric(d: dict, preferred_keys: list) -> Optional[float]:
    for k in preferred_keys:
        parts = k.split(".")
        node = d
        ok = True
        for p in parts:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                ok = False
                break
        if ok and isinstance(node, (int, float)):
            return float(node)

    def scan(x):
        out = []
        if isinstance(x, dict):
            for kk, vv in x.items():
                if isinstance(vv, (int, float)) and re.search(r"(acc|accuracy|pass|score|em|exact)", kk, re.I):
                    out.append(float(vv))
                else:
                    out.extend(scan(vv))
        elif isinstance(x, list):
            for vv in x:
                out.extend(scan(vv))
        return out

    cands = scan(d)
    if cands:
        return sum(cands) / len(cands)
    return None

# -------------------------
# Harness-backed evaluators
# -------------------------
def eval_ifeval_lmeval(model_path: str, limit: int = 100) -> Tuple[Optional[float], str]:
    outdir = tempfile.mkdtemp(prefix="lmeval_ifeval_")
    outfile = os.path.join(outdir, "results.json")

    base_cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},{_dtype_flag()},trust_remote_code=True",
        "--tasks", "ifeval",
        "--num_fewshot", "0",
        "--limit", str(limit),
        "--batch_size", "auto",
        "--output_path", outfile
    ]
    code, out, err = _run(base_cmd)
    if code != 0:
        fb_cmd = base_cmd[:]
        try_idx = fb_cmd.index("--tasks") + 1
        fb_cmd[try_idx] = "IFEval"
        code2, out2, err2 = _run(fb_cmd)
        if code2 != 0:
            shutil.rmtree(outdir, ignore_errors=True)
            reason = "lm-evaluation-harness not found or IFEval task missing.\n" + (err or err2)
            return None, reason

    try:
        metrics = _parse_json_metrics(outfile)
        score = _pick_first_metric(metrics, [
            "results.ifeval.acc",
            "results.IFEval.acc",
            "results.ifeval.strict_acc",
            "results.ifeval.pass_rate",
            "results.IFEval.pass_rate",
        ])
        if score is None:
            return None, "IFEval ran but no recognizable accuracy metric was found."
        return float(score), "ok"
    finally:
        shutil.rmtree(outdir, ignore_errors=True)

### NEW: Hendrycks MATH (hendrycks_math) via lm-eval
def eval_hendrycks_math_lmeval(model_path: str, limit: Optional[int] = None) -> Tuple[Optional[float], str]:
    """
    Runs hendrycks_math (MATH) from lm-evaluation-harness.
    limit: optionally cap the number of problems; if None, use full task.
    """
    outdir = tempfile.mkdtemp(prefix="lmeval_hmath_")
    outfile = os.path.join(outdir, "results.json")

    base_cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},{_dtype_flag()},trust_remote_code=True",
        "--tasks", "hendrycks_math",
        "--num_fewshot", "0",
        "--batch_size", "auto",
        "--output_path", outfile
    ]
    if limit is not None:
        base_cmd += ["--limit", str(limit)]

    code, out, err = _run(base_cmd)
    if code != 0:
        shutil.rmtree(outdir, ignore_errors=True)
        return None, "lm-evaluation-harness not found or hendrycks_math task failed.\n" + (err or out)

    try:
        metrics = _parse_json_metrics(outfile)
        # Prefer exact match / accuracy-like metrics typically reported by the task
        score = _pick_first_metric(metrics, [
            "results.hendrycks_math.acc",
            "results.hendrycks_math.exact_match",
            "results.acc",
            "results.exact_match",
        ])
        if score is None:
            return None, "hendrycks_math ran but no recognizable accuracy/EM metric was found."
        return float(score), "ok"
    finally:
        shutil.rmtree(outdir, ignore_errors=True)

# -------------------------
# Main validation loop
# -------------------------
def main():
    from prepare_args import create_parser
    from main import get_ft_ckpts  # your helper to gather FT checkpoints

    parser = create_parser()
    parser.add_argument('--algo', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--save-path', default='./merged_models/validated/')
    parser.add_argument('--per_task_samples', type=int, default=150)  # kept for CLI compatibility
    parser.add_argument('--ifeval_limit', type=int, default=100)
    ### CHANGED: replace MMLU subject controls with a simple optional limit for hendrycks_math
    parser.add_argument('--math_limit', type=int, default=None,
                        help="If set, caps number of hendrycks_math items during tuning (None = full).")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    os.makedirs(args.save_path, exist_ok=True)

    # Quick tokenizer init (not strictly necessary for harness path)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Prepare merger
    merger_module = importlib.import_module("merging_methods")
    ft_ckpts = get_ft_ckpts(args.base_model)

    grid = GRIDS.get(args.algo)
    if not grid:
        raise ValueError(f"No tuning grid configured for algo '{args.algo}'")

    keys, vals = zip(*grid.items())
    best = dict(score=-1.0, params=None, model_dir=None, detail=None)

    for combo in itertools.product(*vals):
        trial = dict(zip(keys, combo))
        outdir = tempfile.mkdtemp(prefix=f"{args.algo}_", dir=args.save_path)

        merger = getattr(merger_module, args.algo)(args.base_model, ft_ckpts, outdir)

        t0 = time.perf_counter()
        merger.merge(**trial)
        merge_time = time.perf_counter() - t0

        base_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(outdir, torch_dtype=base_dtype, trust_remote_code=True).to(DEFAULT_DEVICE)
        model.eval()
        del model  # harness CLIs will read the saved folder

        scores = []
        detail = {}

        # === Instruction following: IFEval (lm-eval), limit first 100 samples ===
        ifeval_acc, ifeval_note = eval_ifeval_lmeval(outdir, limit=args.ifeval_limit)
        if ifeval_acc is not None and not math.isnan(ifeval_acc):
            scores.append(ifeval_acc); detail["instruction_ifeval_acc"] = ifeval_acc
        else:
            detail["instruction_ifeval_acc"] = f"skipped: {ifeval_note}"

        # === Math: Hendrycks MATH (lm-eval) ===
        hmath_acc, hmath_note = eval_hendrycks_math_lmeval(outdir, limit=args.math_limit)
        if hmath_acc is not None and not math.isnan(hmath_acc):
            scores.append(hmath_acc); detail["hendrycks_math_acc"] = hmath_acc
        else:
            detail["hendrycks_math_acc"] = f"skipped: {hmath_note}"

        # === Coding: MBPP+ (bigcode-evaluation-harness) ===
        mbpp_acc, mbpp_note = eval_mbppplus_bigcode(outdir)
        if mbpp_acc is not None and not math.isnan(mbpp_acc):
            scores.append(mbpp_acc); detail["mbppplus_acc"] = mbpp_acc
        else:
            detail["mbppplus_acc"] = f"skipped: {mbpp_note}"

        numeric_scores = [s for s in scores if isinstance(s, (int, float))]
        if not numeric_scores:
            print(f"[WARN] Trial {trial} produced no numeric scores; skipping.")
            shutil.rmtree(outdir, ignore_errors=True)
            continue

        avg_acc = sum(numeric_scores) / len(numeric_scores)
        detail.update(trial=trial, avg_acc=avg_acc, merge_time_sec=merge_time)

        print(f"[VAL] {trial} => avg_acc={avg_acc:.3f} | detail={detail}")

        if avg_acc > best["score"]:
            if best["model_dir"] and os.path.exists(best["model_dir"]):
                shutil.rmtree(best["model_dir"], ignore_errors=True)
            best.update(score=avg_acc, params=trial, model_dir=outdir, detail=detail)
        else:
            shutil.rmtree(outdir, ignore_errors=True)

    meta = dict(
        algo=args.algo,
        base_model=args.base_model,
        best_score=best["score"],
        best_params=best["params"],
        best_model_dir=best["model_dir"],
        metrics=best["detail"],
        note=(
            "Evaluated with harnesses: IFEval (limit first N), hendrycks_math (MATH), MBPP+.\n"
            "If a harness/task was missing, it was skipped and excluded from the average."
        )
    )
    with open(os.path.join(args.save_path, f"best_{args.algo}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== BEST CONFIG ===")
    print(json.dumps(meta, indent=2))

# -------------------------
# Coding (unchanged helper)
# -------------------------
def eval_mbppplus_bigcode(model_path: str) -> Tuple[Optional[float], str]:
    outdir = tempfile.mkdtemp(prefix="bigcode_mbppplus_")
    outfile = os.path.join(outdir, "results.json")

    variants = [
        [sys.executable, "-m", "bigcode_eval", "evaluate",
         "--model", "hf", "--model_args", f"pretrained={model_path},trust_remote_code=True,{_dtype_flag()}",
         "--tasks", "mbppplus", "--max_samples", "0", "--output_path", outfile],
        [sys.executable, "-m", "bigcode_eval",
         "--model", "hf", "--model_args", f"pretrained={model_path},trust_remote_code=True,{_dtype_flag()}",
         "--tasks", "mbppplus", "--max_samples", "0", "--output_path", outfile],
        ["bigcode-eval",
         "--model", "hf", "--model_args", f"pretrained={model_path},trust_remote_code=True,{_dtype_flag()}",
         "--tasks", "mbppplus", "--max_samples", "0", "--output_path", outfile],
    ]

    last_err = ""
    for cmd in variants:
        code, out, err = _run(cmd)
        if code == 0 and os.path.exists(outfile):
            try:
                metrics = _parse_json_metrics(outfile)
                score = _pick_first_metric(metrics, [
                    "results.mbppplus.pass_at_1",
                    "results.mbppplus.strict_accuracy",
                    "results.pass@1",
                    "results.accuracy",
                ])
                if score is None:
                    shutil.rmtree(outdir, ignore_errors=True)
                    return None, "MBPP+ ran but no recognizable accuracy metric was found."
                shutil.rmtree(outdir, ignore_errors=True)
                return float(score), "ok"
            except Exception as e:
                last_err = f"parse error: {e}"
        else:
            last_err = err or out

    shutil.rmtree(outdir, ignore_errors=True)
    return None, f"bigcode-evaluation-harness not found or mbppplus task failed.\n{last_err}"

if __name__ == "__main__":
    main()
