import os, json, time, itertools, tempfile, shutil, math, random, importlib
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# -------------------------
# Config
# -------------------------
RANDOM_SEED = 2025
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparam search grids (trimmed; extend if you want wider search)
GRIDS = {
    "TaskArithmetic": dict(scaling_coef=[0.2, 0.4, 0.6, 0.8, 1.0]),
    "TIES":           dict(K=[0.1, 0.2, 0.3], scaling_coef=[0.4, 0.6, 0.8], merge_func=["sum"]),
    "DARE":           dict(p=[0.1, 0.2, 0.3],  scaling_coef=[0.4, 0.6, 0.8]),
    "Consensus":      dict(k=[2], lamda=[0.3, 0.5, 0.7], scaling_coef=[0.4, 0.6, 0.8]),
    "LocalizeAndStitch": dict(sparsity=[0.1, 0.2, 0.3, 0.4]),
}

# Subject list biased toward STEM for math (you can tweak)
MMLU_STEM_SUBJECTS = [
    "high_school_mathematics","college_mathematics","abstract_algebra",
    "high_school_physics","physics","high_school_chemistry","chemistry",
    "computer_science","electrical_engineering","machine_learning"
]

# -------------------------
# Small utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normspace(s: str) -> str:
    return " ".join((s or "").strip().split())

def whitespace_normalize(s: str) -> str:
    return "".join((s or "").split())

def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

# -------------------------
# Surrogate datasets
# -------------------------
def load_ifeval_prompts(n: int) -> List[Dict]:
    """
    Try TIGER-Lab/IFEval or lukaemon/IFEval. We only need the 'prompt' text.
    If the official evaluator is available, we’ll use it later to score.
    """
    tried = [("TIGER-Lab/IFEval","validation"), ("lukaemon/IFEval","validation")]
    for repo, split in tried:
        try:
            ds = load_dataset(repo, split=split)
            idx = list(range(min(len(ds), n)))
            random.shuffle(idx)
            ds = ds.select(idx)
            # Keep prompt and any metadata if present
            return [{"prompt": ex.get("prompt", ex.get("instruction",""))} for ex in ds]
        except Exception:
            continue
    return []

def load_mmlu_stem(n_per_subject: int) -> List[Dict]:
    exs = []
    for subj in MMLU_STEM_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            # Random slice
            idx = list(range(min(len(ds), n_per_subject)))
            random.shuffle(idx)
            ds = ds.select(idx)
            for ex in ds:
                exs.append({
                    "subject": subj,
                    "question": ex.get("question",""),
                    "A": ex.get("A",""), "B": ex.get("B",""),
                    "C": ex.get("C",""), "D": ex.get("D",""),
                    "answer": ex.get("answer","")
                })
        except Exception:
            continue
    return exs

def load_conala(n: int) -> List[Dict]:
    # Try validation first, then train (both have intent/snippet)
    tried = [("neulab/conala","validation"), ("neulab/conala","train")]
    for repo, split in tried:
        try:
            ds = load_dataset(repo, split=split)
            idx = list(range(min(len(ds), n)))
            random.shuffle(idx)
            ds = ds.select(idx)
            out = []
            for ex in ds:
                intent = ex.get("intent", "")
                snippet = ex.get("snippet", "")
                if intent and snippet:
                    out.append({"intent": intent, "snippet": snippet})
            return out
        except Exception:
            continue
    return []

# -------------------------
# Evaluators
# -------------------------
def generate(model, tok, prompts: List[str], max_new_tokens=64, temperature=0.0) -> List[str]:
    outs = []
    for batch in batchify(prompts, bs=8):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                num_beams=1,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        # Strip the prompt (heuristic) to get only the completion
        for p, full in zip(batch, texts):
            outs.append(full[len(p):].strip() if full.startswith(p) else full.strip())
    return outs

def mmlu_accuracy_mc(model, tok, items: List[Dict]) -> float:
    """
    Score by option log-likelihood: append each option, compute mean token logprob,
    pick argmax, compare to gold 'answer' (A/B/C/D).
    """
    correct = 0
    total = 0
    for ex in items:
        q = ex["question"]
        options = [("A", ex["A"]), ("B", ex["B"]), ("C", ex["C"]), ("D", ex["D"])]
        # Build prompts like: "Q\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer:"
        stem = f"{q}\nA) {ex['A']}\nB) {ex['B']}\nC) {ex['C']}\nD) {ex['D']}\nAnswer:"
        with torch.no_grad():
            enc_stem = tok(stem, return_tensors="pt").to(model.device)
        # For each option letter, score the letter token
        scores = []
        for letter, _ in options:
            with torch.no_grad():
                enc = tok(letter, return_tensors="pt").to(model.device)
                # Condition on stem, score letter
                out = model(**{**enc_stem, "labels": torch.cat([enc_stem["input_ids"], enc["input_ids"]], dim=1)})
                # The loss is averaged over all tokens (stem+letter); we want only letter.
                # Approximation: compute next-token logprob for the first letter token:
            # Better approach: compute token logprobs autoregressively
            # Simpler: compute logprob of letter given stem
            with torch.no_grad():
                full = tok(stem + " " + letter, return_tensors="pt").to(model.device)
                out = model(**full, labels=full["input_ids"])
                # Loss is per-token; to isolate letter contribution, we compare two losses:
                # However, to keep this simple and stable, we score the whole " stem + letter"
                # and treat the smaller loss as better (consistent across options).
                scores.append(-out.loss.item())
        pred_letter = ["A","B","C","D"][int(torch.tensor(scores).argmax())]
        correct += int(pred_letter == ex["answer"])
        total += 1
    return correct / max(1, total)

def conala_exact_match(model, tok, items: List[Dict]) -> float:
    prompts = [f"# Task:\n{it['intent']}\n# Python code:" for it in items]
    gens = generate(model, tok, prompts, max_new_tokens=64, temperature=0.0)
    correct = 0
    for it, g in zip(items, gens):
        ref = whitespace_normalize(it["snippet"])
        hyp = whitespace_normalize(g.split("\n")[0])  # take first line
        correct += int(hyp == ref)
    return correct / max(1, len(items))

def ifeval_pass_rate(model, tok, items: List[Dict]) -> Tuple[float, str]:
    """
    Try to evaluate IFEval prompts. If the official evaluator is available, use it.
    Otherwise, return (nan, 'skipped') and the caller will drop this task from averaging.
    """
    # Try to import an evaluator if user installed it (placeholder names)
    for mod in ["ifeval", "tiger_ifeval", "TIGER_IFEVAL"]:
        try:
            evaluator = importlib.import_module(mod)
            break
        except Exception:
            evaluator = None
    if evaluator is None:
        return float("nan"), "skipped (no IFEval evaluator installed)"

    prompts = [it["prompt"] for it in items]
    outputs = generate(model, tok, prompts, max_new_tokens=64, temperature=0.0)

    try:
        # Expect evaluator to expose a function like: evaluate(prompts, outputs) -> list[bool]
        results = evaluator.evaluate(prompts, outputs)  # this is a placeholder; adapt to your evaluator
        acc = sum(1 for r in results if r) / max(1, len(results))
        return acc, "ok"
    except Exception as e:
        return float("nan"), f"skipped (evaluator error: {e})"

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
    parser.add_argument('--per_task_samples', type=int, default=150)
    parser.add_argument('--mmlu_per_subject', type=int, default=30)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    os.makedirs(args.save_path, exist_ok=True)

    # Load base tokenizer (and device dtype for candidates)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_dtype = torch.bfloat16 if torch.cuda.is_available() else None

    # Load surrogate datasets
    instr_items = load_ifeval_prompts(args.per_task_samples)    # may be empty if HF id unavailable
    code_items  = load_conala(args.per_task_samples)
    math_items  = load_mmlu_stem(args.mmlu_per_subject)

    available = {
        "instruction": len(instr_items) > 0,
        "coding": len(code_items) > 0,
        "math": len(math_items) > 0,
    }
    if not any(available.values()):
        raise RuntimeError("No surrogate validation data could be loaded. Check internet / datasets.")

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

        model = AutoModelForCausalLM.from_pretrained(outdir, torch_dtype=base_dtype).to(DEFAULT_DEVICE)
        model.eval()

        scores = []
        detail = {}

        # Instruction (IFEval)
        if available["instruction"]:
            acc, note = ifeval_pass_rate(model, tok, instr_items)
            if not math.isnan(acc):
                scores.append(acc); detail["instruction_acc"] = acc
            else:
                detail["instruction_acc"] = note

        # Coding (CoNaLa)
        if available["coding"]:
            acc = conala_exact_match(model, tok, code_items)
            scores.append(acc); detail["coding_acc"] = acc

        # Math (MMLU STEM)
        if available["math"]:
            acc = mmlu_accuracy_mc(model, tok, math_items)
            scores.append(acc); detail["math_acc"] = acc

        # Average over tasks that produced numeric scores
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

    # Save selection
    meta = dict(
        algo=args.algo,
        base_model=args.base_model,
        best_score=best["score"],
        best_params=best["params"],
        best_model_dir=best["model_dir"],
        metrics=best["detail"],
        available_tasks=available,
        note="IFEval is used only if an evaluator is installed; otherwise averaged over coding+math."
    )
    with open(os.path.join(args.save_path, f"best_{args.algo}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== BEST CONFIG ===")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
