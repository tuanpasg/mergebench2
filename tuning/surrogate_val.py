# surrogate_val.py
from datasets import load_dataset, Dataset
import random

def _take(ds, n): 
    idx = list(range(min(len(ds), n))); random.shuffle(idx); return ds.select(idx)

def load_ifeval_like(n=300):
    # try common HF IDs; fall back to simple instruction-style texts if unavailable
    for repo in ["lukaemon/IFEval", "TIGER-Lab/IFEval"]:
        try:
            ds = load_dataset(repo, split="validation")
            texts = [ex.get("prompt", "") or ex.get("instruction", "") for ex in _take(ds, n)]
            return [t for t in texts if t]
        except Exception:
            pass
    # fallback tiny prompts
    return [f"Follow the instruction precisely #{i}: Write a one-sentence answer to the question." for i in range(n)]

def load_mmlu_stem(n=300):
    # Use STEM-heavy subjects as math proxy
    subs = [
        "high_school_mathematics","college_mathematics","abstract_algebra",
        "high_school_physics","physics","high_school_chemistry","chemistry",
        "computer_science","electrical_engineering","machine_learning"
    ]
    try:
        ds_all = []
        for s in subs:
            ds_all.append(load_dataset("cais/mmlu", s, split="test"))
        ds = Dataset.from_dict({k: sum((d[k] for d in ds_all), [])} for k in ds_all[0].features)
        ds = _take(ds, n)
        texts = []
        for ex in ds:
            q = ex.get("question","")
            choices = " ".join([f"({opt}) {ex.get(opt,'')}" for opt in ["A","B","C","D"] if ex.get(opt)])
            texts.append(f"{q}\n{choices}\nAnswer with the correct option.")
        return texts
    except Exception:
        return [f"Solve the math problem #{i}: 12x + 5 = 29. Give x." for i in range(n)]

def load_conala(n=300):
    for repo, split in [("neulab/conala","train"), ("neulab/conala","validation")]:
        try:
            ds = load_dataset(repo, split=split)
            return [ex.get("intent","") for ex in _take(ds, n) if ex.get("intent")]
        except Exception:
            pass
    return [f"Write a Python one-liner to reverse a list #{i}." for i in range(n)]

def get_surrogate_texts(per_task=200):
    return {
        "instruction": load_ifeval_like(per_task),
        "math":        load_mmlu_stem(per_task),
        "code":        load_conala(per_task),
    }
