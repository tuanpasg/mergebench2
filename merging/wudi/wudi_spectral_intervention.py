#!/usr/bin/env python3
"""Create spectral-intervention checkpoints for Adam-WUDI.

The script performs a causal intervention on the *late* Adam-WUDI movement

    Delta_late,l = W_adam300,l - W_adam15,l

using the right-side curvature matrix of WUDI Eq. (21)

    H_l = sum_i [1 / ||tau_i,l||_F^2] tau_i,l^T tau_i,l,
    tau_i,l = W_expert_i,l - W_base,l.

For each selected 2-D linear weight, the top-r eigenspace of H_l defines

    P_high = V_high V_high^T,
    Delta_high = Delta_late P_high,
    Delta_low  = Delta_late - Delta_high.

Here ``low`` means the complement of the dominant/high-curvature eigenspace;
it contains middle- and low-curvature directions. This definition guarantees

    Delta_late = Delta_high + Delta_low

without requiring the extremely expensive smallest-eigenvalue solve.

The script can build the following full Hugging Face checkpoints:

1. high_raw
       W = W15 + Delta_high

2. low_raw
       W = W15 + Delta_low

3. filter_<alpha>
       W = W15 + Delta_high + alpha * Delta_low
       alpha=0 recovers high_raw; alpha=1 recovers Adam-300 on selected keys.

4. high_match_<fraction> and low_match_<fraction>
       Both directions are rescaled per key to the same target norm
       fraction * ||Delta_late||_F. These checkpoints reduce the confounding
       effect of unequal high/low component magnitudes.

Implementation choices
----------------------
* Checkpoints are read lazily from safetensors / HF sharded safetensors.
* The top eigenspace is estimated on GPU through randomized subspace
  iteration using only matrix products H @ Q. The full H matrix is never
  materialized, so down_proj is supported.
* Projected high-curvature components are cached once, then reused to build
  multiple full checkpoints sequentially. Peak RAM therefore stays close to
  one loaded causal-LM checkpoint plus one weight matrix.
* Output checkpoints follow the same full-model ``save_pretrained`` convention
  as the current merge codebase.

Dependencies
------------
    pip install torch transformers accelerate safetensors

Example
-------
    python wudi_spectral_intervention.py \
      --base /models/Llama-3.2-3B \
      --expert instruction=/models/instruction \
      --expert math=/models/math \
      --expert coding=/models/coding \
      --adam15 /merges/adam_iter15 \
      --adam300 /merges/adam_iter300 \
      --output-root /merges/spectral_intervention_r32 \
      --rank 32 \
      --device cuda \
      --variants high,low \
      --matched-fractions 0.25 \
      --filter-alphas 0.25,0.5

For a cheap smoke test, add:

      --layers 0,14,27 --modules q_proj,o_proj,up_proj --rank 8

In that partial-key setting, unselected parameters remain at Adam-15.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LOGGER = logging.getLogger("wudi_spectral_intervention")

DEFAULT_KEY_REGEX = (
    r"model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
MODULE_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str
    value: Optional[float] = None


class CheckpointReader:
    """Lazy reader for local HF safetensors or PyTorch checkpoint shards."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {self.path}. "
                "This script currently expects local checkpoints."
            )

        self.format: str
        self.weight_map: Dict[str, Path] = {}
        self._bin_cache: Dict[Path, Mapping[str, torch.Tensor]] = {}

        if self.path.is_file():
            if self.path.suffix == ".safetensors":
                self.format = "safetensors"
                self._scan_safetensors_files([self.path])
            elif self.path.suffix in {".bin", ".pt", ".pth"}:
                self.format = "bin"
                self._scan_bin_files([self.path])
            else:
                raise ValueError(f"Unsupported checkpoint file: {self.path}")
            return

        safe_index = self.path / "model.safetensors.index.json"
        bin_index = self.path / "pytorch_model.bin.index.json"
        single_safe = self.path / "model.safetensors"
        single_bin = self.path / "pytorch_model.bin"

        if safe_index.exists():
            self.format = "safetensors"
            self._read_index(safe_index)
        elif single_safe.exists():
            self.format = "safetensors"
            self._scan_safetensors_files([single_safe])
        elif bin_index.exists():
            self.format = "bin"
            self._read_index(bin_index)
        elif single_bin.exists():
            self.format = "bin"
            self._scan_bin_files([single_bin])
        else:
            safe_files = sorted(self.path.glob("*.safetensors"))
            bin_files = sorted(self.path.glob("*.bin"))
            if safe_files:
                self.format = "safetensors"
                self._scan_safetensors_files(safe_files)
            elif bin_files:
                self.format = "bin"
                self._scan_bin_files(bin_files)
            else:
                raise FileNotFoundError(
                    f"No HF model weights found under: {self.path}"
                )

    def _read_index(self, index_path: Path) -> None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        raw_map = payload.get("weight_map")
        if not isinstance(raw_map, dict):
            raise ValueError(f"Invalid HF checkpoint index: {index_path}")
        self.weight_map = {
            key: index_path.parent / filename for key, filename in raw_map.items()
        }

    def _scan_safetensors_files(self, files: Sequence[Path]) -> None:
        for file in files:
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in self.weight_map:
                        raise ValueError(f"Duplicate tensor key {key!r} in {self.path}")
                    self.weight_map[key] = file

    def _load_bin_shard(self, file: Path) -> Mapping[str, torch.Tensor]:
        if file not in self._bin_cache:
            payload = torch.load(file, map_location="cpu", weights_only=True)
            if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
                payload = payload["state_dict"]
            if not isinstance(payload, dict):
                raise ValueError(f"Unsupported PyTorch checkpoint payload: {file}")
            self._bin_cache[file] = payload
        return self._bin_cache[file]

    def _scan_bin_files(self, files: Sequence[Path]) -> None:
        for file in files:
            state = self._load_bin_shard(file)
            for key in state.keys():
                if key in self.weight_map:
                    raise ValueError(f"Duplicate tensor key {key!r} in {self.path}")
                self.weight_map[key] = file

    def keys(self) -> Iterable[str]:
        return self.weight_map.keys()

    def has_key(self, key: str) -> bool:
        return key in self.weight_map

    def get_tensor(self, key: str) -> torch.Tensor:
        file = self.weight_map.get(key)
        if file is None:
            raise KeyError(f"Tensor {key!r} not found in {self.path}")
        if self.format == "safetensors":
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(key)
        else:
            tensor = self._load_bin_shard(file)[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Checkpoint object {key!r} is not a tensor")
        return tensor.detach().cpu().contiguous()

    def clear_cache(self) -> None:
        self._bin_cache.clear()
        gc.collect()


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Expert name cannot be empty")
    return name, Path(path).expanduser()


def parse_csv_ints(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    try:
        return {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value!r}") from exc


def parse_csv_strings(value: Optional[str]) -> Optional[set[str]]:
    if value is None or not value.strip():
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_csv_floats(value: Optional[str]) -> List[float]:
    if value is None or not value.strip():
        return []
    try:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value!r}") from exc


def parse_variants(value: str) -> set[str]:
    variants = parse_csv_strings(value) or set()
    allowed = {"high", "low"}
    unknown = variants - allowed
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown core variants {sorted(unknown)}; allowed: high,low"
        )
    return variants


def key_metadata(key: str) -> Tuple[Optional[int], str]:
    layer_match = LAYER_RE.search(key)
    module_match = MODULE_RE.search(key)
    layer = int(layer_match.group(1)) if layer_match else None
    module = module_match.group(1) if module_match else "unknown"
    return layer, module


def select_keys(
    base_reader: CheckpointReader,
    readers: Sequence[CheckpointReader],
    include_regex: str,
    layers: Optional[set[int]],
    modules: Optional[set[str]],
    max_keys: Optional[int],
) -> List[str]:
    pattern = re.compile(include_regex)
    selected: List[str] = []
    for key in sorted(base_reader.keys()):
        if not pattern.search(key):
            continue
        layer, module = key_metadata(key)
        if layers is not None and layer not in layers:
            continue
        if modules is not None and module not in modules:
            continue
        if any(not reader.has_key(key) for reader in readers):
            LOGGER.warning("Skipping key absent from at least one checkpoint: %s", key)
            continue
        selected.append(key)
        if max_keys is not None and len(selected) >= max_keys:
            break
    return selected


def cache_filename(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.safetensors"


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def h_matmat(
    q: torch.Tensor,
    tasks: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    """Apply H Q = sum_i w_i T_i^T(T_i Q) without materializing H."""
    result = torch.zeros_like(q)
    for task, weight in zip(tasks, weights):
        result.add_(task.T @ (task @ q), alpha=float(weight))
    return result


def randomized_top_eigenspace(
    tasks_cpu: Sequence[torch.Tensor],
    weights: Sequence[float],
    rank: int,
    oversample: int,
    power_iters: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Estimate top right-eigenvectors of H using GPU subspace iteration."""
    if not tasks_cpu:
        raise ValueError("No task matrices supplied")
    in_dim = tasks_cpu[0].shape[1]
    if any(task.ndim != 2 or task.shape[1] != in_dim for task in tasks_cpu):
        raise ValueError("Task matrices must be 2-D with equal input dimension")

    effective_rank = min(rank, in_dim)
    q_dim = min(in_dim, effective_rank + max(oversample, 0))
    tasks = [task.to(device=device, dtype=dtype, non_blocking=True) for task in tasks_cpu]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn((in_dim, q_dim), device=device, dtype=dtype, generator=generator)
    q = torch.linalg.qr(q, mode="reduced").Q

    # At least one H application is required. Additional iterations sharpen
    # separation between dominant and weak directions.
    for _ in range(max(power_iters, 1)):
        z = h_matmat(q, tasks, weights)
        q = torch.linalg.qr(z, mode="reduced").Q

    hq = h_matmat(q, tasks, weights)
    small = q.T @ hq
    small = 0.5 * (small + small.T)
    evals_small, evecs_small = torch.linalg.eigh(small)
    order = torch.argsort(evals_small, descending=True)[:effective_rank]
    evals = evals_small[order]
    vectors = q @ evecs_small[:, order]
    vectors = torch.linalg.qr(vectors, mode="reduced").Q

    # Recompute Rayleigh values after the final QR, since QR can slightly
    # rotate the basis inside the estimated invariant subspace.
    hv = h_matmat(vectors, tasks, weights)
    rayleigh_matrix = vectors.T @ hv
    rayleigh_matrix = 0.5 * (rayleigh_matrix + rayleigh_matrix.T)
    rayleigh_values, rotation = torch.linalg.eigh(rayleigh_matrix)
    order = torch.argsort(rayleigh_values, descending=True)
    evals = rayleigh_values[order]
    vectors = vectors @ rotation[:, order]
    hv = h_matmat(vectors, tasks, weights)

    residual = hv - vectors * evals.unsqueeze(0)
    residual_ratio = float(
        residual.norm().double().item() / max(hv.norm().double().item(), eps)
    )
    orthogonality_error = float(
        (vectors.T @ vectors - torch.eye(effective_rank, device=device, dtype=dtype))
        .norm()
        .double()
        .item()
    )
    trace_h = float(
        sum(float(weight) * float(torch.sum(task.double() ** 2).item())
            for task, weight in zip(tasks_cpu, weights))
    )
    explained_trace = float(evals.double().sum().item() / max(trace_h, eps))

    diagnostics = {
        "trace_h": trace_h,
        "top_eigenvalue": float(evals[0].double().item()) if len(evals) else 0.0,
        "bottom_selected_eigenvalue": float(evals[-1].double().item()) if len(evals) else 0.0,
        "selected_eigenvalue_sum": float(evals.double().sum().item()),
        "selected_trace_fraction": explained_trace,
        "eigenspace_residual_ratio": residual_ratio,
        "orthogonality_error": orthogonality_error,
    }

    vectors_cpu = vectors.detach().cpu().to(torch.float32).contiguous()
    evals_cpu = evals.detach().cpu().to(torch.float64).contiguous()
    del tasks, q, hq, small, evals_small, evecs_small, vectors, hv, residual
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vectors_cpu, evals_cpu, diagnostics


def project_right(
    matrix_cpu: torch.Tensor,
    basis_cpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    row_chunk: int,
) -> torch.Tensor:
    """Compute (M V) V^T in row chunks."""
    basis = basis_cpu.to(device=device, dtype=dtype, non_blocking=True)
    result = torch.empty_like(matrix_cpu, dtype=torch.float32, device="cpu")
    for start in range(0, matrix_cpu.shape[0], row_chunk):
        block = matrix_cpu[start : start + row_chunk].to(
            device=device, dtype=dtype, non_blocking=True
        )
        projected = (block @ basis) @ basis.T
        result[start : start + projected.shape[0]].copy_(
            projected.detach().cpu().to(torch.float32)
        )
        del block, projected
    del basis
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def fro_norm(tensor: torch.Tensor) -> float:
    # Float32 is sufficient for checkpoint-scale diagnostics and avoids
    # allocating a full float64 copy of large LLM matrices.
    return float(torch.linalg.vector_norm(tensor.float()).item())


def normalized_rayleigh(
    matrix_cpu: torch.Tensor,
    tasks_cpu: Sequence[torch.Tensor],
    weights: Sequence[float],
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
    row_chunk: int,
) -> float:
    """Compute tr(M H M^T)/||M||^2 by chunked H applications on GPU."""
    denominator = float(torch.sum(matrix_cpu.float() ** 2).item())
    if denominator <= eps:
        return 0.0

    tasks = [task.to(device=device, dtype=dtype, non_blocking=True) for task in tasks_cpu]
    numerator = 0.0
    for start in range(0, matrix_cpu.shape[0], row_chunk):
        block = matrix_cpu[start : start + row_chunk].to(
            device=device, dtype=dtype, non_blocking=True
        )
        # Each row of block is a right-space vector. Apply H to block^T.
        h_block_t = h_matmat(block.T, tasks, weights)
        numerator += float(torch.sum(block.T * h_block_t).float().item())
        del block, h_block_t

    del tasks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return numerator / denominator


def save_high_component(
    cache_path: Path,
    high: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    save_file({"high": high.to(dtype=dtype).contiguous()}, str(tmp_path))
    os.replace(tmp_path, cache_path)


def read_high_component(cache_path: Path) -> torch.Tensor:
    payload = load_file(str(cache_path), device="cpu")
    high = payload.get("high")
    if high is None:
        raise KeyError(f"Cache file lacks tensor 'high': {cache_path}")
    return high.detach().cpu().to(torch.float32).contiguous()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_variant_specs(
    core_variants: set[str],
    filter_alphas: Sequence[float],
    matched_fractions: Sequence[float],
) -> List[VariantSpec]:
    specs: List[VariantSpec] = []
    if "high" in core_variants:
        specs.append(VariantSpec("high_raw", "high"))
    if "low" in core_variants:
        specs.append(VariantSpec("low_raw", "low"))

    for alpha in filter_alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Filter alpha must lie in [0,1], got {alpha}")
        token = str(alpha).replace("-", "m").replace(".", "p")
        specs.append(VariantSpec(f"filter_a{token}", "filter", alpha))

    for fraction in matched_fractions:
        if fraction <= 0:
            raise ValueError(f"Matched fraction must be > 0, got {fraction}")
        token = str(fraction).replace("-", "m").replace(".", "p")
        specs.append(VariantSpec(f"high_match_f{token}", "high_match", fraction))
        specs.append(VariantSpec(f"low_match_f{token}", "low_match", fraction))

    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate output variant names")
    if not specs:
        raise ValueError("No output variants requested")
    return specs


def resolve_new_weight(
    spec: VariantSpec,
    w15: torch.Tensor,
    late: torch.Tensor,
    high: torch.Tensor,
    metrics: Mapping[str, object],
    eps: float,
    max_match_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    low = late - high
    aux: Dict[str, float] = {}

    if spec.kind == "high":
        update = high
    elif spec.kind == "low":
        update = low
    elif spec.kind == "filter":
        assert spec.value is not None
        update = high + float(spec.value) * low
    elif spec.kind in {"high_match", "low_match"}:
        assert spec.value is not None
        source = high if spec.kind == "high_match" else low
        source_norm = fro_norm(source)
        late_norm = float(metrics["late_norm"])
        target_norm = float(spec.value) * late_norm
        scale = target_norm / max(source_norm, eps)
        unclipped_scale = scale
        if max_match_scale > 0:
            scale = min(scale, max_match_scale)
        update = scale * source
        aux = {
            "target_norm": target_norm,
            "source_norm": source_norm,
            "unclipped_scale": unclipped_scale,
            "applied_scale": scale,
            "scale_was_clipped": float(scale != unclipped_scale),
        }
    else:
        raise ValueError(f"Unknown variant kind: {spec.kind}")

    return w15 + update, aux


def copy_or_save_tokenizer(source: str | Path, output_dir: Path, trust_remote_code: bool) -> None:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(source), trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:  # Model checkpoint remains valid without tokenizer files.
        LOGGER.warning("Could not save tokenizer from %s: %s", source, exc)


def load_manifest(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, payload: Mapping[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)



def build_cache_signature(args: argparse.Namespace, selected_keys: Sequence[str]) -> Dict[str, object]:
    return {
        "base": str(Path(args.base).expanduser().resolve()),
        "experts": [(name, str(Path(path).expanduser().resolve())) for name, path in args.expert],
        "adam15": str(Path(args.adam15).expanduser().resolve()),
        "adam300": str(Path(args.adam300).expanduser().resolve()),
        "selected_keys": list(selected_keys),
        "rank": args.rank,
        "oversample": args.oversample,
        "power_iters": args.power_iters,
        "spectral_dtype": args.spectral_dtype,
        "cache_dtype": args.cache_dtype,
        "seed": args.seed,
    }


def signatures_match(a: Mapping[str, object], b: Mapping[str, object]) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)

def compute_component_cache(args: argparse.Namespace) -> Tuple[Dict[str, object], CheckpointReader]:
    output_root = Path(args.output_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else output_root / "_spectral_cache"
    component_dir = cache_dir / "components"
    manifest_path = cache_dir / "cache_manifest.json"
    metrics_csv = cache_dir / "component_metrics.csv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    component_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Opening checkpoints lazily")
    base_reader = CheckpointReader(args.base)
    expert_readers = {
        name: CheckpointReader(path) for name, path in args.expert
    }
    adam15_reader = CheckpointReader(args.adam15)
    adam300_reader = CheckpointReader(args.adam300)

    all_readers = [*expert_readers.values(), adam15_reader, adam300_reader]
    keys = select_keys(
        base_reader,
        all_readers,
        include_regex=args.include_regex,
        layers=parse_csv_ints(args.layers),
        modules=parse_csv_strings(args.modules),
        max_keys=args.max_keys,
    )
    if not keys:
        raise RuntimeError("No matching linear weight keys were found")

    cache_signature = build_cache_signature(args, keys)
    prior_manifest: Dict[str, object] = {}
    prior_by_key: Dict[str, Mapping[str, object]] = {}
    if args.resume and manifest_path.exists():
        prior_manifest = load_manifest(manifest_path)
        prior_signature = prior_manifest.get("cache_signature", {})
        if not signatures_match(prior_signature, cache_signature):
            message = (
                "Existing cache is incompatible with the current checkpoints or spectral settings. "
                "Use a different --cache-dir or pass --force-recompute-cache."
            )
            if not args.force_recompute_cache:
                raise RuntimeError(message)
            LOGGER.warning("%s Recomputing cache.", message)
            shutil.rmtree(component_dir, ignore_errors=True)
            component_dir.mkdir(parents=True, exist_ok=True)
        else:
            prior_by_key = {
                str(row["key"]): row
                for row in prior_manifest.get("components", [])
                if isinstance(row, dict) and "key" in row
            }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    spectral_dtype = torch_dtype(args.spectral_dtype)
    cache_dtype = torch_dtype(args.cache_dtype)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    started = time.perf_counter()
    expert_names = list(expert_readers.keys())

    LOGGER.info("Computing high-curvature components for %d keys", len(keys))
    for index, key in enumerate(keys, start=1):
        layer, module = key_metadata(key)
        filename = cache_filename(key)
        component_path = component_dir / filename

        if args.resume and component_path.exists() and key in prior_by_key:
            LOGGER.info("[%d/%d] cached: %s", index, len(keys), key)
            rows.append(dict(prior_by_key[key]))
            continue

        LOGGER.info("[%d/%d] projecting: %s", index, len(keys), key)
        key_start = time.perf_counter()
        try:
            base = base_reader.get_tensor(key).to(torch.float32)
            if base.ndim != 2:
                raise ValueError(f"Expected 2-D tensor, got shape={tuple(base.shape)}")

            tasks: List[torch.Tensor] = []
            for name in expert_names:
                expert = expert_readers[name].get_tensor(key)
                if expert.shape != base.shape:
                    raise ValueError(
                        f"Shape mismatch for expert {name}: {tuple(expert.shape)} vs {tuple(base.shape)}"
                    )
                tasks.append(expert.to(torch.float32) - base)

            norms_sq = [float(torch.sum(task.double() ** 2).item()) for task in tasks]
            if any(value <= args.eps for value in norms_sq):
                raise ValueError(f"Near-zero task-vector norm: {norms_sq}")
            weights = [1.0 / (value + args.eps) for value in norms_sq]

            w15 = adam15_reader.get_tensor(key).to(torch.float32)
            w300 = adam300_reader.get_tensor(key).to(torch.float32)
            if w15.shape != base.shape or w300.shape != base.shape:
                raise ValueError("Adam checkpoint shape mismatch")
            late = w300 - w15

            key_seed = args.seed + int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
            basis, evals, diag = randomized_top_eigenspace(
                tasks_cpu=tasks,
                weights=weights,
                rank=args.rank,
                oversample=args.oversample,
                power_iters=args.power_iters,
                device=device,
                dtype=spectral_dtype,
                seed=key_seed,
                eps=args.eps,
            )
            high = project_right(
                late,
                basis,
                device=device,
                dtype=spectral_dtype,
                row_chunk=args.projection_chunk_rows,
            )
            low = late - high

            late_norm = fro_norm(late)
            high_norm = fro_norm(high)
            low_norm = fro_norm(low)
            reconstruction_error = fro_norm(late - high - low) / max(late_norm, args.eps)
            inner = float(torch.sum(high.float() * low.float()).item())
            orthogonal_ratio = abs(inner) / max(high_norm * low_norm, args.eps)

            save_high_component(component_path, high, cache_dtype)
            cached_high = read_high_component(component_path)
            cache_quantization_error = fro_norm(high - cached_high) / max(high_norm, args.eps)

            row: Dict[str, object] = {
                "key": key,
                "layer": layer,
                "module": module,
                "out_dim": base.shape[0],
                "in_dim": base.shape[1],
                "rank": basis.shape[1],
                "cache_file": filename,
                "late_norm": late_norm,
                "high_norm": high_norm,
                "low_norm": low_norm,
                "high_energy_fraction": (high_norm / max(late_norm, args.eps)) ** 2,
                "low_energy_fraction": (low_norm / max(late_norm, args.eps)) ** 2,
                "high_low_inner_product": inner,
                "high_low_cos_abs": orthogonal_ratio,
                "reconstruction_relative_error": reconstruction_error,
                "cache_quantization_relative_error": cache_quantization_error,
                "late_rayleigh": normalized_rayleigh(
                    late, tasks, weights, args.eps, device, spectral_dtype,
                    args.rayleigh_chunk_rows
                ),
                "high_rayleigh": normalized_rayleigh(
                    high, tasks, weights, args.eps, device, spectral_dtype,
                    args.rayleigh_chunk_rows
                ),
                "low_rayleigh": normalized_rayleigh(
                    low, tasks, weights, args.eps, device, spectral_dtype,
                    args.rayleigh_chunk_rows
                ),
                "runtime_sec": time.perf_counter() - key_start,
                **diag,
            }
            rows.append(row)
            write_csv(metrics_csv, rows)

            manifest = {
                "version": 1,
                "complete": False,
                "arguments": vars(args),
                "cache_signature": cache_signature,
                "selected_keys": keys,
                "components": rows,
                "failures": failures,
            }
            save_manifest(manifest_path, manifest)

            del base, tasks, w15, w300, late, basis, evals, high, low, cached_high
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            LOGGER.exception("Failed to process key %s", key)
            failures.append({"key": key, "error": repr(exc)})
            if args.fail_fast:
                raise

    completed_keys = {str(row["key"]) for row in rows}
    missing_keys = [key for key in keys if key not in completed_keys]
    manifest = {
        "version": 1,
        "complete": not missing_keys and not failures,
        "arguments": vars(args),
        "cache_signature": cache_signature,
        "base": str(Path(args.base).expanduser()),
        "experts": {name: str(path) for name, path in args.expert},
        "adam15": str(Path(args.adam15).expanduser()),
        "adam300": str(Path(args.adam300).expanduser()),
        "selected_keys": keys,
        "completed_keys": len(rows),
        "missing_keys": missing_keys,
        "components": rows,
        "failures": failures,
        "cache_dir": str(cache_dir),
        "runtime_sec": time.perf_counter() - started,
        "decomposition": {
            "P_high": "top-r right eigenspace of H_l",
            "P_low": "I - P_high (residual complement; includes middle and low curvature)",
            "identity": "Delta_late = Delta_high + Delta_low",
        },
    }
    save_manifest(manifest_path, manifest)
    write_csv(metrics_csv, rows)

    for reader in [base_reader, *expert_readers.values(), adam15_reader]:
        reader.clear_cache()
    # Keep Adam-300 reader for checkpoint generation.
    return manifest, adam300_reader


def build_checkpoints(
    args: argparse.Namespace,
    manifest: Mapping[str, object],
    adam300_reader: CheckpointReader,
) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    cache_dir = Path(manifest["cache_dir"])
    component_dir = cache_dir / "components"
    rows = [row for row in manifest.get("components", []) if isinstance(row, dict)]
    row_by_key: Dict[str, Mapping[str, object]] = {str(row["key"]): row for row in rows}
    selected_keys = [str(key) for key in manifest.get("selected_keys", [])]

    missing = [key for key in selected_keys if key not in row_by_key]
    if missing:
        raise RuntimeError(
            f"Cannot build checkpoints: {len(missing)} selected keys lack cached components"
        )

    specs = build_variant_specs(
        core_variants=parse_variants(args.variants),
        filter_alphas=parse_csv_floats(args.filter_alphas),
        matched_fractions=parse_csv_floats(args.matched_fractions),
    )
    model_dtype = torch_dtype(args.model_dtype)
    model_kwargs = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.model_device_map.lower() != "none":
        model_kwargs["device_map"] = args.model_device_map

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint generation requires transformers: pip install transformers accelerate"
        ) from exc

    LOGGER.info("Building %d full checkpoints sequentially", len(specs))
    for variant_index, spec in enumerate(specs, start=1):
        output_dir = output_root / spec.name
        if output_dir.exists():
            if not args.overwrite:
                LOGGER.info(
                    "[%d/%d] output exists, skipping: %s",
                    variant_index,
                    len(specs),
                    output_dir,
                )
                continue
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "[%d/%d] loading Adam-15 model for %s",
            variant_index,
            len(specs),
            spec.name,
        )
        model = AutoModelForCausalLM.from_pretrained(str(args.adam15), **model_kwargs)
        param_map = dict(model.named_parameters())
        per_key_build_rows: List[Dict[str, object]] = []

        started = time.perf_counter()
        with torch.no_grad():
            for key_index, key in enumerate(selected_keys, start=1):
                if key not in param_map:
                    raise KeyError(
                        f"Selected checkpoint key is not a named model parameter: {key}"
                    )
                row = row_by_key[key]
                component_path = component_dir / str(row["cache_file"])
                high = read_high_component(component_path)
                parameter = param_map[key]
                w15 = parameter.detach().cpu().to(torch.float32)
                w300 = adam300_reader.get_tensor(key).to(torch.float32)
                if w300.shape != w15.shape or high.shape != w15.shape:
                    raise ValueError(f"Shape mismatch while building {spec.name}: {key}")
                late = w300 - w15

                new_weight, aux = resolve_new_weight(
                    spec=spec,
                    w15=w15,
                    late=late,
                    high=high,
                    metrics=row,
                    eps=args.eps,
                    max_match_scale=args.max_match_scale,
                )
                parameter.copy_(new_weight.to(device=parameter.device, dtype=parameter.dtype))

                per_key_build_rows.append(
                    {
                        "key": key,
                        "variant": spec.name,
                        "kind": spec.kind,
                        "value": spec.value,
                        "new_update_norm_from_w15": fro_norm(new_weight - w15),
                        **aux,
                    }
                )
                if key_index % max(args.log_every, 1) == 0 or key_index == len(selected_keys):
                    LOGGER.info(
                        "    %s: %d/%d keys",
                        spec.name,
                        key_index,
                        len(selected_keys),
                    )
                del high, w15, w300, late, new_weight

        model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer_source = args.tokenizer_source or args.adam15
        copy_or_save_tokenizer(tokenizer_source, output_dir, args.trust_remote_code)

        variant_metadata = {
            "algorithm": "Adam-WUDI spectral intervention",
            "variant": spec.name,
            "kind": spec.kind,
            "value": spec.value,
            "base_checkpoint": str(args.base),
            "adam15_checkpoint": str(args.adam15),
            "adam300_checkpoint": str(args.adam300),
            "rank": args.rank,
            "selected_key_count": len(selected_keys),
            "selected_keys": selected_keys,
            "unselected_parameters": "copied from Adam-15",
            "runtime_sec": time.perf_counter() - started,
            "formulae": {
                "high_raw": "W15 + P_high (W300-W15)",
                "low_raw": "W15 + (I-P_high) (W300-W15)",
                "filter": "W15 + P_high Delta_late + alpha (I-P_high) Delta_late",
                "matched": "direction component rescaled per key to fraction * ||Delta_late||_F",
            },
            "cache_manifest": str(cache_dir / "cache_manifest.json"),
            "arguments": vars(args),
        }
        (output_dir / "spectral_intervention.json").write_text(
            json.dumps(variant_metadata, indent=2, default=str), encoding="utf-8"
        )
        write_csv(output_dir / "per_key_intervention.csv", per_key_build_rows)
        (output_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {spec.name}",
                    "",
                    "Full Hugging Face checkpoint generated by Adam-WUDI spectral intervention.",
                    "",
                    f"- Adam-15 source: `{args.adam15}`",
                    f"- Adam-300 source: `{args.adam300}`",
                    f"- High-curvature rank: `{args.rank}`",
                    f"- Selected keys: `{len(selected_keys)}`",
                    "- Unselected parameters remain at Adam-15.",
                    "",
                    "See `spectral_intervention.json` and `per_key_intervention.csv` for details.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        LOGGER.info("Saved checkpoint: %s", output_dir)

        del model, param_map, per_key_build_rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create spectral-intervention Adam-WUDI checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", required=True, help="Local base-model checkpoint")
    parser.add_argument(
        "--expert",
        action="append",
        required=True,
        type=parse_named_path,
        metavar="NAME=PATH",
        help="Task expert checkpoint; repeat for every task",
    )
    parser.add_argument("--adam15", required=True, help="Adam-WUDI iter-15 checkpoint")
    parser.add_argument("--adam300", required=True, help="Adam-WUDI iter-300 checkpoint")
    parser.add_argument("--output-root", required=True, help="Root directory for output checkpoints")
    parser.add_argument("--cache-dir", default=None, help="Reusable component cache directory")

    parser.add_argument("--include-regex", default=DEFAULT_KEY_REGEX)
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices")
    parser.add_argument(
        "--modules",
        default=None,
        help="Comma-separated modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--max-keys", type=int, default=None)

    parser.add_argument("--rank", type=int, default=32, help="Dimension of P_high")
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iters", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--spectral-dtype", choices=["float32"], default="float32",
        help="QR/eigh are intentionally kept in float32 for numerical stability"
    )
    parser.add_argument(
        "--cache-dtype", choices=["float32", "float16", "bfloat16"], default="float16"
    )
    parser.add_argument("--projection-chunk-rows", type=int, default=1024)
    parser.add_argument(
        "--rayleigh-chunk-rows", type=int, default=128,
        help="Row chunk used for exact Rayleigh diagnostics"
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--variants",
        default="high,low",
        help="Core raw variants to build: high,low",
    )
    parser.add_argument(
        "--filter-alphas",
        default="",
        help="Comma-separated alpha values for W15 + high + alpha*low",
    )
    parser.add_argument(
        "--matched-fractions",
        default="",
        help="Comma-separated per-key target norm fractions for matched high/low controls",
    )
    parser.add_argument(
        "--max-match-scale",
        type=float,
        default=10.0,
        help="Cap for matched-direction amplification; <=0 disables clipping",
    )

    parser.add_argument(
        "--model-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--model-device-map", default="cpu")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--tokenizer-source", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--resume", action="store_true", help="Reuse completed cache entries")
    parser.add_argument(
        "--force-recompute-cache", action="store_true",
        help="Discard an incompatible existing cache instead of raising an error"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directories")
    parser.add_argument("--compute-only", action="store_true", help="Only build spectral cache")
    parser.add_argument("--build-only", action="store_true", help="Only build checkpoints from an existing cache")
    parser.add_argument("--keep-cache", action="store_true", help="Keep cache after checkpoint generation")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.rank <= 0:
        parser.error("--rank must be positive")
    if args.oversample < 0:
        parser.error("--oversample must be non-negative")
    if args.power_iters < 1:
        parser.error("--power-iters must be at least 1")
    if args.compute_only and args.build_only:
        parser.error("--compute-only and --build-only are mutually exclusive")
    if args.projection_chunk_rows <= 0 or args.rayleigh_chunk_rows <= 0:
        parser.error("Chunk sizes must be positive")
    if len({name for name, _ in args.expert}) != len(args.expert):
        parser.error("Expert names must be unique")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else output_root / "_spectral_cache"
    manifest_path = cache_dir / "cache_manifest.json"

    adam300_reader: Optional[CheckpointReader] = None
    if args.build_only:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"--build-only requested but cache manifest does not exist: {manifest_path}"
            )
        manifest = load_manifest(manifest_path)
        cached_rank = manifest.get("cache_signature", {}).get("rank")
        if cached_rank is not None and int(cached_rank) != args.rank:
            LOGGER.warning(
                "Using cached rank=%s instead of command-line rank=%s", cached_rank, args.rank
            )
            args.rank = int(cached_rank)
        adam300_reader = CheckpointReader(args.adam300)
    else:
        manifest, adam300_reader = compute_component_cache(args)

    if not args.compute_only:
        assert adam300_reader is not None
        build_checkpoints(args, manifest, adam300_reader)

    if adam300_reader is not None:
        adam300_reader.clear_cache()

    if not args.compute_only and not args.keep_cache:
        LOGGER.info("Removing reusable component cache: %s", cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)
    elif cache_dir.exists():
        LOGGER.info("Keeping reusable component cache: %s", cache_dir)

    LOGGER.info("Done. Output root: %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
