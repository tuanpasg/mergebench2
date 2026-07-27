#!/usr/bin/env python3
"""Spectral diagnosis for Adam-WUDI / CFS-WUDI checkpoints.

This script tests the hypothesis that late Adam-WUDI updates and weakly
regularized CFS-WUDI solutions move increasingly into low-curvature directions
of the Eq. (21) quadratic objective.

For every selected 2-D linear weight W_l, it constructs task vectors

    tau_i,l = W_i,l - W_base,l

and the Eq. (21) right-side curvature matrix

    H_l = sum_i [1 / ||tau_i,l||_F^2] tau_i,l^T tau_i,l.

It then:
  1. estimates top and bottom eigen-directions of H_l;
  2. projects Adam/CFS task vectors and checkpoint differences into those
     spectral subspaces;
  3. compares early Adam movement (TA -> Adam-15) with late movement
     (Adam-15 -> Adam-300);
  4. measures whether CFS 1e-7 amplifies low-curvature directions relative to
     CFS 1e-5 / 1e-6;
  5. measures task-specific support and pairwise principal-subspace overlap;
  6. exports CSV files, JSON metadata, and aggregate plots.

Dependencies:
    pip install torch numpy pandas scipy matplotlib safetensors

The script reads Hugging Face checkpoints stored as:
  * a single .safetensors file;
  * a directory containing model.safetensors;
  * a sharded safetensors directory with model.safetensors.index.json;
  * .bin / sharded .bin checkpoints (supported but less memory efficient).

Example:
    python wudi_spectral_diagnosis.py \
      --base /models/Llama-3.2-3B \
      --expert instruction=/models/instruction \
      --expert math=/models/math \
      --expert coding=/models/coding \
      --adam15 /merges/adam_iter15 \
      --adam300 /merges/adam_iter300 \
      --cfs1e5 /merges/cfs_1e-5 \
      --cfs1e6 /merges/cfs_1e-6 \
      --cfs1e7 /merges/cfs_1e-7 \
      --layers 0,7,14,21,27 \
      --modules q_proj,o_proj,up_proj,down_proj \
      --device cuda \
      --output-dir spectral_results

Notes:
  * TA is computed as sum_i tau_i unless --ta-checkpoint is supplied.
  * For large input dimensions, the script uses matrix-free ARPACK eigensolvers
    and never materializes the full H_l matrix.
  * Bottom eigenvectors can be numerically unstable when a large nullspace has
    repeated near-zero eigenvalues. Use Rayleigh quotients together with tail
    energy fractions; do not interpret a single bottom eigenvector in isolation.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'safetensors'. Install with: pip install safetensors"
    ) from exc


LOGGER = logging.getLogger("wudi_spectral")
DEFAULT_KEY_REGEX = (
    r"model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
MODULE_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass
class ExtremalSpectrum:
    top_values: np.ndarray
    top_vectors: np.ndarray
    tail_values: np.ndarray
    tail_vectors: np.ndarray
    solver: str
    full_values: Optional[np.ndarray] = None


class CheckpointReader:
    """Lazy checkpoint reader for HF safetensors or PyTorch .bin checkpoints."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.path}")

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
                    f"No model weights found under checkpoint directory: {self.path}"
                )

    def _read_index(self, index_path: Path) -> None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        raw_map = payload.get("weight_map")
        if not isinstance(raw_map, dict):
            raise ValueError(f"Invalid HF index file: {index_path}")
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

    def _scan_bin_files(self, files: Sequence[Path]) -> None:
        for file in files:
            state = self._load_bin_shard(file)
            for key in state.keys():
                if key in self.weight_map:
                    raise ValueError(f"Duplicate tensor key {key!r} in {self.path}")
                self.weight_map[key] = file

    def _load_bin_shard(self, file: Path) -> Mapping[str, torch.Tensor]:
        if file not in self._bin_cache:
            payload = torch.load(file, map_location="cpu", weights_only=True)
            if "state_dict" in payload and isinstance(payload["state_dict"], dict):
                payload = payload["state_dict"]
            if not isinstance(payload, dict):
                raise ValueError(f"Unsupported PyTorch checkpoint payload: {file}")
            self._bin_cache[file] = payload
        return self._bin_cache[file]

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
            raise TypeError(f"Object {key!r} is not a tensor in {self.path}")
        return tensor.detach().cpu().contiguous()

    def clear_cache(self) -> None:
        self._bin_cache.clear()
        gc.collect()


class WudiHessianOperator:
    """Matrix-free H x = sum_i w_i T_i^T (T_i x)."""

    def __init__(self, task_matrices: Sequence[np.ndarray], weights: Sequence[float]):
        if not task_matrices:
            raise ValueError("At least one task matrix is required")
        self.tasks = task_matrices
        self.weights = np.asarray(weights, dtype=np.float64)
        self.in_dim = task_matrices[0].shape[1]
        self.dtype = np.result_type(*[task.dtype for task in task_matrices])
        for task in task_matrices:
            if task.ndim != 2 or task.shape[1] != self.in_dim:
                raise ValueError("All task matrices must be 2-D with equal input dimension")

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype)
        result = np.zeros(self.in_dim, dtype=self.dtype)
        for weight, task in zip(self.weights, self.tasks):
            result += weight * (task.T @ (task @ x))
        return result

    def matmat(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype)
        result = np.zeros((self.in_dim, x.shape[1]), dtype=self.dtype)
        for weight, task in zip(self.weights, self.tasks):
            result += weight * (task.T @ (task @ x))
        return result

    def as_scipy(self) -> LinearOperator:
        return LinearOperator(
            shape=(self.in_dim, self.in_dim),
            matvec=self.matvec,
            matmat=self.matmat,
            dtype=self.dtype,
        )


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, received {value!r}"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Expert name cannot be empty")
    return name, Path(path).expanduser()


def parse_csv_ints(value: Optional[str]) -> Optional[set[int]]:
    if value is None or value.strip() == "":
        return None
    try:
        return {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value}") from exc


def parse_csv_strings(value: Optional[str]) -> Optional[set[str]]:
    if value is None or value.strip() == "":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def key_metadata(key: str) -> Tuple[Optional[int], str]:
    layer_match = LAYER_RE.search(key)
    module_match = MODULE_RE.search(key)
    layer = int(layer_match.group(1)) if layer_match else None
    module = module_match.group(1) if module_match else "unknown"
    return layer, module


def select_keys(
    base_reader: CheckpointReader,
    all_readers: Sequence[CheckpointReader],
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
        if any(not reader.has_key(key) for reader in all_readers):
            LOGGER.warning("Skipping key missing from at least one checkpoint: %s", key)
            continue
        selected.append(key)
        if max_keys is not None and len(selected) >= max_keys:
            break
    return selected


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def numpy_dtype(name: str) -> np.dtype:
    return np.float64 if name == "float64" else np.float32


def build_dense_spectrum(
    task_vectors: Sequence[torch.Tensor],
    weights: Sequence[float],
    k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ExtremalSpectrum:
    in_dim = task_vectors[0].shape[1]
    hessian = torch.zeros((in_dim, in_dim), dtype=dtype, device=device)
    for weight, task in zip(weights, task_vectors):
        task_dev = task.to(device=device, dtype=dtype, non_blocking=True)
        hessian.addmm_(task_dev.T, task_dev, beta=1.0, alpha=float(weight))
        del task_dev
    hessian = 0.5 * (hessian + hessian.T)
    values, vectors = torch.linalg.eigh(hessian)
    values_np = values.detach().cpu().numpy()
    vectors_np = vectors.detach().cpu().numpy()
    del hessian, values, vectors
    if device.type == "cuda":
        torch.cuda.empty_cache()

    k_eff = min(k, max(1, in_dim // 2))
    tail_values = values_np[:k_eff]
    tail_vectors = vectors_np[:, :k_eff]
    top_values = values_np[-k_eff:][::-1].copy()
    top_vectors = vectors_np[:, -k_eff:][:, ::-1].copy()
    return ExtremalSpectrum(
        top_values=top_values,
        top_vectors=top_vectors,
        tail_values=tail_values,
        tail_vectors=tail_vectors,
        solver="dense_eigh",
        full_values=values_np,
    )


def _eigsh_with_partial(
    operator: LinearOperator,
    k: int,
    which: str,
    tol: float,
    maxiter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        values, vectors = eigsh(
            operator,
            k=k,
            which=which,
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=True,
        )
    except ArpackNoConvergence as exc:
        values = exc.eigenvalues
        vectors = exc.eigenvectors
        if values is None or vectors is None or len(values) == 0:
            raise RuntimeError(
                f"ARPACK did not converge for which={which}; no eigenpairs returned"
            ) from exc
        LOGGER.warning(
            "ARPACK partially converged for which=%s: %d/%d eigenpairs",
            which,
            len(values),
            k,
        )
    return np.asarray(values), np.asarray(vectors)


def build_matrix_free_spectrum(
    task_vectors: Sequence[torch.Tensor],
    weights: Sequence[float],
    k: int,
    dtype_name: str,
    tol: float,
    maxiter: int,
) -> ExtremalSpectrum:
    np_dtype = numpy_dtype(dtype_name)
    tasks_np = [task.detach().cpu().numpy().astype(np_dtype, copy=False) for task in task_vectors]
    op = WudiHessianOperator(tasks_np, weights).as_scipy()
    in_dim = tasks_np[0].shape[1]
    k_eff = min(k, max(1, in_dim // 2 - 1))

    top_values, top_vectors = _eigsh_with_partial(op, k_eff, "LA", tol, maxiter)
    tail_values, tail_vectors = _eigsh_with_partial(op, k_eff, "SA", tol, maxiter)

    top_order = np.argsort(top_values)[::-1]
    tail_order = np.argsort(tail_values)
    return ExtremalSpectrum(
        top_values=top_values[top_order],
        top_vectors=top_vectors[:, top_order],
        tail_values=tail_values[tail_order],
        tail_vectors=tail_vectors[:, tail_order],
        solver="matrix_free_eigsh",
        full_values=None,
    )


def solve_spectrum(
    task_vectors: Sequence[torch.Tensor],
    weights: Sequence[float],
    k: int,
    solver: str,
    dense_threshold: int,
    device: torch.device,
    dtype_name: str,
    eig_tol: float,
    eig_maxiter: int,
) -> ExtremalSpectrum:
    in_dim = task_vectors[0].shape[1]
    selected_solver = solver
    if solver == "auto":
        selected_solver = "dense" if in_dim <= dense_threshold else "eigsh"

    if selected_solver == "dense":
        try:
            return build_dense_spectrum(
                task_vectors,
                weights,
                k,
                device=device,
                dtype=torch_dtype(dtype_name),
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if solver != "auto":
                raise
            LOGGER.warning(
                "Dense eigendecomposition failed (%s); falling back to matrix-free eigsh",
                exc,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    return build_matrix_free_spectrum(
        task_vectors,
        weights,
        k,
        dtype_name=dtype_name,
        tol=eig_tol,
        maxiter=eig_maxiter,
    )


def projection_energy(
    matrix: torch.Tensor,
    vectors: np.ndarray,
    chunk_rows: int,
) -> float:
    if vectors.size == 0:
        return 0.0
    v = torch.from_numpy(np.ascontiguousarray(vectors)).to(dtype=torch.float64)
    total = 0.0
    matrix64 = matrix.detach().cpu()
    for start in range(0, matrix64.shape[0], chunk_rows):
        block = matrix64[start : start + chunk_rows].to(dtype=torch.float64)
        projected = block @ v
        total += float(torch.sum(projected * projected).item())
        del block, projected
    return total


def total_energy(matrix: torch.Tensor) -> float:
    return float(torch.sum(matrix.detach().cpu().to(torch.float64) ** 2).item())


def spectral_energy_metrics(
    matrix: torch.Tensor,
    spectrum: ExtremalSpectrum,
    chunk_rows: int,
    eps: float,
) -> Dict[str, float]:
    total = total_energy(matrix)
    top = projection_energy(matrix, spectrum.top_vectors, chunk_rows)
    tail = projection_energy(matrix, spectrum.tail_vectors, chunk_rows)
    middle = max(total - top - tail, 0.0)
    denom = max(total, eps)
    return {
        "fro_energy": total,
        "fro_norm": math.sqrt(max(total, 0.0)),
        "top_energy": top,
        "tail_energy": tail,
        "middle_energy": middle,
        "top_fraction": top / denom,
        "tail_fraction": tail / denom,
        "middle_fraction": middle / denom,
        "tail_to_top": tail / max(top, eps),
    }


def rayleigh_quotient(
    matrix: torch.Tensor,
    task_vectors: Sequence[torch.Tensor],
    weights: Sequence[float],
    eps: float,
) -> float:
    """tr(M H M^T) / ||M||_F^2 without constructing H."""
    denominator = total_energy(matrix)
    if denominator <= eps:
        return 0.0
    m = matrix.detach().cpu().to(torch.float64)
    numerator = 0.0
    for weight, task in zip(weights, task_vectors):
        t = task.detach().cpu().to(torch.float64)
        projected = m @ t.T
        numerator += float(weight) * float(torch.sum(projected * projected).item())
        del t, projected
    return numerator / denominator


def per_task_visibility(
    matrix: torch.Tensor,
    task: torch.Tensor,
    eps: float,
) -> float:
    """||M T^T||_F^2 / (||M||_F^2 ||T||_F^2)."""
    m_energy = total_energy(matrix)
    t_energy = total_energy(task)
    if m_energy <= eps or t_energy <= eps:
        return 0.0
    m = matrix.detach().cpu().to(torch.float64)
    t = task.detach().cpu().to(torch.float64)
    projected = m @ t.T
    value = float(torch.sum(projected * projected).item()) / (m_energy * t_energy)
    return value


def randomized_right_subspace(
    matrix: torch.Tensor,
    rank: int,
    oversample: int,
    niter: int,
    device: torch.device,
) -> np.ndarray:
    min_dim = min(matrix.shape)
    if min_dim <= 1:
        return np.empty((matrix.shape[1], 0), dtype=np.float64)
    q = min(rank + oversample, min_dim - 1)
    target_rank = min(rank, q)
    x = matrix.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, _, v = torch.pca_lowrank(x, q=q, center=False, niter=niter)
    result = v[:, :target_rank].detach().cpu().to(torch.float64).numpy()
    del x, v
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def subspace_overlap(v1: np.ndarray, v2: np.ndarray, eps: float) -> float:
    if v1.size == 0 or v2.size == 0:
        return float("nan")
    rank = min(v1.shape[1], v2.shape[1])
    cross = v1[:, :rank].T @ v2[:, :rank]
    return float(np.sum(cross * cross) / max(rank, eps))


def matrix_from_checkpoint(
    reader: CheckpointReader,
    base: torch.Tensor,
    key: str,
) -> torch.Tensor:
    value = reader.get_tensor(key)
    if value.shape != base.shape:
        raise ValueError(
            f"Shape mismatch for {key}: base={tuple(base.shape)}, checkpoint={tuple(value.shape)}"
        )
    return value.to(torch.float32) - base.to(torch.float32)


def sanitize_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def save_spectrum_plot(
    output_path: Path,
    spectrum: ExtremalSpectrum,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if spectrum.full_values is not None:
        values = np.clip(spectrum.full_values, a_min=0.0, a_max=None)
        ax.semilogy(np.arange(len(values)), values + 1e-30)
        ax.set_xlabel("Eigenvalue index (ascending)")
    else:
        tail = np.clip(spectrum.tail_values, a_min=0.0, a_max=None)
        top = np.clip(spectrum.top_values[::-1], a_min=0.0, a_max=None)
        x_tail = np.arange(len(tail))
        x_top = np.arange(len(tail), len(tail) + len(top))
        ax.semilogy(x_tail, tail + 1e-30, marker="o", label="Bottom eigenvalues")
        ax.semilogy(x_top, top + 1e-30, marker="o", label="Top eigenvalues")
        ax.set_xlabel("Extremal eigenvalue samples")
        ax.legend()
    ax.set_ylabel("Eigenvalue of H_l")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_aggregate_plots(
    state_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if state_df.empty:
        return
    grouped = (
        state_df.groupby("vector_name")[["top_fraction", "middle_fraction", "tail_fraction"]]
        .mean()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    grouped.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Mean fraction of update energy")
    ax.set_xlabel("")
    ax.set_title("Energy distribution across WUDI curvature subspaces")
    ax.legend(title="Spectral region")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_spectral_energy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rayleigh = state_df.groupby("vector_name")["rayleigh_quotient"].mean().sort_values()
    rayleigh.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean Rayleigh quotient")
    ax.set_xlabel("")
    ax.set_title("Curvature-weighted concentration of checkpoint movements")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_rayleigh_quotient.png", dpi=180)
    plt.close(fig)


def build_auto_conclusions(state_df: pd.DataFrame) -> Dict[str, object]:
    conclusions: Dict[str, object] = {}
    if state_df.empty:
        return conclusions

    means = state_df.groupby("vector_name").mean(numeric_only=True)

    def metric(name: str, column: str) -> Optional[float]:
        if name not in means.index or column not in means.columns:
            return None
        value = means.loc[name, column]
        return None if pd.isna(value) else float(value)

    early_tail = metric("adam_early_ta_to_15", "tail_fraction")
    late_tail = metric("adam_late_15_to_300", "tail_fraction")
    early_rq = metric("adam_early_ta_to_15", "rayleigh_quotient")
    late_rq = metric("adam_late_15_to_300", "rayleigh_quotient")

    if early_tail is not None and late_tail is not None:
        conclusions["adam_tail_fraction_mean"] = {
            "early": early_tail,
            "late": late_tail,
            "late_over_early": late_tail / max(early_tail, 1e-30),
            "supports_tail_concentration": late_tail > early_tail,
        }
    if early_rq is not None and late_rq is not None:
        conclusions["adam_rayleigh_mean"] = {
            "early": early_rq,
            "late": late_rq,
            "late_over_early": late_rq / max(early_rq, 1e-30),
            "supports_lower_curvature_late_update": late_rq < early_rq,
        }

    cfs_names = [
        "cfs_1e6_minus_1e5",
        "cfs_1e7_minus_1e6",
        "cfs_1e7_minus_1e5",
    ]
    conclusions["cfs_difference_means"] = {
        name: {
            "tail_fraction": metric(name, "tail_fraction"),
            "rayleigh_quotient": metric(name, "rayleigh_quotient"),
            "fro_norm": metric(name, "fro_norm"),
        }
        for name in cfs_names
        if name in means.index
    }
    return conclusions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spectral diagnosis of Adam-WUDI and CFS-WUDI checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", required=True, help="Base pretrained checkpoint")
    parser.add_argument(
        "--expert",
        action="append",
        required=True,
        type=parse_named_path,
        metavar="NAME=PATH",
        help="Task expert checkpoint; repeat for instruction/math/coding",
    )
    parser.add_argument("--adam15", required=True, help="Adam-WUDI iter-15 checkpoint")
    parser.add_argument("--adam300", required=True, help="Adam-WUDI iter-300 checkpoint")
    parser.add_argument("--cfs1e5", required=True, help="CFS-WUDI omega=1e-5 checkpoint")
    parser.add_argument("--cfs1e6", required=True, help="CFS-WUDI omega=1e-6 checkpoint")
    parser.add_argument("--cfs1e7", required=True, help="CFS-WUDI omega=1e-7 checkpoint")
    parser.add_argument(
        "--ta-checkpoint",
        default=None,
        help="Optional TA checkpoint. Otherwise TA task vector is sum of expert task vectors.",
    )
    parser.add_argument("--output-dir", default="wudi_spectral_results")
    parser.add_argument("--include-regex", default=DEFAULT_KEY_REGEX)
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices; omit to analyze all matching layers",
    )
    parser.add_argument(
        "--modules",
        default=None,
        help="Comma-separated modules, e.g. q_proj,o_proj,up_proj,down_proj",
    )
    parser.add_argument("--max-keys", type=int, default=None)
    parser.add_argument("--tail-k", type=int, default=32)
    parser.add_argument(
        "--solver", choices=["auto", "dense", "eigsh"], default="auto"
    )
    parser.add_argument(
        "--dense-threshold",
        type=int,
        default=4096,
        help="Use dense eigh when input dimension is at most this value",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--spectral-dtype", choices=["float32", "float64"], default="float32"
    )
    parser.add_argument("--eig-tol", type=float, default=1e-5)
    parser.add_argument("--eig-maxiter", type=int, default=5000)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--projection-chunk-rows", type=int, default=1024)
    parser.add_argument("--task-subspace-rank", type=int, default=32)
    parser.add_argument("--task-subspace-oversample", type=int, default=8)
    parser.add_argument("--task-subspace-niter", type=int, default=2)
    parser.add_argument(
        "--skip-task-subspace",
        action="store_true",
        help="Skip randomized task principal-subspace analysis",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    output_dir = Path(args.output_dir).expanduser().resolve()
    spectrum_plot_dir = output_dir / "spectra"
    output_dir.mkdir(parents=True, exist_ok=True)
    spectrum_plot_dir.mkdir(parents=True, exist_ok=True)

    expert_entries: List[Tuple[str, Path]] = args.expert
    expert_names = [name for name, _ in expert_entries]
    if len(set(expert_names)) != len(expert_names):
        raise SystemExit("Expert names must be unique")

    LOGGER.info("Opening checkpoints")
    base_reader = CheckpointReader(args.base)
    expert_readers = {
        name: CheckpointReader(path) for name, path in expert_entries
    }
    merged_readers = {
        "adam15": CheckpointReader(args.adam15),
        "adam300": CheckpointReader(args.adam300),
        "cfs1e5": CheckpointReader(args.cfs1e5),
        "cfs1e6": CheckpointReader(args.cfs1e6),
        "cfs1e7": CheckpointReader(args.cfs1e7),
    }
    ta_reader = CheckpointReader(args.ta_checkpoint) if args.ta_checkpoint else None

    readers_for_key_check: List[CheckpointReader] = [
        *expert_readers.values(),
        *merged_readers.values(),
    ]
    if ta_reader is not None:
        readers_for_key_check.append(ta_reader)

    layers = parse_csv_ints(args.layers)
    modules = parse_csv_strings(args.modules)
    keys = select_keys(
        base_reader,
        readers_for_key_check,
        include_regex=args.include_regex,
        layers=layers,
        modules=modules,
        max_keys=args.max_keys,
    )
    if not keys:
        raise SystemExit("No matching 2-D weight keys were found")

    LOGGER.info("Selected %d weight matrices", len(keys))

    layer_rows: List[Dict[str, object]] = []
    task_rows: List[Dict[str, object]] = []
    state_rows: List[Dict[str, object]] = []
    visibility_rows: List[Dict[str, object]] = []
    overlap_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, str]] = []

    for key_index, key in enumerate(keys, start=1):
        LOGGER.info("[%d/%d] %s", key_index, len(keys), key)
        layer, module = key_metadata(key)
        try:
            base = base_reader.get_tensor(key).to(torch.float32)
            if base.ndim != 2:
                LOGGER.warning("Skipping non-2D tensor: %s", key)
                continue

            task_vectors: List[torch.Tensor] = []
            for name in expert_names:
                task_vectors.append(matrix_from_checkpoint(expert_readers[name], base, key))

            norms_sq = [total_energy(task) for task in task_vectors]
            if any(value <= args.eps for value in norms_sq):
                raise ValueError(f"Near-zero task vector norm for {key}: {norms_sq}")
            weights = [1.0 / (value + args.eps) for value in norms_sq]

            spectrum = solve_spectrum(
                task_vectors,
                weights,
                k=args.tail_k,
                solver=args.solver,
                dense_threshold=args.dense_threshold,
                device=device,
                dtype_name=args.spectral_dtype,
                eig_tol=args.eig_tol,
                eig_maxiter=args.eig_maxiter,
            )

            top_max = float(np.max(spectrum.top_values))
            tail_min = float(np.min(spectrum.tail_values))
            smallest_positive = float(
                np.min(spectrum.tail_values[spectrum.tail_values > args.eps])
            ) if np.any(spectrum.tail_values > args.eps) else float("nan")
            condition_estimate = (
                top_max / smallest_positive
                if math.isfinite(smallest_positive) and smallest_positive > 0
                else float("inf")
            )
            trace_h = float(len(task_vectors))  # exact under norm weighting, up to eps.
            stable_rank = trace_h / max(top_max, args.eps)

            layer_rows.append(
                {
                    "key": key,
                    "layer": layer,
                    "module": module,
                    "out_dim": base.shape[0],
                    "in_dim": base.shape[1],
                    "solver": spectrum.solver,
                    "num_top_eigenvectors": spectrum.top_vectors.shape[1],
                    "num_tail_eigenvectors": spectrum.tail_vectors.shape[1],
                    "lambda_max": top_max,
                    "lambda_tail_min": tail_min,
                    "lambda_smallest_positive_in_sample": smallest_positive,
                    "condition_estimate": condition_estimate,
                    "trace_h_expected": trace_h,
                    "stable_rank_estimate": stable_rank,
                }
            )

            save_spectrum_plot(
                spectrum_plot_dir / f"{sanitize_key(key)}.png",
                spectrum,
                title=f"WUDI curvature spectrum: layer={layer}, module={module}",
            )

            task_subspaces: Dict[str, np.ndarray] = {}
            for task_name, task in zip(expert_names, task_vectors):
                metrics = spectral_energy_metrics(
                    task, spectrum, args.projection_chunk_rows, args.eps
                )
                metrics["rayleigh_quotient"] = rayleigh_quotient(
                    task, task_vectors, weights, args.eps
                )
                task_rows.append(
                    {
                        "key": key,
                        "layer": layer,
                        "module": module,
                        "task": task_name,
                        **metrics,
                    }
                )

                if not args.skip_task_subspace:
                    task_subspaces[task_name] = randomized_right_subspace(
                        task,
                        rank=args.task_subspace_rank,
                        oversample=args.task_subspace_oversample,
                        niter=args.task_subspace_niter,
                        device=device,
                    )
                    v_task = task_subspaces[task_name]
                    overlap_rows.append(
                        {
                            "key": key,
                            "layer": layer,
                            "module": module,
                            "task_a": task_name,
                            "task_b": "GLOBAL_TOP_H",
                            "subspace_overlap": subspace_overlap(
                                v_task, spectrum.top_vectors, args.eps
                            ),
                        }
                    )
                    overlap_rows.append(
                        {
                            "key": key,
                            "layer": layer,
                            "module": module,
                            "task_a": task_name,
                            "task_b": "GLOBAL_TAIL_H",
                            "subspace_overlap": subspace_overlap(
                                v_task, spectrum.tail_vectors, args.eps
                            ),
                        }
                    )

            if task_subspaces:
                for i, name_a in enumerate(expert_names):
                    for name_b in expert_names[i + 1 :]:
                        overlap_rows.append(
                            {
                                "key": key,
                                "layer": layer,
                                "module": module,
                                "task_a": name_a,
                                "task_b": name_b,
                                "subspace_overlap": subspace_overlap(
                                    task_subspaces[name_a],
                                    task_subspaces[name_b],
                                    args.eps,
                                ),
                            }
                        )

            ta = (
                matrix_from_checkpoint(ta_reader, base, key)
                if ta_reader is not None
                else torch.stack(task_vectors, dim=0).sum(dim=0)
            )
            adam15 = matrix_from_checkpoint(merged_readers["adam15"], base, key)
            adam300 = matrix_from_checkpoint(merged_readers["adam300"], base, key)
            cfs1e5 = matrix_from_checkpoint(merged_readers["cfs1e5"], base, key)
            cfs1e6 = matrix_from_checkpoint(merged_readers["cfs1e6"], base, key)
            cfs1e7 = matrix_from_checkpoint(merged_readers["cfs1e7"], base, key)

            matrices: Dict[str, torch.Tensor] = {
                "ta": ta,
                "adam15": adam15,
                "adam300": adam300,
                "cfs_1e5": cfs1e5,
                "cfs_1e6": cfs1e6,
                "cfs_1e7": cfs1e7,
                "adam_early_ta_to_15": adam15 - ta,
                "adam_late_15_to_300": adam300 - adam15,
                "adam_total_ta_to_300": adam300 - ta,
                "cfs_1e5_minus_ta": cfs1e5 - ta,
                "cfs_1e6_minus_1e5": cfs1e6 - cfs1e5,
                "cfs_1e7_minus_1e6": cfs1e7 - cfs1e6,
                "cfs_1e7_minus_1e5": cfs1e7 - cfs1e5,
            }

            for vector_name, matrix in matrices.items():
                metrics = spectral_energy_metrics(
                    matrix, spectrum, args.projection_chunk_rows, args.eps
                )
                metrics["rayleigh_quotient"] = rayleigh_quotient(
                    matrix, task_vectors, weights, args.eps
                )
                state_rows.append(
                    {
                        "key": key,
                        "layer": layer,
                        "module": module,
                        "vector_name": vector_name,
                        **metrics,
                    }
                )

                for task_name, task in zip(expert_names, task_vectors):
                    visibility_rows.append(
                        {
                            "key": key,
                            "layer": layer,
                            "module": module,
                            "vector_name": vector_name,
                            "task": task_name,
                            "normalized_task_visibility": per_task_visibility(
                                matrix, task, args.eps
                            ),
                        }
                    )

            del (
                base,
                task_vectors,
                ta,
                adam15,
                adam300,
                cfs1e5,
                cfs1e6,
                cfs1e7,
                matrices,
                spectrum,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:  # Continue to preserve results from other layers.
            LOGGER.exception("Failed to analyze %s", key)
            failed_rows.append({"key": key, "error": repr(exc)})

    layer_df = pd.DataFrame(layer_rows)
    task_df = pd.DataFrame(task_rows)
    state_df = pd.DataFrame(state_rows)
    visibility_df = pd.DataFrame(visibility_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    failed_df = pd.DataFrame(failed_rows)

    layer_df.to_csv(output_dir / "per_layer_spectrum.csv", index=False)
    task_df.to_csv(output_dir / "per_task_spectral_energy.csv", index=False)
    state_df.to_csv(output_dir / "per_state_and_update_spectral_energy.csv", index=False)
    visibility_df.to_csv(output_dir / "per_vector_task_visibility.csv", index=False)
    overlap_df.to_csv(output_dir / "task_subspace_overlap.csv", index=False)
    failed_df.to_csv(output_dir / "failed_keys.csv", index=False)

    if not state_df.empty:
        aggregate_state = (
            state_df.groupby(["vector_name", "module"], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        aggregate_state.to_csv(output_dir / "aggregate_by_vector_and_module.csv", index=False)
    if not task_df.empty:
        aggregate_task = (
            task_df.groupby(["task", "module"], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        aggregate_task.to_csv(output_dir / "aggregate_by_task_and_module.csv", index=False)
    if not visibility_df.empty:
        aggregate_visibility = (
            visibility_df.groupby(["vector_name", "task", "module"], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        aggregate_visibility.to_csv(
            output_dir / "aggregate_task_visibility.csv", index=False
        )

    save_aggregate_plots(state_df, output_dir)
    conclusions = build_auto_conclusions(state_df)

    metadata = {
        "base": str(Path(args.base).expanduser()),
        "experts": {name: str(path) for name, path in expert_entries},
        "adam15": str(Path(args.adam15).expanduser()),
        "adam300": str(Path(args.adam300).expanduser()),
        "cfs1e5": str(Path(args.cfs1e5).expanduser()),
        "cfs1e6": str(Path(args.cfs1e6).expanduser()),
        "cfs1e7": str(Path(args.cfs1e7).expanduser()),
        "ta_checkpoint": args.ta_checkpoint,
        "selected_keys": keys,
        "completed_keys": len(layer_rows),
        "failed_keys": len(failed_rows),
        "arguments": vars(args),
        "automatic_diagnostic_summary": conclusions,
        "interpretation_rules": {
            "late_adam_tail_hypothesis": (
                "Supported when adam_late_15_to_300 has higher tail_fraction and/or "
                "lower Rayleigh quotient than adam_early_ta_to_15 across many layers."
            ),
            "weak_cfs_regularization_hypothesis": (
                "Supported when cfs_1e7_minus_1e5 or cfs_1e7_minus_1e6 has large "
                "Frobenius norm but low Rayleigh quotient / high tail fraction."
            ),
            "domain_heterogeneity_hypothesis": (
                "Supported when task tail fractions, global-top overlaps, or pairwise "
                "principal-subspace overlaps differ systematically among tasks."
            ),
        },
    }
    (output_dir / "run_metadata_and_summary.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    for reader in [
        base_reader,
        *expert_readers.values(),
        *merged_readers.values(),
        *([ta_reader] if ta_reader is not None else []),
    ]:
        reader.clear_cache()

    LOGGER.info("Finished. Results written to %s", output_dir)
    if failed_rows:
        LOGGER.warning("%d keys failed; inspect failed_keys.csv", len(failed_rows))
    return 0 if layer_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
