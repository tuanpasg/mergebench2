# merging/merging_methods/ISO.py
import torch
from pathlib import Path
from typing import Dict, List, Optional
from merging_methods.merger import Merger
from merging_methods.utils import get_task_vector_dict_iso


def _running_mean_update(dst: Optional[torch.Tensor], x: torch.Tensor, i: int) -> torch.Tensor:
    # numerically stable online mean
    if dst is None:
        return x.clone()
    return dst + (x - dst) / (i + 1)


def _svd_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    # torch.linalg.svd does not support bf16/fp16 on CPU and can be backend-limited.
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _svd_safe(x: torch.Tensor):
    x_work = x.to(dtype=_svd_compute_dtype(x.dtype)).contiguous()
    x_work = torch.nan_to_num(x_work, nan=0.0, posinf=1e4, neginf=-1e4)

    def _run_svd(t: torch.Tensor):
        return torch.linalg.svd(t, full_matrices=False)

    try:
        return _run_svd(x_work)
    except torch._C._LinAlgError:
        pass

    x64 = x_work.to(dtype=torch.float64)
    try:
        return _run_svd(x64)
    except torch._C._LinAlgError:
        pass

    # Last in-device retry: tiny diagonal jitter to break pathological spectra.
    m, n = x64.shape
    p = min(m, n)
    # scale epsilon by matrix norm so it remains tiny but effective
    eps = torch.finfo(torch.float64).eps * max(m, n) * (x64.norm() + 1.0)
    x_jitter = x64.clone()
    x_jitter[:p, :p] = x_jitter[:p, :p] + eps * torch.eye(p, device=x64.device, dtype=x64.dtype)
    try:
        return _run_svd(x_jitter)
    except torch._C._LinAlgError:
        # Final fallback: run on CPU float64 and move factors back.
        u_cpu, s_cpu, vh_cpu = _run_svd(x_jitter.cpu())
        return u_cpu.to(x.device), s_cpu.to(x.device), vh_cpu.to(x.device)


def iso_c_merge_task_vectors(
    task_vectors: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Iso-C on task vectors:
      - compute TA matrix per key via sum (or mean then *T)
      - SVD on 2D tensors
      - replace singular values by their mean (isotropic)
      - non-2D => average

    This matches iso_c() in provided reference code. :contentReference[oaicite:4]{index=4}
    """
    assert len(task_vectors) > 0, "Need at least one task vector"
    keys = list(task_vectors[0].keys())
    T = len(task_vectors)

    out: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for k in keys:
            # avg across tasks first
            avg = None
            for i, tv in enumerate(task_vectors):
                avg = _running_mean_update(avg, tv[k].to(device), i)

            w = avg  # mean
            shape = w.shape

            is_2d = (w.ndim == 2) and ("text_projection" not in k)
            if not is_2d:
                out[k] = w
                continue

            # original code: new_vector[key] *= len(tvs) (=> TA sum)
            w = w * T

            # full_matrices=False matches reference
            out_dtype = w.dtype
            U, S, Vh = _svd_safe(w)
            S_iso = torch.ones_like(S) * S.mean()

            out[k] = (U @ torch.diag(S_iso) @ Vh).to(dtype=out_dtype)

    return out


def iso_cts_merge_task_vectors(
    task_vectors: List[Dict[str, torch.Tensor]],
    device: torch.device,
    common_space_fraction: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """
    Iso-CTS on task vectors:
      - build common subspace from sum of task vectors
      - remove common subspace from each task (projection)
      - take equal-sized task-specific subspaces
      - concatenate (task-specific + common), orthogonalize
      - isotropic scaling of singular values (mean)
      - non-2D => average

    This matches iso_cts() in provided reference code. :contentReference[oaicite:5]{index=5}
    """
    assert len(task_vectors) > 0, "Need at least one task vector"
    keys = list(task_vectors[0].keys())
    T = len(task_vectors)

    out: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for k in keys:
            print(k)
            shape0 = task_vectors[0][k].shape
            is_2d = (len(shape0) == 2) and ("text_projection" not in k)

            if not is_2d:
                print("is averaged due to 1D shape")
                avg = None
                for i, tv in enumerate(task_vectors):
                    avg = _running_mean_update(avg, tv[k].to(device), i)
                out[k] = avg
                continue

            # common space from sum
            combined_w = None
            out_dtype = task_vectors[0][k].dtype
            compute_dtype = _svd_compute_dtype(out_dtype)
            for tv in task_vectors:
                x = tv[k].to(device=device, dtype=compute_dtype)
                combined_w = x if combined_w is None else (combined_w + x)

            m, n = combined_w.shape
            r = min(m, n)

            # Determine common_space_index_s like reference
            common_space_index_s = int(r * common_space_fraction)
            task_specific_total = round((r - common_space_index_s) / T) * T
            common_space_index_s = r - task_specific_total

            # SVD on common
            U, S, Vh = _svd_safe(combined_w)
            common_u = U[:, :common_space_index_s]
            common_s = S[:common_space_index_s]
            common_v = Vh[:common_space_index_s, :]

            # task-specific dims per task
            n_dims_per_task = int((r - common_space_index_s) / T)

            combined_u = None
            combined_s = None
            combined_v = None

            for i, tv in enumerate(task_vectors):
                w = tv[k].to(device=device, dtype=compute_dtype)

                # remove common subspace (left projection), matches reference
                w_ts = w - common_u @ (common_u.T @ w)

                u_ts, s_ts, v_ts = _svd_safe(w_ts)

                if i == 0:
                    combined_u = torch.zeros_like(u_ts, device=device)
                    combined_s = torch.zeros_like(s_ts, device=device)
                    combined_v = torch.zeros_like(v_ts, device=device)

                a, b = i * n_dims_per_task, (i + 1) * n_dims_per_task
                if n_dims_per_task > 0:
                    combined_u[:, a:b] = u_ts[:, :n_dims_per_task]
                    combined_s[a:b] = s_ts[:n_dims_per_task]
                    combined_v[a:b, :] = v_ts[:n_dims_per_task, :]

            # append common subspace to the tail (same layout as reference)
            start = T * n_dims_per_task
            end = start + common_space_index_s
            if common_space_index_s > 0:
                combined_u[:, start:end] = common_u
                combined_s[start:end] = common_s
                combined_v[start:end, :] = common_v

            # Orthogonalize U and V via SVD-based whitening (reference behavior)
           # Orthogonalize U and V via SVD-based whitening
            u_u, _, v_u = _svd_safe(combined_u)
            u_v, _, v_v = _svd_safe(combined_v)
            combined_u = u_u @ v_u
            combined_v = u_v @ v_v
            
            # Force all factors to same compute dtype before reconstruction
            combined_u = combined_u.to(dtype=compute_dtype)
            combined_v = combined_v.to(dtype=compute_dtype)
            combined_s = combined_s.to(dtype=compute_dtype)
            
            # isotropic scaling
            combined_s = torch.ones_like(combined_s) * combined_s.mean()
            
            out[k] = (combined_u @ torch.diag(combined_s) @ combined_v).to(dtype=out_dtype)

    return out


# -------------------------
# MergeBench wrapper classes
# -------------------------

def _apply_delta_to_state(base_state: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state = {k: v.clone() for k, v in base_state.items()}
    for name, dv in delta.items():
        if name in state:
            state[name] = state[name].to("cpu") + dv.to("cpu")
    return state


def _scale_delta(delta: Dict[str, torch.Tensor], scaling_coef: float) -> Dict[str, torch.Tensor]:
    return {k: v * scaling_coef for k, v in delta.items()}


def _format_scaling_coef(scaling_coef: float) -> str:
    return f"{scaling_coef:.4f}".rstrip("0").rstrip(".")


def _build_save_path_with_scaling(save_path: str, scaling_coef: float) -> str:
    coef_str = _format_scaling_coef(scaling_coef)
    return f"{save_path}_scaling_coef_{coef_str}"


def _get_validate_scaling_coefs(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError(f"validate_scaling_step must be > 0, got {step}")
    if stop <= start:
        raise ValueError(f"validate_scaling_stop must be > validate_scaling_start, got start={start}, stop={stop}")
    coefs = torch.arange(start, stop, step, dtype=torch.float64).tolist()
    if len(coefs) == 0:
        raise ValueError(f"No scaling coefficients generated from start={start}, stop={stop}, step={step}")
    return [float(x) for x in coefs]


class IsoC(Merger):
    """
    MergeBench method name (CLI): IsoC
    """
    method_name = "iso_c"

    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)

    def merge(self, **kwargs):
        device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        exclude_embed = bool(kwargs.get("exclude_embed", True))

        task_vectors = [
            get_task_vector_dict_iso(ft_model, self.base_model, exclude_embed=exclude_embed)
            for ft_model in self.ft_ckpts
        ]
        merged_tv = iso_c_merge_task_vectors(task_vectors, device=device)

        base_state = {k: v.clone() for k, v in self.base_model.state_dict().items()}
        state = _apply_delta_to_state(base_state, merged_tv)
        self.base_model.load_state_dict(state)

        self.base_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)


class IsoCTS(Merger):
    """
    MergeBench method name (CLI): IsoCTS
    """
    method_name = "iso_cts"

    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)

    def merge(self, **kwargs):
        device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        scaling_coef = float(kwargs.get("scaling_coef", 1.0))
        frac = float(kwargs.get("common_space_fraction", 0.8))
        exclude_embed = bool(kwargs.get("exclude_embed", True))
        validate_gen_flag = bool(kwargs.get("validate_gen_flag", False))
        validate_scaling_start = float(kwargs.get("validate_scaling_start", 0.0))
        validate_scaling_stop = float(kwargs.get("validate_scaling_stop", 1.0))
        validate_scaling_step = float(kwargs.get("validate_scaling_step", 0.2))

        task_vectors = [
            get_task_vector_dict_iso(ft_model, self.base_model, exclude_embed=exclude_embed)
            for ft_model in self.ft_ckpts
        ]
        merged_tv = iso_cts_merge_task_vectors(
            task_vectors,
            device=device,
            common_space_fraction=frac,
        )

        base_state = {k: v.clone() for k, v in self.base_model.state_dict().items()}
        scaling_coefs = [scaling_coef]
        if validate_gen_flag:
            scaling_coefs = _get_validate_scaling_coefs(
                start=validate_scaling_start,
                stop=validate_scaling_stop,
                step=validate_scaling_step,
            )

        for coef in scaling_coefs:
            scaled_merged_tv = _scale_delta(merged_tv, coef)
            state = _apply_delta_to_state(base_state, scaled_merged_tv)
            self.base_model.load_state_dict(state)

            target_save_path = _build_save_path_with_scaling(self.save_path, coef)
            Path(target_save_path).mkdir(parents=True, exist_ok=True)
            self.base_model.save_pretrained(target_save_path)
            self.tokenizer.save_pretrained(target_save_path)
