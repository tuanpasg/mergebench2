# merging/merging_methods/ISO.py
import torch
from typing import Dict, List, Optional


def _running_mean_update(dst: Optional[torch.Tensor], x: torch.Tensor, i: int) -> torch.Tensor:
    # numerically stable online mean
    if dst is None:
        return x.clone()
    return dst + (x - dst) / (i + 1)


def iso_c_merge_task_vectors(
    task_vectors: List[Dict[str, torch.Tensor]],
    device: torch.device,
    skip_if: Optional[callable] = None,
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
            if skip_if is not None and skip_if(k):
                # still merge by simple avg to keep output consistent
                avg = None
                for i, tv in enumerate(task_vectors):
                    avg = _running_mean_update(avg, tv[k].to(device), i)
                out[k] = avg
                continue

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
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            S_iso = torch.ones_like(S) * S.mean()

            out[k] = (U @ torch.diag(S_iso) @ Vh)

    return out


def iso_cts_merge_task_vectors(
    task_vectors: List[Dict[str, torch.Tensor]],
    device: torch.device,
    common_space_fraction: float = 0.8,
    skip_if: Optional[callable] = None,
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
            if skip_if is not None and skip_if(k):
                avg = None
                for i, tv in enumerate(task_vectors):
                    avg = _running_mean_update(avg, tv[k].to(device), i)
                out[k] = avg
                continue

            shape0 = task_vectors[0][k].shape
            is_2d = (len(shape0) == 2) and ("text_projection" not in k)

            if not is_2d:
                avg = None
                for i, tv in enumerate(task_vectors):
                    avg = _running_mean_update(avg, tv[k].to(device), i)
                out[k] = avg
                continue

            # common space from sum
            combined_w = None
            for tv in task_vectors:
                x = tv[k].to(device)
                combined_w = x if combined_w is None else (combined_w + x)

            m, n = combined_w.shape
            r = min(m, n)

            # Determine common_space_index_s like reference
            common_space_index_s = int(r * common_space_fraction)
            task_specific_total = round((r - common_space_index_s) / T) * T
            common_space_index_s = r - task_specific_total

            # SVD on common
            U, S, Vh = torch.linalg.svd(combined_w, full_matrices=False)
            common_u = U[:, :common_space_index_s]
            common_s = S[:common_space_index_s]
            common_v = Vh[:common_space_index_s, :]

            # task-specific dims per task
            n_dims_per_task = int((r - common_space_index_s) / T)

            combined_u = None
            combined_s = None
            combined_v = None

            for i, tv in enumerate(task_vectors):
                w = tv[k].to(device)

                # remove common subspace (left projection), matches reference
                w_ts = w - common_u @ (common_u.T @ w)

                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

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
            u_u, _, v_u = torch.linalg.svd(combined_u, full_matrices=False)
            u_v, _, v_v = torch.linalg.svd(combined_v, full_matrices=False)
            combined_u = u_u @ v_u
            combined_v = u_v @ v_v

            # isotropic scaling
            combined_s = torch.ones_like(combined_s) * combined_s.mean()

            out[k] = combined_u @ torch.diag(combined_s) @ combined_v

    return out


# -------------------------
# MergeBench wrapper classes
# -------------------------

class IsoC:
    """
    MergeBench method name: iso_c
    Expects task vectors already computed (theta_t - theta_0).
    """
    method_name = "iso_c"

    def merge(self, task_vectors: List[Dict[str, torch.Tensor]], config) -> Dict[str, torch.Tensor]:
        device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        skip_if = getattr(config, "skip_if", None)
        return iso_c_merge_task_vectors(task_vectors, device=device, skip_if=skip_if)


class IsoCTS:
    """
    MergeBench method name: iso_cts
    """
    method_name = "iso_cts"

    def merge(self, task_vectors: List[Dict[str, torch.Tensor]], config) -> Dict[str, torch.Tensor]:
        device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        frac = float(getattr(getattr(config, "method", config), "common_space_fraction", 0.8))
        skip_if = getattr(config, "skip_if", None)
        return iso_cts_merge_task_vectors(
            task_vectors,
            device=device,
            common_space_fraction=frac,
            skip_if=skip_if,
        )
