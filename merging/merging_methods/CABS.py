import os
import torch
from typing import Dict, List, Tuple
from merging_methods.utils import get_task_vector_dict
from merging_methods.merger import Merger


TensorDict = Dict[str, torch.Tensor]


def _nm_prune_with_availability(
    t: torch.Tensor,
    avail: torch.Tensor,
    n: int,
    m: int,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    n:m pruning with an availability mask (0/1). We select top-n by |t| within each group of m,
    but only among positions where avail==1. If a group has < n available entries, we still run,
    but warn (fallback case). In that case, some selected indices may correspond to unavailable
    positions, resulting in fewer than n effective non-zeros in that group.

    Returns:
      pruned_t: t * mask
      mask: 0/1 float mask
      needs_fallback_warning: True if any group has < n available positions
    """
    if t.numel() == 0:
        mask = torch.zeros_like(t, dtype=torch.float32)
        return t, mask, False

    # Flatten
    orig_shape = t.shape
    flat = t.reshape(-1)
    flat_av = avail.reshape(-1).to(dtype=torch.bool)

    # Pad to multiple of m
    L = flat.numel()
    pad = (m - (L % m)) % m
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)], dim=0)
        flat_av = torch.cat([flat_av, torch.zeros(pad, device=flat.device, dtype=torch.bool)], dim=0)

    flat2 = flat.view(-1, m)
    av2 = flat_av.view(-1, m)

    # Fallback condition check: any block has < n available entries
    avail_counts = av2.sum(dim=1)
    needs_warn = bool((avail_counts < n).any().item())

    # Score: abs(t), but disallow unavailable positions by setting score=-inf
    scores = flat2.abs()
    scores = scores.masked_fill(~av2, float("-inf"))

    # topk: even if fewer than n valid, topk returns something (possibly -inf indices)
    topk_idx = scores.topk(k=n, dim=1, largest=True).indices

    mask2 = torch.zeros_like(flat2, dtype=torch.float32)
    mask2.scatter_(1, topk_idx, 1.0)

    # Make sure unavailable positions are not marked as kept
    mask2 = mask2 * av2.to(dtype=torch.float32)

    pruned2 = flat2 * mask2.to(dtype=flat2.dtype)

    pruned = pruned2.reshape(-1)
    mask = mask2.reshape(-1)

    # Unpad
    if pad:
        pruned = pruned[:-pad]
        mask = mask[:-pad]

    return pruned.reshape(orig_shape), mask.reshape(orig_shape), needs_warn


def _sequential_cabs_nm_prune(
    task_vectors: List[TensorDict],
    n: int,
    m: int,
    warn_min_overlap_fallback: bool = True,
) -> Tuple[List[TensorDict], bool, List[str]]:
    """
    Sequential CA-style pruning with BS = n:m pruning.
    Fixed order: task_vectors[0], task_vectors[1], task_vectors[2] (or k vectors).
    For each parameter tensor:
      avail_mask starts as all-ones
      prune tv_i with avail_mask
      update avail_mask <- avail_mask * (1 - mask_i)

    Returns:
      pruned_tvs: list of pruned task vectors (same structure)
      any_warn: True if any layer/block triggered fallback warning
      warn_msgs: list of warning messages (deduplicated-ish)
    """
    k = len(task_vectors)
    pruned_tvs: List[TensorDict] = [dict() for _ in range(k)]
    any_warn = False
    warn_msgs: List[str] = []

    # Iterate per-parameter name; assume all task_vectors share same keys
    keys = list(task_vectors[0].keys())

    for name in keys:
        # availability mask is per-parameter tensor
        base_shape = task_vectors[0][name].shape
        device = task_vectors[0][name].device
        avail = torch.ones(base_shape, device=device, dtype=torch.bool)

        for i in range(k):
            t = task_vectors[i][name]

            # Ensure avail on same device/shape
            if t.shape != base_shape:
                raise ValueError(f"Shape mismatch for param {name}: tv0={base_shape}, tv{i}={t.shape}")

            pruned_t, mask, needs_warn = _nm_prune_with_availability(t, avail, n=n, m=m)
            pruned_tvs[i][name] = pruned_t

            # Update availability for next vectors (CA masking)
            avail = avail & (mask == 0)

            if warn_min_overlap_fallback and needs_warn:
                any_warn = True
                warn_msgs.append(
                    f"[CABS-NM WARNING] Param '{name}': some n:m blocks had < n available weights "
                    f"after CA masking (would require Algorithm-2 fallback to reintroduce overlap)."
                )

    # Optionally compact warnings
    if any_warn:
        # Keep only first ~30 unique warnings to avoid log spam
        seen = set()
        compact = []
        for msg in warn_msgs:
            if msg not in seen:
                seen.add(msg)
                compact.append(msg)
            if len(compact) >= 30:
                compact.append("[CABS-NM WARNING] (more layers triggered fallback check; truncated)")
                break
        warn_msgs = compact

    return pruned_tvs, any_warn, warn_msgs

def _apply_delta_to_state(base_state: Dict[str, torch.Tensor], delta: TensorDict) -> Dict[str, torch.Tensor]:
    state = {k: v.clone() for k, v in base_state.items()}
    for name, dv in delta.items():
        if name in state:
            state[name] = state[name].to(dv.device) + dv
    return state

class CABS(Merger):
    """
    CRITICAL NOTICE: This merger exclude embed and lm_head layers 
    MergeBench-style merger:
      - base_model: HF model name/path
      - ft_models: list of HF model names/paths (expects exactly 3 for this script)
      - save_path: output directory
    """

    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)

    def merge(self, **kwargs):
        """
        Expected kwargs:
          - n (int): keep n per block
          - m (int): block size m
          - scaling_coef: float OR list/tuple of 3 floats
          - save_pruned_models (bool): default True
          - pruned_subdir (str): default "pruned_models"
          - warn_min_overlap_fallback (bool): default True
        """
        if len(self.ft_ckpts) != 3:
            raise ValueError(f"CABSNM expects exactly 3 fine-tuned checkpoints, got {len(self.ft_ckpts)}")

        n = int(kwargs.get("n", 64))
        m = int(kwargs.get("m", 256))
        warn_min_overlap_fallback = bool(kwargs.get("warn_min_overlap_fallback", True))

        scaling = kwargs.get("scaling_coef", 1.0)
        if isinstance(scaling, (list, tuple)):
            if len(scaling) != 3:
                raise ValueError(f"scaling_coef list must have length 3, got {len(scaling)}")
            scaling_coefs = [float(x) for x in scaling]
        else:
            scaling_coefs = [float(scaling)] * 3

        save_pruned_models = bool(kwargs.get("save_pruned_models", True))
        pruned_subdir = kwargs.get("pruned_subdir", "pruned_models")

        # 1) Extract task vectors (ft - base), MergeBench-style
        task_vectors: List[TensorDict] = [
            get_task_vector_dict(ft_model, self.base_model) for ft_model in self.ft_ckpts
        ]
        base_state = {k: v.clone() for k, v in self.base_model.state_dict().items()}

        # 2) Sequential CA + BS (n:m) pruning in fixed order
        pruned_tvs, any_warn, warn_msgs = _sequential_cabs_nm_prune(
            task_vectors,
            n=n,
            m=m,
            warn_min_overlap_fallback=warn_min_overlap_fallback,
        )
        if any_warn:
            for msg in warn_msgs:
                print(msg)

        # 3) Save pruned per-task models (base + pruned_tv_i) for later λ tuning
        if save_pruned_models:
            root = os.path.join(self.save_path, pruned_subdir)
            os.makedirs(root, exist_ok=True)
            for i, pruned_tv in enumerate(pruned_tvs):
                out_dir = os.path.join(root, f"task_{i+1}")
                os.makedirs(out_dir, exist_ok=True)

                state = _apply_delta_to_state(base_state, pruned_tv)
                self.base_model.load_state_dict(state)
                self.base_model.save_pretrained(out_dir)
                self.tokenizer.save_pretrained(out_dir)

        # 4) Merge pruned vectors (optionally with per-task scaling)
        merged_tv: TensorDict = {}
        keys = pruned_tvs[0].keys()
        for k in keys:
            merged_tv[k] = (
                scaling_coefs[0] * pruned_tvs[0][k]
                + scaling_coefs[1] * pruned_tvs[1][k]
                + scaling_coefs[2] * pruned_tvs[2][k]
            )

        # 5) Merge the merged-pruned task vector to the base model
        os.makedirs(self.save_path, exist_ok=True)
        
        # state = _apply_delta_to_state(base_state, merged_tv)
        # self.base_model.load_state_dict(state)
        for name, dv in merged_tv.items():
            if name in base_state:
                base_state[name] = base_state[name].to(dv.device) + dv
        self.base_model.load_state_dict(base_state)

        # 6) Save merged model
        self.base_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
