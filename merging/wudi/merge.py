import csv
import os
import torch
import tqdm
import re
import utils
from openpyxl import Workbook
from param import param

class MergingMethod:
    WUDI_LOSS_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 300]

    @utils.args_inspector
    def __init__(
        self, 
        models_to_merge, 
        models_name,
    ):
        self.models_name = {n:i for i,n in enumerate(models_name)}
        # dict(zip(models_name, range(0, N)))
        self.models_to_merge = models_to_merge

    def get_model(self, model_name):
        return self.models_to_merge[self.models_name[model_name]]

    def _optimize_wudi_vector(
        self,
        vecs_cpu: list,
        dev: torch.device,
        iter_num: int,
        lr: float,
        weight_decay: float,
        eps: float,
        warm_start: bool = True,
        cfs_ridge: float = 1e-5,
        loss_steps: list = None,
    ):
        if cfs_ridge < 0:
            raise ValueError(f"cfs_ridge must be >= 0, got {cfs_ridge}")

        # stack on device: (n_task, out, in)
        vectors = torch.stack([v.to(dev) for v in vecs_cpu], dim=0)

        orig_dtype = vectors.dtype
        vectors_f = vectors

        n_task, out_dim, in_dim = vectors_f.shape

        # l2 norms per task vector: (n_task,)
        l2_norms = (
            torch.square(torch.norm(vectors_f.reshape(n_task, -1), p=2, dim=-1))
            + eps
        )

        weights = 1.0 / l2_norms  # (n_task,)

        cold_init = vectors_f.sum(dim=0)

        def _wudi_objective(candidate: torch.Tensor) -> torch.Tensor:
            disturbing = candidate.unsqueeze(0) - vectors_f

            # inner: (n_task, out, out)
            inner = torch.matmul(
                disturbing,
                vectors_f.transpose(1, 2),
            )

            return torch.sum(
                (inner * inner) / l2_norms.view(-1, 1, 1)
            )

        init_delta = cold_init
        if warm_start:
            A = torch.zeros((in_dim, in_dim), device=dev, dtype=torch.float32)
            B = torch.zeros((out_dim, in_dim), device=dev, dtype=torch.float32)

            for i in range(n_task):
                v = vectors_f[i].float()
                w = weights[i].float()

                vt_v = v.T @ v
                A += w * vt_v
                B += w * (v @ vt_v)

            A = A + cfs_ridge * torch.eye(in_dim, device=dev, dtype=torch.float32)

            try:
                warm_init = torch.linalg.solve(A.T, B.T).T.to(dtype=orig_dtype)
                with torch.no_grad():
                    cold_loss = _wudi_objective(cold_init).detach().float()
                    warm_loss = _wudi_objective(warm_init).detach().float()

                if bool(torch.isfinite(warm_loss).item()) and bool((warm_loss <= cold_loss).item()):
                    init_delta = warm_init
                else:
                    print(
                        "[INFO] WUDI warm_init neglected; falling back to cold_init "
                        f"(cold_loss={cold_loss.item()}, warm_loss={warm_loss.item()})"
                    )
            except RuntimeError:
                print("[INFO] WUDI warm_init neglected; falling back to cold_init (CFS solve failed)")
                init_delta = cold_init

        merging_vector = torch.nn.Parameter(init_delta.to(dtype=orig_dtype))

        opt = torch.optim.Adam(
            [merging_vector],
            lr=lr,
            weight_decay=weight_decay,
        )

        active_loss_steps = set(loss_steps or [])
        losses = []

        def _loss() -> torch.Tensor:
            return _wudi_objective(merging_vector)

        if 0 in active_loss_steps:
            with torch.no_grad():
                losses.append(float(_loss().detach().float().cpu().item()))

        # Few-step Adam refinement
        for step in range(1, iter_num + 1):
            loss = _loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step in active_loss_steps:
                with torch.no_grad():
                    losses.append(float(_loss().detach().float().cpu().item()))

        return merging_vector.detach().to(dtype=orig_dtype, device="cpu"), losses

    def _get_wudi_loss_steps(self, iter_num: int) -> list:
        return [step for step in self.WUDI_LOSS_STEPS if step <= iter_num]

    def _write_wudi_loss_csv(self, path: str, rows: list, loss_steps: list):
        if not path or not rows:
            return

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fieldnames = ["key"] + [f"loss_{step}" for step in loss_steps]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        xlsx_path = os.path.splitext(path)[0] + ".xlsx"
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "WUDI loss"
        worksheet.append(fieldnames)
        for row in rows:
            worksheet.append([row.get(fieldname) for fieldname in fieldnames])
        workbook.save(xlsx_path)

        print(f"Saved WUDI loss log to: {path}")
        print(f"Saved WUDI loss log to: {xlsx_path}")

    # def _optimize_wudi_vector(
    #     self,
    #     vecs_cpu: list,
    #     dev: torch.device,
    #     iter_num: int,
    #     lr: float,
    #     weight_decay: float,
    #     eps: float,
    # ) -> torch.Tensor:
    #     # stack on device: (n_task, out, in)
    #     vectors = torch.stack([v.to(dev) for v in vecs_cpu], dim=0)

    #     # init like the ViT script: sum(vectors)
    #     merging_vector = torch.nn.Parameter(vectors.sum(dim=0))

    #     opt = torch.optim.Adam([merging_vector], lr=lr, weight_decay=weight_decay)

    #     # l2 norms per task vector
    #     l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)) + eps

    #     for _ in range(iter_num):
    #         # disturbing_vectors: (n_task, out, in)
    #         disturbing = merging_vector.unsqueeze(0) - vectors

    #         # inner_product: (n_task, out, out)
    #         inner = torch.matmul(disturbing, vectors.transpose(1, 2))

    #         loss = torch.sum((inner * inner) / l2_norms.view(-1, 1, 1))

    #         opt.zero_grad(set_to_none=True)
    #         loss.backward()
    #         opt.step()

    #     return merging_vector.detach().to("cpu")

    def _sparsify_task_vectors(
        self,
        tvs: list,
        keys: list,
        K: float,
    ) -> list:
        if not 0 < K <= 1:
            raise ValueError(f"K must be in (0, 1], got {K}")

        sparse_tvs = []
        for tv in tvs:
            tv_keys = [k for k in keys if k in tv]
            flat_tv = torch.cat([tv[k].reshape(-1) for k in tv_keys])
            keep_count = max(1, int(flat_tv.numel() * K))

            if keep_count == flat_tv.numel():
                sparse_tvs.append(tv)
                continue

            threshold_idx = flat_tv.numel() - keep_count + 1
            threshold, _ = flat_tv.abs().kthvalue(threshold_idx)
            sparse_tvs.append(param({
                k: tv[k] * (tv[k].abs() >= threshold)
                for k in tv_keys
            }))

        return sparse_tvs

    def _dare_sparsify_task_vectors(
        self,
        tvs: list,
        keys: list,
        K: float,
    ) -> list:
        if not 0 < K <= 1:
            raise ValueError(f"K must be in (0, 1], got {K}")

        sparse_tvs = []
        for tv in tvs:
            tv_keys = [k for k in keys if k in tv]
            sparse_tvs.append(param({
                k: tv[k] * torch.bernoulli(
                    torch.full(tv[k].shape, K, device=tv[k].device, dtype=torch.float32)
                ).to(dtype=tv[k].dtype) / K
                for k in tv_keys
            }))

        return sparse_tvs

    def _lines_scale_task_vector(
        self,
        task_vector_dict: dict,
        alpha: float,
        beta: float,
        num_blocks: int,
        layer_pattern: re.Pattern,
    ) -> dict:
        scaled = {}
        denom = num_blocks - 1 if num_blocks > 1 else 1

        for name, tensor in task_vector_dict.items():
            layer_scale = alpha
            match = layer_pattern.match(name)
            if match:
                layer = int(match.group(1))
                layer_scale = alpha + beta * (layer / denom)
            scaled[name] = tensor * layer_scale

        print(f"LiNeS: The layers are scaled between {alpha} to {alpha + beta}")
        return scaled

    def _task_arithmetic_fallback(
        self,
        vecs_cpu: list,
        fallback_scaling: float,
    ) -> torch.Tensor:
        merged = torch.zeros_like(vecs_cpu[0])
        for v in vecs_cpu:
            merged.add_(v)
        return fallback_scaling * merged

    def _get_task_vectors_for_key(
        self,
        base_model: param,
        models_to_merge: list,
        key: str,
    ) -> list:
        if key not in models_to_merge[0]:
            raise ValueError(f"Missing key in task vectors: {key}")

        base_tensor = base_model[key]
        vecs_cpu = []
        for model in models_to_merge:
            if key not in model:
                raise ValueError(f"Missing key in model to merge: {key}")
            if model[key].shape != base_tensor.shape:
                raise ValueError(f"Shape mismatch for key: {key}")
            vecs_cpu.append(model[key] - base_tensor)
        return vecs_cpu

    def _write_merged_key(
        self,
        base_model: param,
        key: str,
        merged_delta: torch.Tensor,
        scaling: float,
    ):
        delta = merged_delta.to(
            device=base_model[key].device,
            dtype=base_model[key].dtype,
        )
        base_model.param_dict[key].add_(delta, alpha=scaling)

    @utils.args_inspector
    @torch.inference_mode()
    def task_arithmetic(
        self,
        base_model: param,
        models_to_merge: list,
        scaling: float = 1.0,
    ):
        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        return base_model + scaling * sum(task_vectors)
    
    @utils.args_inspector
    def wudi_merge(
        self,
        base_model: param,
        models_to_merge: list,
        scaling: float = 1.0,

        # WUDI optimizer knobs
        iter_num: int = 200,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        device: str = "cuda",   # "cuda" strongly recommended; "cpu" will be very slow

        # fallback for keys not WUDI-optimized
        fallback: str = "task_arithmetic",  # "task_arithmetic" or "zero"
        fallback_scaling: float = 1.0,
        eps: float = 1e-12,
        warm_start: bool = True,
        cfs_ridge: float = 1e-5,
        loss_log_path: str = None,
        verbose: bool = True,
    ):
        """
        WUDI-style merge:
          - Compute task vectors tv_i = ft_i - base
          - For selected 2D keys, optimize a merged task vector per key via redundancy loss
          - For other keys, fallback to sum(tv_i) (Task Arithmetic)
          - Return base + scaling * merged_task_vector
        """
        base_keys = list(base_model.keys())

        def _use_wudi_for_key(name: str, t: torch.Tensor) -> bool:
            if t.ndim != 2:
                return False
            return True

        # choose device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dev = torch.device(device)
        loss_steps = self._get_wudi_loss_steps(iter_num) if loss_log_path else []
        loss_log_rows = []

        for k in tqdm.tqdm(base_keys, desc="WUDI merge (per-key)"):
            # Build task vectors only for this key to avoid full-checkpoint deltas.
            vecs_cpu = self._get_task_vectors_for_key(
                base_model=base_model,
                models_to_merge=models_to_merge,
                key=k,
            )
            t0 = vecs_cpu[0]

            if _use_wudi_for_key(k, t0):
                merged_delta, losses = self._optimize_wudi_vector(
                    vecs_cpu=vecs_cpu,
                    dev=dev,
                    iter_num=iter_num,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    warm_start=warm_start,
                    cfs_ridge=cfs_ridge,
                    loss_steps=loss_steps,
                )
                if loss_log_path:
                    loss_log_rows.append({
                        "key": k,
                        **{
                            f"loss_{step}": loss
                            for step, loss in zip(loss_steps, losses)
                        },
                    })

                if verbose:
                     print(f'[INFO] {k} is optimized under WUDI')

            else:
                # fallback for embeddings / norms / lm_head / 1D etc.
                print(f'{k} is not optimized with WUDI and fallback to {fallback}')
                if fallback == "zero":
                    merged_delta = torch.zeros_like(t0)
                else:
                    merged_delta = self._task_arithmetic_fallback(vecs_cpu, fallback_scaling)

            self._write_merged_key(base_model, k, merged_delta, scaling)
            del vecs_cpu, merged_delta

        self._write_wudi_loss_csv(loss_log_path, loss_log_rows, loss_steps)
        return base_model

    @utils.args_inspector
    def sparsed_wudi_merge(
        self,
        base_model: param,
        models_to_merge: list,
        scaling: float = 1.0,

        # WUDI optimizer knobs
        iter_num: int = 200,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        device: str = "cuda",   # "cuda" strongly recommended; "cpu" will be very slow

        # Sparsification before WUDI
        K: float = 0.7,
        sparsify_variant: str = "ties_sparsify",

        # fallback for keys not WUDI-optimized
        fallback: str = "task_arithmetic",  # "task_arithmetic" or "zero"
        fallback_scaling: float = 1.0,
        eps: float = 1e-12,
        warm_start: bool = True,
        cfs_ridge: float = 1e-5,
        loss_log_path: str = None,
        verbose: bool = True,
    ):
        """
        Sparsed WUDI-style merge:
          - Compute task vectors tv_i = ft_i - base
          - Apply ties_sparsify or dare_sparsify to each task vector
          - For 2D keys, optimize a merged task vector per key via redundancy loss
          - For other keys, fallback to sum(tv_i) (Task Arithmetic)
          - Return base + scaling * merged_task_vector
        """
        base_keys = list(base_model.keys())

        def _use_wudi_for_key(name: str, t: torch.Tensor) -> bool:
            if t.ndim != 2:
                return False
            return True

        # task vectors: tv_i = ft_i - base
        tvs = [m - base_model for m in models_to_merge]
        if sparsify_variant == "ties_sparsify":
            tvs = self._sparsify_task_vectors(tvs=tvs, keys=base_keys, K=K)
        elif sparsify_variant == "dare_sparsify":
            tvs = self._dare_sparsify_task_vectors(tvs=tvs, keys=base_keys, K=K)
        else:
            raise ValueError(
                f"Unknown sparsify_variant={sparsify_variant}. "
                "Choose one of: ties_sparsify, dare_sparsify"
            )

        # choose device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dev = torch.device(device)
        loss_steps = self._get_wudi_loss_steps(iter_num) if loss_log_path else []
        loss_log_rows = []

        for k in tqdm.tqdm(base_keys, desc="Sparsed WUDI merge (per-key)"):
            if k not in tvs[0]:
                raise ValueError(f"Missing key in task vectors: {k}")

            # gather per-task tensors (keep dtype consistent)
            vecs_cpu = [tv[k] for tv in tvs]
            t0 = vecs_cpu[0]

            # Standard checkpoints should have identical tensor shapes.
            if any(v.shape != t0.shape for v in vecs_cpu):
                raise ValueError(f"Shape mismatch for key: {k}")

            if _use_wudi_for_key(k, t0):
                merged_delta, losses = self._optimize_wudi_vector(
                    vecs_cpu=vecs_cpu,
                    dev=dev,
                    iter_num=iter_num,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    warm_start=warm_start,
                    cfs_ridge=cfs_ridge,
                    loss_steps=loss_steps,
                )
                if loss_log_path:
                    loss_log_rows.append({
                        "key": k,
                        **{
                            f"loss_{step}": loss
                            for step, loss in zip(loss_steps, losses)
                        },
                    })

                if verbose:
                     print(f'[INFO] {k} is optimized under sparsed WUDI')

            else:
                # fallback for embeddings / norms / lm_head / 1D etc.
                print(f'{k} is not optimized with WUDI and fallback to {fallback}')
                if fallback == "zero":
                    merged_delta = torch.zeros_like(t0)
                else:
                    merged_delta = self._task_arithmetic_fallback(vecs_cpu, fallback_scaling)

            self._write_merged_key(base_model, k, merged_delta, scaling)
            del vecs_cpu, merged_delta

        self._write_wudi_loss_csv(loss_log_path, loss_log_rows, loss_steps)
        del tvs
        return base_model
    
    @utils.args_inspector
    def selective_wudi_merge(
        self,
        base_model: param,
        models_to_merge: list,
        scaling: float = 1.0,

        # WUDI optimizer knobs
        iter_num: int = 200,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        device: str = "cuda",   # "cuda" strongly recommended; "cpu" will be very slow

        # which params to run WUDI on
        variant: str = "wudi_all_linear",

        # fallback for keys not WUDI-optimized
        fallback: str = "task_arithmetic",  # "task_arithmetic" or "zero"
        fallback_scaling: float = 1.0,
        eps: float = 1e-12,
        warm_start: bool = True,
        cfs_ridge: float = 1e-5,
        loss_log_path: str = None,
        verbose: bool = True,
    ):
        """
        WUDI-style merge:
          - Compute task vectors tv_i = ft_i - base
          - For selected 2D keys, optimize a merged task vector per key via redundancy loss
          - For other keys, fallback to sum(tv_i) (Task Arithmetic)
          - Return base + scaling * merged_task_vector
        """

        attention_proj = r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$"
        mlp_proj = r".*mlp\.(gate_proj|up_proj|down_proj)\.weight$"
        projection_patterns = [attention_proj, mlp_proj]
        layer_pattern = re.compile(r".*model\.layers\.(\d+)\..*")

        def _match_any(patterns, name: str) -> bool:
            return any(re.match(p, name) for p in patterns)

        def _infer_layer_count(keys) -> int:
            layer_indices = []
            for name in keys:
                match = layer_pattern.match(name)
                if match:
                    layer_indices.append(int(match.group(1)))
            if not layer_indices:
                return 0
            return max(layer_indices) + 1

        base_keys = list(base_model.keys())
        num_layers = _infer_layer_count(base_keys)

        use_lines_scaling = variant == "wudi_lines_all_linear"

        if variant in {"wudi_all_linear", "wudi_lines_all_linear"}:
            variant_include = None
            selected_layers = None
        elif variant == "wudi_attention_only":
            variant_include = [attention_proj]
            selected_layers = None
        elif variant == "wudi_mlp_only":
            variant_include = [mlp_proj]
            selected_layers = None
        elif variant in {"wudi_last_7_layers", "wudi_last_14_layers", "wudi_last_21_layers"}:
            if num_layers == 0:
                raise ValueError("Could not infer Llama layer count from model.layers.{idx} parameter names")
            last_n_by_variant = {
                "wudi_last_7_layers": 7,
                "wudi_last_14_layers": 14,
                "wudi_last_21_layers": 21,
            }
            last_n = last_n_by_variant[variant]
            start_layer = max(num_layers - last_n, 0)
            selected_layers = set(range(start_layer, num_layers))
            variant_include = projection_patterns
        else:
            raise ValueError(
                f"Unknown WUDI variant={variant}. Choose one of: "
                "wudi_all_linear, wudi_lines_all_linear, wudi_attention_only, wudi_mlp_only, "
                "wudi_last_7_layers, wudi_last_14_layers, wudi_last_21_layers"
            )

        def _layer_index(name: str):
            match = layer_pattern.match(name)
            if not match:
                return None
            return int(match.group(1))

        def _use_wudi_for_key(name: str, t: torch.Tensor) -> bool:
            if t.ndim != 2:
                return False
            if variant_include and not _match_any(variant_include, name):
                return False
            if selected_layers is not None and _layer_index(name) not in selected_layers:
                return False
            return True

        # choose device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dev = torch.device(device)
        loss_steps = self._get_wudi_loss_steps(iter_num) if loss_log_path else []
        loss_log_rows = []

        for k in tqdm.tqdm(base_keys, desc="WUDI merge (per-key)"):
            # Build task vectors only for this key to avoid full-checkpoint deltas.
            vecs_cpu = self._get_task_vectors_for_key(
                base_model=base_model,
                models_to_merge=models_to_merge,
                key=k,
            )
            t0 = vecs_cpu[0]

            if _use_wudi_for_key(k, t0):
                merged_delta, losses = self._optimize_wudi_vector(
                    vecs_cpu=vecs_cpu,
                    dev=dev,
                    iter_num=iter_num,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    warm_start=warm_start,
                    cfs_ridge=cfs_ridge,
                    loss_steps=loss_steps,
                )
                if loss_log_path:
                    loss_log_rows.append({
                        "key": k,
                        **{
                            f"loss_{step}": loss
                            for step, loss in zip(loss_steps, losses)
                        },
                    })

                if verbose:
                     print(f'[INFO] {k} is optimized under WUDI')

            else:
                # fallback for embeddings / norms / lm_head / 1D etc.
                print(f'{k} is not optimized with WUDI and fallback to {fallback}')
                if fallback == "zero":
                    merged_delta = torch.zeros_like(t0)
                else:
                    merged_delta = self._task_arithmetic_fallback(vecs_cpu, fallback_scaling)

            if use_lines_scaling:
                if num_layers == 0:
                    raise ValueError("Could not infer Llama layer count from model.layers.{idx} parameter names")
                denom = num_layers - 1 if num_layers > 1 else 1
                layer_scale = 1 / len(models_to_merge)
                match = layer_pattern.match(k)
                if match:
                    layer = int(match.group(1))
                    layer_scale = layer_scale + scaling * (layer / denom)
                self._write_merged_key(base_model, k, merged_delta, layer_scale)
            else:
                self._write_merged_key(base_model, k, merged_delta, scaling)
            del vecs_cpu, merged_delta

        if use_lines_scaling:
            print(f"LiNeS: The layers are scaled between {1 / len(models_to_merge)} to {1 / len(models_to_merge) + scaling}")
        self._write_wudi_loss_csv(loss_log_path, loss_log_rows, loss_steps)
        return base_model
    
