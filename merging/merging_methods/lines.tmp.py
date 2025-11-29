from merging_methods.utils import *
from merging_methods.merger import Merger
import copy

def _get_param_deltas(ft_model, base_model):
    """Return dict[name] -> tensor delta, excluding embeddings."""
    ft_model.to("cpu")
    base_model.to("cpu")
    ft_params = select_trainable_params(ft_model)
    base_params = select_trainable_params(base_model)
    return {k: (ft_params[k].detach() - base_params[k].detach()) for k in ft_params.keys()}

def _sum_param_deltas(delta_dicts):
    """Elementwise sum of a list of param-delta dicts."""
    summed = {}
    for d in delta_dicts:
        for k, v in d.items():
            summed[k] = summed.get(k, 0) + v
    return summed

def LiNeS_scaling(task_vector_dict, alpha, beta, num_blocks):
    """
    LiNeS: Progressively scales the task vector based on layer depth.

    task_vector_dict: dict[param_name, Tensor]
    """
    key_blocks = [f".{i}." for i in range(num_blocks)]
    scaled = {}

    for name, tensor in task_vector_dict.items():
        layer_scale = alpha
        for layer, block in enumerate(key_blocks):
            if block in name:
                layer_scale = alpha + beta * (layer / (num_blocks - 1 if num_blocks > 1 else 1))
                break
        scaled[name] = tensor * layer_scale

    print(f"LiNeS: The layers are scaled between {alpha} to {alpha + beta}")
    return scaled

class LiNeS(Merger):
    ...
    def merge(self, **kwargs):
        print("Start LineS merging ...")
        beta = kwargs["beta_coef"]

        # 1) Build multi-task task vector as per-parameter deltas
        task_vectors = [_get_param_deltas(ft_model, self.base_model) for ft_model in self.ft_ckpts]
        num_tasks = len(task_vectors)
        if num_tasks == 0:
            raise ValueError("LiNeSMerger: no fine-tuned checkpoints provided.")

        multi_task_tv = _sum_param_deltas(task_vectors)

        # 2) Infer num_blocks
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "num_hidden_layers"):
            num_blocks = self.base_model.config.num_hidden_layers
        else:
            raise ValueError("Cannot retrieve model architecture info")

        # 3) Compute alpha
        alpha = 1 / num_tasks

        # 4) Apply LiNeS depth-wise scaling
        print(f"Scaling aggregated task vector with LineS: alpha={alpha}, beta={beta}, num_blocks={num_blocks}")
        scaled_tv = LiNeS_scaling(
            multi_task_tv,
            alpha=alpha,
            beta=beta,
            num_blocks=num_blocks,
        )

        # 5) Turn scaled task vector into merged model weights
        state = self.base_model.state_dict()
        for name, delta in scaled_tv.items():
            if name in state:
                state[name] = state[name].to(delta.device) + delta
        self.base_model.load_state_dict(state)

        # 6) Save
        self.base_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
