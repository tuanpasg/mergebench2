import copy
import torch

def LiNeS_scaling(task_vector, alpha, beta, num_blocks):
    """
    LiNeS: Progressively scales the task vector based on layer depth.

    task_vector: TaskVector-like object with a `.vector` dict[str, Tensor]
    alpha: min scaling factor
    beta: max scaling difference between last and first block
    num_blocks: number of transformer blocks / layers
    """
    scaled_task_vector = copy.deepcopy(task_vector)

    # match layers by ".0.", ".1.", ..., ".(num_blocks-1)."
    key_blocks = [f".{i}." for i in range(num_blocks)]

    layer_scalings_dict = {}
    for k in scaled_task_vector.vector.keys():
        for layer, block in enumerate(key_blocks):
            if block in k:
                layer_scalings_dict[k] = alpha + beta * (layer / (num_blocks - 1))
                break

    print(f"LiNeS: The layers are scaled between {alpha} to {alpha + beta}")

    scaled_task_vector.vector = {
        k: scaled_task_vector.vector[k] * layer_scalings_dict.get(k, alpha)
        for k in scaled_task_vector.vector.keys()
    }

    return scaled_task_vector


def _infer_num_blocks(model, args=None):
    """
    Infer number of blocks for architecture-agnostic LiNeS.

    Priority:
    1. args.num_blocks (if provided)
    2. model.config.num_hidden_layers (HF transformers, e.g. Llama-3.2-3B)
    3. ViT fallback used in the original LiNeS repo
    """
    # 1) explicit override
    if args is not None and hasattr(args, "num_blocks") and args.num_blocks is not None:
        return args.num_blocks

    # 2) Hugging Face-style config (covers Llama-3.x, etc.)
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers

    # 3) Fallback: original ViT rule from LiNeS repo
    if args is not None and getattr(args, "model", None) == "ViT-L-14":
        return 24
    else:
        return 12


def _tv_l2_norm(task_vector):
    """
    L2 norm of a TaskVector (assuming PyTorch tensors).
    """
    flat_params = [p.flatten() for p in task_vector.vector.values()]
    return torch.linalg.norm(torch.cat(flat_params))
