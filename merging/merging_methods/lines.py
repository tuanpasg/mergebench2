from merging_methods.utils import *
from merging_methods.merger import Merger
import copy

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

class LiNeSMerger(Merger):
    """
    LiNeS merging method (architecture-agnostic, supports Llama-3.2-3B).

    Expected kwargs for merge():
        - scaling_coef (float): LiNeS beta (max depth-dependent increment)
        - alpha (float, optional): if not given, computed from task vectors 
                                   using the multi-task heuristic.
        - args (optional): config/args object (used only for num_blocks override).
    """

    def __init__(self, base_model, ft_models, save_path, args=None):
        super().__init__(base_model, ft_models, save_path)
        self.args = args  # optional, only for num_blocks/alpha heuristic

    def merge(self, **kwargs):
        # LiNeS beta (depth-dependent increment)
        beta = kwargs["scaling_coef"]

        # 1) Build multi-task task vector using baseline merging methods: Here is only Task Arithmetic
        task_vectors = [get_task_vector(ft_model, self.base_model) for ft_model in self.ft_ckpts]
        num_tasks = len(task_vectors)

        if num_tasks == 0:
            raise ValueError("LiNeSMerger: no fine-tuned checkpoints provided.")

        multi_task_tv = sum(task_vectors)  # TaskVector __add__ should be defined

        # 2) Infer num_blocks for the architecture (e.g. Llama-3.2-3B)
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "num_hidden_layers"):
            num_blocks = self.base_model.config.num_hidden_layers
        else:
            raise ValueError("Cannot retrieve model architecture info")

        # 3) Compute alpha (Hardcode for Task Arithmetic)
        alpha = 1/num_tasks

        # 4) Apply LiNeS depth-wise scaling to the multi-task vector
        scaled_tv = LiNeS_scaling(
            multi_task_tv,
            alpha=alpha,
            beta=beta,
            num_blocks=num_blocks,
        )

        # 5) Turn scaled task vector into merged model weights
        merged_model = vector_to_state_dict(scaled_tv, self.base_model)

        # 6) Save like other methods
        merged_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
