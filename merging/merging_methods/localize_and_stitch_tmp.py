from merging_methods.utils import *
from merging_methods.merger import Merger
from merging_methods.localize_utils import *
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import torch
from typing import List, Optional

@torch.no_grad()
def stitch_dataless_las(
    task_vectors: List[torch.Tensor],
    masks: List[torch.Tensor],
    *,
    eps: float = 0.0,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Paper-style Dataless Localize-and-Stitch stitching:
      sum_masked_tv = Σ_i (mask_i * tv_i)
      sum_masks     = Σ_i mask_i
      stitched_tv   = sum_masked_tv / sum_masks   (elementwise, only where sum_masks>0)

    - masks can be bool or 0/1 numeric tensors.
    - returns 0 where sum_masks == 0 (i.e., no task selected that position).
    """
    assert len(task_vectors) == len(masks) and len(task_vectors) > 0

    # choose a device/dtype from the first tv
    dev = task_vectors[0].device
    tv_dtype = task_vectors[0].dtype

    # accumulation in float32 for stability (you can change if you want)
    acc_dtype = torch.float32
    out_dtype = out_dtype or tv_dtype

    sum_masked = torch.zeros_like(task_vectors[0], device=dev, dtype=acc_dtype)
    sum_masks = torch.zeros_like(task_vectors[0], device=dev, dtype=acc_dtype)

    for tv, m in zip(task_vectors, masks):
        assert tv.shape == task_vectors[0].shape, "All task vectors must have the same shape"
        assert m.shape == tv.shape, "Each mask must match its task vector shape"

        tv = tv.to(device=dev, dtype=acc_dtype)
        m_f = m.to(device=dev)
        if m_f.dtype != torch.bool:
            # treat nonzero as True
            m_f = m_f != 0
        m_f = m_f.to(dtype=acc_dtype)

        sum_masked += m_f * tv
        sum_masks += m_f

    stitched = torch.zeros_like(sum_masked, dtype=acc_dtype)
    denom = sum_masks + eps

    nonzero = sum_masks != 0
    stitched[nonzero] = sum_masked[nonzero] / denom[nonzero]

    return stitched.to(dtype=out_dtype)

class LocalizeAndStitchTmp(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)

        self.task_names = ['instruction', 'math', 'coding', 'safety', 'multilingual']
    
    def extract_format_keys(self, task):
        dataset = load_dataset(f'MergeBench/{task}_val', split='train')

        if task == 'safety':
            format_keys = {"instruction_key": "prompt", "output_key": "response"}
        elif task == 'multilingual':
            format_keys = {"instruction_key": "inputs", "output_key": "targets"}
        elif task == 'math': 
            format_keys = {"instruction_key": "query", "output_key": "response"}
        elif task == 'instruction':
            format_keys = {"instruction_key": "instruction", "output_key": "output"}
        elif task == 'coding': 
            format_keys = {"output_key": "response"}
        
        return dataset, format_keys
    
    @torch.no_grad()
    def merge(self, **kwargs):
        graft_args = {}
        dataless = kwargs['dataless']
        graft_args['sparsity'] = kwargs['sparsity']
        graft_args['sigmoid_bias'] = kwargs['sigmoid_bias']
        if not dataless:
            graft_args['lr'] = kwargs['learning_rate']
            graft_args['num_train_epochs'] = kwargs['num_train_epochs']
            graft_args['l1_strength'] = kwargs['l1_strength']

        # Localize
        masks = []
        task_vectors = []
        trainable_params = None
        if dataless:
            for i in range(len(self.ft_ckpts)):
                current_task = self.task_names[i]
                print(f'Localizing {current_task} model')
                ft_model = self.ft_ckpts[i]

                # Compute task vector
                print("Computing task vectors")
                task_vector = get_task_vector(ft_model, self.base_model)
                num_params = len(task_vector)

                print("Generating sparity mask")
                abs_tv = torch.abs(task_vector)
                k = int(graft_args['sparsity'] * abs_tv.numel())  # 1% of the total number of elements

                # Get the k largest values; returns values and their indices
                values, _ = torch.topk(abs_tv.view(-1), k)
                threshold = values.min()

                mask = torch.zeros_like(task_vector, requires_grad=False)
                pos_mask = abs_tv >= threshold
                mask[pos_mask] = graft_args["sigmoid_bias"]
                print('Initial topk sparsity in my mask: ', torch.nonzero(mask).numel() / num_params)
                mask[~pos_mask] = -graft_args["sigmoid_bias"]

                sigmoid = torch.nn.Sigmoid()
                frac = torch.round(sigmoid(mask))
                
                print('Proportion in mask:', frac.count_nonzero().item() / frac.numel())

                masks.append(frac)
                task_vectors.append(task_vector)
                
                del abs_tv, values, mask, frac
                import gc; gc.collect()
        else:
            for i in range(len(self.ft_ckpts)):
                current_task = self.task_names[i]
                print(f'Localizing {current_task} model')
                ft_model = self.ft_ckpts[i]
                trainable_params = select_trainable_params(ft_model)

                localizer = Localizer(trainable_params, self.base_model, ft_model, graft_args, self.base_model_name)
                
                if not dataless:
                    print(f'Training mask {current_task} model')
                    dataset, format_keys = self.extract_format_keys(self.task_names[i])

                    localizer.train_mask(dataset, format_keys) 
                
                mask, _ = localizer.interpolate_model(round_=True, return_mask=True, train=False)
                masks.append(mask)
        
        # Stitch
        print("Appling sparity masks and generating merged model")
        # final_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        # stitcher = Stitcher(trainable_params, final_model, self.base_model, self.ft_ckpts, masks)
        # merged_model = stitcher.interpolate_models()

        merged_tv = stitch_dataless_las(task_vectors,masks)
        merged_model = vector_to_state_dict(merged_tv, self.base_model)

        merged_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)