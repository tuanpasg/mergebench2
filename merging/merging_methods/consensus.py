import copy
import torch
import os
from merging_methods.utils import *
from merging_methods.merger import Merger
from huggingface_hub import HfApi, upload_folder


class Consensus(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)
    
    # def tune_lamda(self, mtl_tv, tv, i):
    #     for lamda in [0.2,0.6]:    
    #         print(f'Tuning lamda: {lamda} for model {i}')
    #         tall_mask = (torch.abs(tv) > torch.abs(mtl_tv - tv) * lamda)
        
    #         masked_model = vector_to_state_dict(tv * tall_mask, self.base_model)
            
    #         save_dir = './tmp/' + self.base_model_name.split('/')[1] + '/' + f'Consensus_{i}_lamda_' + str(lamda)
    #         masked_model.save_pretrained(save_dir)
    #         self.tokenizer.save_pretrained(save_dir)

    def tune_lamda_all(self):
        task_vectors = [get_task_vector_optimized(ft_model, self.base_model) for ft_model in self.ft_ckpts]
        mtl_tv = sum(task_vectors)

        api = HfApi()
        base = self.base_model_name.split("/")[1]

        for i in range(len(task_vectors)):
            tv = task_vectors[i]
            for lamda in [0.2, 0.3, 0.4, 0.5, 0.6]:    
                print(f'Tuning lamda: {lamda} for model {i}')
                tall_mask = (torch.abs(tv) > torch.abs(mtl_tv - tv) * lamda)
            
                base_model_copy = copy.deepcopy(self.base_model)
                masked_model = vector_to_state_dict_optimized(mtl_tv * tall_mask, base_model_copy)
                
                model_name = f"{base}-Consensus_{i}_lambda_{lamda}"
                save_dir = f"/root/tmp/{model_name}"
                os.makedirs(save_dir, exist_ok=True)

                masked_model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)

                # Upload model to HF space
                repo_id = f"tuanpasg/{model_name}"
                print(f"Creating repo {repo_id} (if not exists)...")
                api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

                upload_folder(
                    folder_path=save_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Upload model",
                )

    def merge(self, **kwargs):
        lamda_tuning = kwargs.get('lamda_tuning', False)
        if lamda_tuning:
            print("CONSENSUS SPARSITY TUNING...")
            self.tune_lamda_all()
            return

        k = kwargs['k']
        scaling_coef = kwargs['scaling_coef']
        lamdas = kwargs.get('lamda', [0.2])  # list from CLI

        task_vectors = [get_task_vector(ft_model, self.base_model) for ft_model in self.ft_ckpts]
        mtl_tv = sum(task_vectors)

        tall_masks = []
        # replace this with results from the tune_lamda function
        # lamdas = [0.2, 0.2, 0.2, 0.2, 0.2]

        if len(lamdas) == 1:
            lamdas = lamdas * len(task_vectors)
        elif len(lamdas) != len(task_vectors):
            raise ValueError(f'lamda list length {len(lamdas)} must be 1 or equal to number of tasks {len(task_vectors)}')

        print(f'[k = {k}][scaling_coef = {scaling_coef}][lamda = {lamdas}]')
        
        for i in range(len(task_vectors)):
            tv = task_vectors[i]
            tall_mask = (torch.abs(tv) > torch.abs(mtl_tv - tv) * lamdas[i])
            tall_masks.append(tall_mask)
        
        consensus_mask = torch.zeros_like(tall_masks[0], dtype=torch.int16)
        for mask in tall_masks:
            consensus_mask += mask.to(torch.int16)
        consensus_mask = consensus_mask >= k
        
        merged_tv = mtl_tv * consensus_mask * scaling_coef
            
        merged_model = vector_to_state_dict(merged_tv, self.base_model)

        merged_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
