import torch
from merging_methods.utils import *
from merging_methods.merger import Merger


class Consensus(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)\
    
    def tune_lamda(self, mtl_tv, tv, i):
        for lamda in [0.2,0.6]:    
            print(f'Tuning lamda: {lamda} for model {i}')
            tall_mask = (torch.abs(tv) > torch.abs(mtl_tv - tv) * lamda)
        
            masked_model = vector_to_state_dict(tv * tall_mask, self.base_model)
            
            save_dir = './tmp/' + self.base_model_name.split('/')[1] + '/' + f'Consensus_{i}_lamda_' + str(lamda)
            masked_model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
    
    def merge(self, **kwargs):
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