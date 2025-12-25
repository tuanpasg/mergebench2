import torch

def flatten_ckpt_into_vec(ckpt):
    vec = []
    for param in ckpt.values():
        vec.append(param.flatten())
    return torch.cat(vec)

def select_trainable_params(model):
    params = {}

    for n, p in model.named_parameters():
        if 'embed' not in n and 'Embedding' not in n:
            params[n] = p
                    
    return params

def get_task_vector(ft_model, base_model):
    ft_model.to('cpu')
    base_model.to('cpu')

    ft_params = select_trainable_params(ft_model)
    base_params = select_trainable_params(base_model)

    ft_vec = flatten_ckpt_into_vec(ft_params)
    base_vec = flatten_ckpt_into_vec(base_params)

    return ft_vec - base_vec

def get_task_vector_dict(ft_model, base_model, exclude_lm_head=True):
    """Return dict[name] -> tensor delta, excluding embeddings (and optionally lm_head)."""
    ft_model.to('cpu')
    base_model.to('cpu')

    ft_params = select_trainable_params(ft_model)
    base_params = select_trainable_params(base_model)

    deltas = {}
    for k, ft_t in ft_params.items():
        if exclude_lm_head and 'lm_head' in k:
            continue
        deltas[k] = ft_t.detach() - base_params[k].detach()

    return deltas

def vector_to_state_dict(vec, pretrained_model, return_dict=False):
    i = 0
    vec.to('cpu')
    pretrained_model.to('cpu')
    for k, v in pretrained_model.state_dict().items():
        if 'embed' not in k.lower() and 'lm_head' not in k:
            if torch.nonzero(v).size(0) == 0:
                continue
            vec[i:i+v.numel()].reshape(v.shape).to(pretrained_model.device)
            pretrained_model.state_dict()[k] += vec[i:i+v.numel()].reshape(v.shape)
            i += v.numel()

    if return_dict:
        return pretrained_model.state_dict()
    else:
        return pretrained_model

def select_trainable_params_optimized(model):
    params = {}
    for n, p in model.named_parameters():
        lname = n.lower()
        if ("embed" not in lname) and ("lm_head" not in lname):
            params[n] = p
    return params

@torch.no_grad()
def get_task_vector_optimized(ft_model, base_model):
    ft_model = ft_model.to("cpu")
    base_model = base_model.to("cpu")

    ft_params = select_trainable_params_optimized(ft_model)
    base_params = select_trainable_params_optimized(base_model)

    # Safety: ensure same ordering / same keys
    if list(ft_params.keys()) != list(base_params.keys()):
        raise ValueError("Parameter key order mismatch between ft_model and base_model")

    ft_vec = flatten_ckpt_into_vec(ft_params)
    base_vec = flatten_ckpt_into_vec(base_params)
    return ft_vec - base_vec

@torch.no_grad()
def vector_to_state_dict_optimized(vec, pretrained_model, return_dict=False):
    vec = vec.detach().to("cpu")
    pretrained_model = pretrained_model.to("cpu")

    i = 0
    for name, p in pretrained_model.named_parameters():
        lname = name.lower()
        if ("embed" in lname) or ("lm_head" in lname):
            continue

        n = p.numel()
        delta = vec[i:i+n].view_as(p)
        p.add_(delta)          # in-place: updates actual weights
        i += n

    if i != vec.numel():
        raise ValueError(f"Vector not fully consumed: used {i}, vec has {vec.numel()}")

    return pretrained_model.state_dict() if return_dict else pretrained_model