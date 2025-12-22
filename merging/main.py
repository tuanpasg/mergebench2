from prepare_args import prepare_args, create_parser
import importlib
import time
import os
from pathlib import Path
import json

def write_readme(save_path, args, kwargs, ft_ckpts, runtime_sec):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    readme_path = os.path.join(save_path, "README.md")

    args_dict = vars(args)
    content = [
        f"# Merged Model",
        f"- Base model: `{args.base_model}`",
        f"- Algorithm: `{args.algo}`",
        f"- Save path: `{save_path}`",
        f"- Fine-tuned checkpoints: {ft_ckpts}",
        f"- Merge runtime (s): {runtime_sec:.3f}",
        "",
        "## Arguments",
        "```json",
        json.dumps(args_dict, indent=2),
        "```",
        "",
        "## Merge kwargs",
        "```json",
        json.dumps(kwargs, indent=2),
        "```",
    ]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

def get_ft_ckpts(base_model):
    model_name = base_model.split('/')[-1]
    # task_names = ['instruction', 'math', 'coding', 'safety', 'multilingual']
    task_names = ['instruction', 'math', 'coding']
    return [f'MergeBench/{model_name}_{task_name}' for task_name in task_names]

def parse_args():
    parser = create_parser()

    parser.add_argument('--base-model', default='meta-llama/Llama-3.2-3B', type=str)
    parser.add_argument('--algo', default='TaskArithmetic', type=str, choices=['TaskArithmetic', 'TIES', 'DARE', 'LocalizeAndStitch', 'Consensus', 'RegMean', 'Fisher','LiNeS', 'CABS'])
    parser.add_argument('--save-path', default='./merged_models/', type=str)

    return parser.parse_args()

def main(args):
    kwargs = prepare_args(args)
    merger_module = importlib.import_module("merging_methods")
    ft_ckpts = get_ft_ckpts(args.base_model)

    kwargs_str = "_".join(f"{key}_{value}" for key, value in kwargs.items() if key not in ['fisher_only','merge_only','save_group','task_names','keep_checkpoints'])
    if args.save_group:
        task_group = args.save_group
    elif args.task_names:
        task_group = args.task_names
    else:
        task_group = None

    save_path = args.save_path + args.base_model.split('/')[1] + '_merged/' + args.algo
    if task_group:
        save_path += '_task_names_' + task_group
    if kwargs_str != '':
        save_path += '_' + kwargs_str
        
    print('merged model save to:',save_path)
    merger = getattr(merger_module, args.algo)(args.base_model, ft_ckpts, save_path)
    print(args)
    print(kwargs)

    start = time.perf_counter()
    merger.merge(**kwargs)
    runtime_sec = time.perf_counter() - start

    write_readme(save_path, args, kwargs, ft_ckpts, runtime_sec)
    print("✅ Merging completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
