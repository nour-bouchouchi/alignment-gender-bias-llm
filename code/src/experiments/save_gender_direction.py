import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import ledoit_wolf
import os 
import argparse
import pandas as pd 
import gc 
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  
import json

from models.hf import get_model                    
from utils.cli import str2bool, get_suffix_folder                 
from constants import SYSTEM_PROMPTS
from utils.directions import compute_gender_direction
from experiments.generation import generate_all
from experiments.generation_classif import load_pipeline_llama3_70b, analyze_generated_texts_llama3_70b, save_generation


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))

device = "cuda" if torch.cuda.is_available() else "cpu"


import torch._dynamo
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

if hasattr(torch, "compile"):
    torch.compile = lambda model, *args, **kwargs: model

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"



def main(args: argparse.Namespace) -> None:
    """Estimate and save the per-layer gender direction(s) to a JSON file

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    print("====== Gender direction ======")
          
    if args.use_model_ft_dpo and args.use_model_ft_sft:
        raise ValueError("Cannot use both DPO and LoRA fine-tuning at the same time.")
    
    ft = args.use_model_ft_dpo or args.use_model_ft_sft

    system_prompt = SYSTEM_PROMPTS.get(args.system_prompt_key, None) if hasattr(args, 'system_prompt_key') else None        

    prefix_folder_layer = get_suffix_folder(args.instruction_in_prompt, args.system_prompt_key, args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft)

    # Load model and tokenizer and generate    
    model, tokenizer = get_model(args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft)

    folder_base = ROOT / "results" / "direction_ablation" 
    os.makedirs(folder_base, exist_ok=True)

    dict_ablation_directions = {}
    for i,layer in enumerate(args.list_layers):
        print(f"Gender direction : Processing layer {layer}...")
        v_gender, _ = compute_gender_direction(model, tokenizer, layer, args.model_name, ft, system_prompt, args.instruction_in_prompt)
        dict_ablation_directions[layer] = v_gender
    
    dict_ablation_directions_serializable = {
        layer: direction.cpu().tolist() if torch.is_tensor(direction) else direction
        for layer, direction in dict_ablation_directions.items()
    }

    folder = folder_base / prefix_folder_layer
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/ablation_directions.json", "w") as f:
        json.dump(dict_ablation_directions_serializable, f, indent=4)

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--overwrite', type=str2bool, default=True)
    parser.add_argument('--use_model_ft_dpo', type=str2bool, default=False)
    parser.add_argument('--use_model_ft_sft', type=str2bool, default=False)
    parser.add_argument('--list_concepts', nargs='+',
                        default=['professions', 'colors', 'months'])
    parser.add_argument('--list_layers', nargs='+', default=[5], type=int)
    parser.add_argument('--system_prompt_key', type=str, default="none",
                        help='Type de system prompt à utiliser')
    parser.add_argument('--instruction_in_prompt', type=str2bool, default=False,
                   help='Put instruction directly in prompt rather than system prompt')
    args = parser.parse_args()
    main(args)