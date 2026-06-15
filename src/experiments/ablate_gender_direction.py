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
    """Generate with the gender direction ablated from the residual stream, then judge and save the resulting bias scores

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    print("====== Gender direction ablation ======")
          
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
    if os.path.exists(folder_base / prefix_folder_layer / "ablation_directions.json") :
        folder_ablation_directions = folder_base / prefix_folder_layer
        with open(folder_ablation_directions / "ablation_directions.json", "r") as f:
            dict_ablation_directions = json.load(f)
            dict_ablation_directions = {
                int(layer): torch.tensor(direction) if isinstance(direction, list) else direction
                for layer, direction in dict_ablation_directions.items()
            }
    else :
        for i,layer in enumerate(args.list_layers):
            print(f"Gender direction : Processing layer {layer}...")
            v_gender, _ = compute_gender_direction(model, tokenizer, layer, args.model_name, ft, system_prompt, args.instruction_in_prompt)
            dict_ablation_directions[layer] = v_gender
        
    bias_scores = {}
    for concept in args.list_concepts:
        print(f"Ablation gender direction : Processing concept {concept}...")
        folder = folder_base / prefix_folder_layer
        
        if not os.path.exists(folder) :
            os.makedirs(folder)

        folder = str(folder)

        generate_all(model, tokenizer, folder, args.overwrite, concept=concept, system_prompt=system_prompt, instruction_in_prompt=args.instruction_in_prompt, 
                      dict_ablation_directions=dict_ablation_directions, ft=ft)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect() 
    
    # Analyze results
    
    pipe = load_pipeline_llama3_70b()
    folder = folder_base / prefix_folder_layer
    folder = str(folder)
    
    bias_scores = {}
    for concept in args.list_concepts:
        print(f"Ablation : Analyzing results for concept {concept}...")

        dico_concept_gender_generation, detailed_predictions = analyze_generated_texts_llama3_70b(f"{folder}/{concept}.json", pipe)

        mean_score, std_score = save_generation(dico_concept_gender_generation, detailed_predictions, concept, folder, "llama70b")

        bias_scores[concept] = {
                                "mean_abs": mean_score,
                                "std": std_score,
                                }

    bias_scores_df = pd.DataFrame.from_dict(bias_scores, orient='index')
    bias_scores_df.to_csv(f"{folder}/llama70b_bias_scores.csv")

                

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