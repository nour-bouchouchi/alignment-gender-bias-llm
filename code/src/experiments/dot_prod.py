import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  

from models.hf import get_model                    
from utils.cli import str2bool, get_suffix_folder, get_sentence_prompt
from constants import PERSON, CONCEPTS, SCHEMA, SYSTEM_PROMPTS
from utils.directions import compute_lambdas, compute_gender_direction

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))

device = "cuda" if torch.cuda.is_available() else "cpu"


import torch._dynamo
import torch.nn.functional as F

torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

if hasattr(torch, "compile"):
    torch.compile = lambda model, *args, **kwargs: model

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"




def compute_dot_product(model: Any, tokenizer: Any, layer: int, concept: str, v_gender: torch.Tensor, model_name: str, ft: bool, system_prompt: str | None = None, instruction_in_prompt: bool = False, eps: float = 1e-12):
    """Compute the dot product between the gender direction and the concept activations

    Args:
        model (Any): the language model
        tokenizer (Any): the tokenizer for the model
        layer (int): the layer number to extract activations from
        concept (str): the concept to analyze (e.g. 'professions')
        v_gender (torch.Tensor): the gender direction vector
        model_name (str): the name of the model
        ft (bool): whether the model is fine-tuned
        system_prompt (str | None): optional system prompt to use
        instruction_in_prompt (bool): whether the instruction is in the prompt
        eps (float): small value to avoid division by zero
    Returns:
        tuple: mean dot products, std dot products, all neutral activations, and activation records
    """
    lambdas_concept = {"neutral": {}}
    
    activations_records = [] 
    for entity in CONCEPTS[concept]:
        if concept.startswith('ruted_'):
            sentences = [get_sentence_prompt(concept, "", entity, idx=i, dot=True) for i in range(len(PERSON["neutral"]))]
        else:
            sentences = [get_sentence_prompt(concept, persona, entity, dot=True) for persona in PERSON["neutral"]]
        activations = [compute_lambdas(s, model, tokenizer, layer, model_name, ft, system_prompt, instruction_in_prompt) for s in sentences]
        lambdas_concept["neutral"][entity] = activations
        
        for i, act in enumerate(activations):
            activations_records.append({
                "entity": entity,
                "persona": PERSON["neutral"][i] if concept.startswith('ruted_')==False else str(i),
                "activation": act[0].detach().float().cpu().numpy().ravel()
            })

    all_neutral_activations = [v for sub in lambdas_concept["neutral"].values() for v in sub]

    
    dot_products = {"neutral": {}}
    
    for c in CONCEPTS[concept]:  
        dot_products["neutral"][c] = [torch.dot(v[0].float(), v_gender).item() for v in lambdas_concept["neutral"][c]]

    dico_dot_products_mean = {"neutral": {}}
    dico_dot_products_std = {"neutral": {}}

    for c in CONCEPTS[concept]:
        dico_dot_products_mean["neutral"][c] = np.mean(dot_products["neutral"][c])
        dico_dot_products_std["neutral"][c] = np.std(dot_products["neutral"][c])
        
    
    return (dico_dot_products_mean, 
            dico_dot_products_std, 
            all_neutral_activations, 
            activations_records)


def save_dot_products(dico_mean: dict, dico_std: dict, concept: str, folder: str, layer: int, measure: str = "dot_products") -> None:
    """Save the dot product results to a CSV file

    Args:
        dico_mean (dict): dictionary of mean dot products
        dico_std (dict): dictionary of standard deviation of dot products
        concept (str): the concept being analyzed
        folder (str): the folder to save the CSV file in
        layer (int): the layer number
        measure (str): the measure type
    Returns:
        None
    """
    mean_values = dico_mean['neutral']
    std_values = dico_std['neutral']

    sorted_items = sorted(mean_values.items(), key=lambda x: x[1]) 

    concepts = [c for c, _ in sorted_items]
    means = [mean_values[c] for c in concepts]
    stds = [std_values[c] for c in concepts]

    df_dot = pd.DataFrame({
        'Concept': concepts,
        'Mean': means,
        'Std': stds
    })

    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, f"{measure}_{concept}_L{layer}.csv")
    
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_old = df_old[~df_old["Concept"].isin(df_dot["Concept"])]
        df_final = pd.concat([df_old, df_dot], ignore_index=True)
    else:
        df_final = df_dot

    df_final.to_csv(csv_path, index=False)

    


def compute_bias_scores(mean_values: dict, activations: list | None = None) -> dict:
    """Compute summary statistics of latent bias for a concept

    Args:
        mean_values (dict): mean dot products or cosine similarities for each concept
        activations (list | None): list of neutral activations for the concept
    Returns:
        dict: mean absolute dot product, standard deviation, and normalized standard deviation
    """
    values = np.array(list(mean_values.values()))
    mean_abs = np.mean(np.abs(values))
    std = np.std(values)

    if activations is not None :
        norms = [torch.norm(act[0].float()).item() for act in activations]
        mean_norm = np.mean(norms)
        std_normalized = std / mean_norm if mean_norm != 0 else 0
    else:
        std_normalized = None

    return {
        "mean_abs_dot": mean_abs,
        "std_dot": std,
        "std_norm_dot": std_normalized, 
    }



def main(args: argparse.Namespace) -> None:
    """Compute and save per-layer latent gender scores (projections onto the gender direction) for each concept

    Args:
        args (argparse.Namespace): parsed command-line arguments
    Returns:
        None
    """
    print("====== Dot Product Analysis ======")
    if args.use_model_ft_dpo and args.use_model_ft_sft:
        raise ValueError("Cannot use both DPO and LoRA fine-tuning at the same time.")
    

    ft = args.use_model_ft_dpo or args.use_model_ft_sft

    # Resolve the system-prompt key to its actual instruction text (None if "none"/unset).
    system_prompt = SYSTEM_PROMPTS.get(args.system_prompt_key, None)

    suffix_folder = get_suffix_folder(args.instruction_in_prompt, args.system_prompt_key, args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft)

    folder = ROOT / "results" / "dot_prod" / suffix_folder
    folder_dot_prod = folder / "dot_products"
    folder_activations = folder / "activations"
    os.makedirs(folder_dot_prod, exist_ok=True)
    os.makedirs(folder_activations, exist_ok=True)

    folder = str(folder)
    folder_dot_prod = str(folder_dot_prod)
    folder_activations = str(folder_activations)
    
    model, tokenizer = get_model(args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft)
    
    bias_scores_all_layers_dot_prod = []
    for i,layer in enumerate(args.list_layers):
        print(f"Processing layer {layer}...")
        v_gender, _ = compute_gender_direction(model, tokenizer, layer, args.model_name, ft, system_prompt=None, instruction_in_prompt=False)
        print("Gender direction computed.")
        for concept in args.list_concepts:
            print(f"Processing concept: {concept}...")
            (
                dico_dot_products_mean,
                dico_dot_products_std,
                neutral_activations, 
                activations_records
            ) = compute_dot_product(model, tokenizer, layer, concept, v_gender, args.model_name, ft, system_prompt=system_prompt, instruction_in_prompt=args.instruction_in_prompt)

            df_acts = pd.DataFrame([
                {
                    "entity": a["entity"],
                    "persona": a["persona"],
                    "activation": " ".join(map(str, a["activation"]))
                }
                for a in activations_records
            ])

            df_acts.to_csv(os.path.join(folder_activations, f"activations_{concept}_L{layer}.csv"), index=False)           

            save_dot_products(dico_dot_products_mean, dico_dot_products_std, concept, folder_dot_prod, layer, measure="dot_products")

            bias_score_dot_prod = compute_bias_scores(dico_dot_products_mean["neutral"],  activations=neutral_activations)
            bias_score_dot_prod["layer"] = layer
            bias_score_dot_prod["concept"] = concept
            bias_scores_all_layers_dot_prod.append(bias_score_dot_prod)
            
            


    df_scores_dot_prod = pd.DataFrame(bias_scores_all_layers_dot_prod)
    df_scores_dot_prod.to_csv(os.path.join(folder, "bias_scores_dot_products.csv"), index=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--use_model_ft_sft', type=str2bool, default=False)
    parser.add_argument('--use_model_ft_dpo', type=str2bool, default=False)
    parser.add_argument('--list_concepts', nargs='+',
                        default=['professions', 'colors', 'months'])
    parser.add_argument('--list_layers', nargs='+', default=[5], type=int)
    parser.add_argument('--instruction_in_prompt', type=str2bool, default=False,
                   help='Put instruction directly in prompt rather than system prompt')
    parser.add_argument('--system_prompt_key', type=str, default=None,
                        help='Type de system prompt à utiliser')
    args = parser.parse_args()
    main(args)