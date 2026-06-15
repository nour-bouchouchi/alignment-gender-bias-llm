import torch
import argparse
import json
import os
from typing import Any, Dict, List
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from models.hf import get_model                    
from models.cache import format_with_system_prompt 
from utils.cli import str2bool, get_suffix_folder, get_sentence_prompt                  
from constants import PERSON, CONCEPTS, SYSTEM_PROMPTS
from models.cache import add_hooks, get_direction_ablation_input_pre_hook, get_format_block_names, get_steering_hook

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



def generate_only(model: Any, tokenizer: Any, persona: str, entity: str, concept: str = "professions", system_prompt: str | None = None, instruction_in_prompt: bool = False,
                  dict_ablation_directions: Dict[int, torch.Tensor] | None = None, ablation_layer: int | None = None, ablation_direction: torch.Tensor | None = None,
                  steering_layer: int | None = None, steering_direction: torch.Tensor | None = None, module_type: str | None = None, model_name: str | None = None,
                  ft: bool = False, idx: int | None = None, use_chat_template: bool = False) -> str:
    """Generate text for a given neutral person and entity

    Args:
        model (Any): the language model
        tokenizer (Any): the tokenizer for the model
        persona (str): the persona to use
        entity (str): the entity to include in the prompt
        concept (str): the concept to analyze (e.g. professions)
        system_prompt (str | None): optional system prompt to use
        instruction_in_prompt (bool): whether the instruction is placed in the prompt
        dict_ablation_directions (Dict[int, torch.Tensor] | None): per-layer direction vectors for ablation
        ablation_layer (int | None): layer number for direction ablation
        ablation_direction (torch.Tensor | None): direction vector for ablation
        steering_layer (int | None): layer number for steering
        steering_direction (torch.Tensor | None): direction vector for steering
        module_type (str | None): type of module targeted for steering
        model_name (str | None): the name of the model
        ft (bool): whether the model is fine-tuned
        idx (int | None): index used to select the prompt for ruted concepts
        use_chat_template (bool): whether the chat template is used
    Returns:
        str: the generated text from the model
    """

    prompt = get_sentence_prompt(concept, persona, entity, idx=idx) 

    formatted_prompt = format_with_system_prompt(tokenizer, prompt, system_prompt, instruction_in_prompt=instruction_in_prompt, use_chat_template=use_chat_template)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": 100 if concept.startswith("ruted_") else 50,
        "do_sample": True,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        if dict_ablation_directions is not None : 
            modules = dict(model.named_modules())
            hooks = []
            for layer in sorted(dict_ablation_directions.keys()):
                direction = dict_ablation_directions[layer]
                block_name = get_format_block_names(model_name, layer, ft=ft)
                module = modules[block_name]
                hooks.append((module, get_direction_ablation_input_pre_hook(direction)))
            with add_hooks(module_forward_pre_hooks=hooks):
                outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)
                
        elif ablation_layer is not None and ablation_direction is not None:
            block_name = get_format_block_names(model_name, ablation_layer, ft=ft)
            module = dict(model.named_modules())[block_name]
            with add_hooks(module_forward_pre_hooks=[(module, get_direction_ablation_input_pre_hook(ablation_direction))]):
                outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)
        
        elif steering_layer is not None and steering_direction is not None:
            block_name = get_format_block_names(model_name, steering_layer, ft=ft, module_type=module_type)
            module = dict(model.named_modules())[block_name]
            with add_hooks(module_forward_hooks=[(module, get_steering_hook(steering_direction))]):
                outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)
        
        else:
            outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)


    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    idx = generated_text.find("user")
    return generated_text[idx:] if idx != -1 else generated_text


def generate_all(model: Any, tokenizer: Any, folder: str, overwrite: bool, concept: str = "professions", type_persona: str = "neutral", system_prompt: str | None = None, instruction_in_prompt: bool = False,
                 dict_ablation_directions: Dict[int, torch.Tensor] | None = None, ablation_layer: int | None = None, ablation_direction: torch.Tensor | None = None,
                 steering_layer: int | None = None, steering_direction: torch.Tensor | None = None, module_type: str | None = None, model_name: str | None = None, ft: bool = False, nb_generations: int = 10) -> None:
    """Generate texts for all entities in a concept and save to a JSON file

    Args:
        model (Any): the language model
        tokenizer (Any): the tokenizer for the model
        folder (str): folder to save the generated texts
        overwrite (bool): whether to overwrite existing files
        concept (str): the concept to analyze (e.g. professions)
        type_persona (str): the persona type to iterate over
        system_prompt (str | None): optional system prompt to use
        instruction_in_prompt (bool): whether the instruction is placed in the prompt
        dict_ablation_directions (Dict[int, torch.Tensor] | None): per-layer direction vectors for ablation
        ablation_layer (int | None): layer number for direction ablation
        ablation_direction (torch.Tensor | None): direction vector for ablation
        steering_layer (int | None): layer number for steering
        steering_direction (torch.Tensor | None): direction vector for steering
        module_type (str | None): type of module targeted for steering
        model_name (str | None): the name of the model
        ft (bool): whether the model is fine-tuned
        nb_generations (int): number of generations per persona and entity
    Returns:
        None
    """
    if not os.path.exists(f"{folder}/{concept}.json") or overwrite:
        generation_data = {}
        for entity in tqdm(CONCEPTS[concept]):
            generation_data[entity] = {}
            for i, persona in enumerate(PERSON[type_persona]):
                idx=None
                if concept.startswith('ruted_'):
                    persona = str(i)
                    idx = i # No persona for ruted prompts -> use index
                generation_data[entity][persona] = []
                for _ in range(nb_generations):
                    text = generate_only(model, tokenizer, persona, entity, concept=concept, system_prompt=system_prompt, instruction_in_prompt=instruction_in_prompt, 
                                         dict_ablation_directions=dict_ablation_directions, ablation_layer=ablation_layer, ablation_direction=ablation_direction, 
                                         steering_layer=steering_layer, steering_direction=steering_direction,
                                         module_type=module_type, model_name=model_name, ft=ft, idx=idx, use_chat_template=False)
                    generation_data[entity][persona].append(text)
                
        with open(f"{folder}/{concept}.json", "w", encoding="utf-8") as f:
            json.dump(generation_data, f, indent=2, ensure_ascii=False)


def generation(model: Any, tokenizer: Any, folder: str, overwrite: bool, list_concepts: List[str], type_persona: str = "neutral", system_prompt: str | None = None, instruction_in_prompt: bool = False, ft: bool = False, nb_generations: int = 10) -> None:
    """Generate and save completions for every concept in list_concepts

    Args:
        model (Any): the language model
        tokenizer (Any): the tokenizer for the model
        folder (str): folder to save the generated texts
        overwrite (bool): whether to overwrite existing files
        list_concepts (List[str]): the concepts to generate completions for
        type_persona (str): the persona type to iterate over
        system_prompt (str | None): optional system prompt to use
        instruction_in_prompt (bool): whether the instruction is placed in the prompt
        ft (bool): whether the model is fine-tuned
        nb_generations (int): number of generations per persona and entity
    Returns:
        None
    """
    for concept in list_concepts:
        print(f"Generating {concept} examples...")
        generate_all(model, tokenizer, folder, overwrite, concept=concept, type_persona=type_persona, system_prompt=system_prompt, instruction_in_prompt=instruction_in_prompt, 
                     model_name=args.model_name, ft=ft, nb_generations=nb_generations)
        print(f"Generation for {concept} completed.")
    
    
    

def main(args: argparse.Namespace) -> None:
    print("====== Text Generation ======")
    if args.use_model_ft_dpo and args.use_model_ft_sft:
        raise ValueError("Cannot use both DPO and LoRA fine-tuning at the same time.")
    
    system_prompt = SYSTEM_PROMPTS.get(args.system_prompt_key, None) if hasattr(args, 'system_prompt_key') else None
     
     
    model, tokenizer = get_model(args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft, args.lora_scale)


    suffix_folder = get_suffix_folder(args.instruction_in_prompt, args.system_prompt_key, args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft, args.lora_scale)

    folder = ROOT / "results" / f"generation{'_'+args.person_type if args.person_type != 'neutral' else ''}" / suffix_folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    ft = args.use_model_ft_dpo or args.use_model_ft_sft
    generation(model, tokenizer, str(folder), args.overwrite, args.list_concepts, type_persona=args.person_type, system_prompt=system_prompt, 
                instruction_in_prompt=args.instruction_in_prompt, ft=ft, nb_generations=args.nb_generations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--use_model_ft_dpo', type=str2bool, default=False)
    parser.add_argument('--use_model_ft_sft', type=str2bool, default=False)
    parser.add_argument('--person_type', type=str, default="neutral", choices=["neutral", "M", "F"])
    parser.add_argument('--lora_scale', type=float, default=None)
    parser.add_argument('--overwrite', type=str2bool, default=True)
    parser.add_argument('--nb_generations', type=int, default=10)
    parser.add_argument('--list_concepts', nargs='+',
                        default=['professions', 'colors', 'months'])
    parser.add_argument('--system_prompt_key', type=str, default="none",
                        help='Type de system prompt à utiliser')
    parser.add_argument('--instruction_in_prompt', type=str2bool, default=False,
                   help='Put instruction directly in prompt rather than system prompt')
    args = parser.parse_args()
    main(args)