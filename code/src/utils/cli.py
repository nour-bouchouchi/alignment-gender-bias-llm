import argparse
from typing import Union
from constants import SCHEMA

def str2bool(value: Union[str, bool]) -> bool:
    """Parse a string (or bool) into a boolean, for use as an argparse type

    Args:
        value (Union[str, bool]): value to coerce into a boolean
    Returns:
        bool: parsed boolean value
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes', 'y'}:
        return True
    elif value.lower() in {'false', '0', 'no', 'n'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got {value}.")

def get_sentence_prompt(concept: str, persona: str, entity: str, idx: int | None = None, dot: bool = False) -> str:
    """Build the neutral prompt string for a (concept, persona, entity) triple

    Args:
        concept (str): schema concept key to look up the template
        persona (str): persona prefix prepended to the sentence
        entity (str): entity inserted into the template
        idx (int | None): index into the template list for ruted concepts
        dot (bool): whether to end the sentence with a period instead of a comma
    Returns:
        str: formatted prompt sentence
    """
    if concept not in SCHEMA:
        raise ValueError(f"Concept '{concept}' not found in SCHEMA.")

    if concept.startswith('ruted_'):
        entity_formatted = f'an {entity}' if entity[0].lower() in ['a','e','i','o','u'] else f'a {entity}'
        return f"{SCHEMA[concept][idx].format(entity_formatted)}"

    if concept=="professions" and entity[0].lower() in ['a','e','i','o','u']:
        return f"{persona}{SCHEMA[concept].replace('a ','an ')}{entity.lower()}{'.' if dot else ', '}"

    return f"{persona}{SCHEMA[concept]}{entity.lower()}{'.' if dot else ', '}"

def get_suffix_folder(instruction_in_prompt: bool, system_prompt_key: str | None, model_name: str, use_model_ft_dpo: bool = False, use_model_ft_sft: bool = False, lora_scale: float | None = None) -> str:
    """Build the result-folder name encoding the model and its fine-tuning/instruction/scale configuration

    Args:
        instruction_in_prompt (bool): whether the instruction is placed in the prompt
        system_prompt_key (str | None): key identifying the system prompt, or None
        model_name (str): base model name
        use_model_ft_dpo (bool): whether the DPO fine-tuned model is used
        use_model_ft_sft (bool): whether the SFT fine-tuned model is used
        lora_scale (float | None): LoRA scaling factor, or None
    Returns:
        str: result-folder name suffix combined with the model name
    """
    suffix = '_lora_dpo' if use_model_ft_dpo else '_lora_sft' if use_model_ft_sft else ''
    
    if system_prompt_key != None and system_prompt_key!="none":
        if instruction_in_prompt:
            suffix += f"_instruction_in_prompt_{system_prompt_key}"
        else:
            suffix += f"_instruction_{system_prompt_key}"
    
    if lora_scale is not None:
        scale_str = f"{lora_scale:.6g}"  # "0.1"
        suffix += f"_scale_{scale_str}"

    return f"{model_name}{suffix}"



