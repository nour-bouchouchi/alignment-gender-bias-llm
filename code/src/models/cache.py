from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import contextlib
import functools
from torch import Tensor


def block_name_suffix(module_type: str) -> str:
    """Get the suffix for the block name based on the module type

    Args:
        module_type (str): Type of the module (e.g., "o_proj", "down_proj")
    Returns:
        str: The corresponding suffix string
    """
    if module_type == "o_proj":
        return ".self_attn.o_proj"

    elif module_type == "down_proj":
        return ".mlp.down_proj"

    elif module_type == "block":
        return ""
    
    else:
        print(f"Invalid module_type '{module_type}'")
        return ""


def get_format_block_names(model_name: str, layer: int, ft: bool = False, module_type="block") -> str:
    """Get the formatted block name for a given layer in the model

    Args:
        model_name (str): Name of the model
        layer (int): Layer number
        ft (bool): Whether the model is fine-tuned
        module_type (str): Type of the module used to build the block name suffix
    Returns:
        str: Formatted block name as a string
    """
    if ft:
        base =  f"_orig_mod.base_model.model.model.layers.{layer}"
    else:
        base = f"model.layers.{layer}"
    
    base += block_name_suffix(module_type)    

    return base



def format_with_system_prompt(
    tokenizer: Any,
    text: Union[str, List[str]],
    system_prompt: str | None = None,
    instruction_in_prompt: bool = False,
    use_chat_template:bool= False,
):
    """Format the input text with a system prompt if provided

    Args:
        tokenizer (Any): The tokenizer for the model
        text (str | List[str]): The input text or list of texts to format
        system_prompt (str | None): Optional system prompt to use
        instruction_in_prompt (bool): Whether the instruction is in the prompt
        use_chat_template (bool): Whether the chat template is used
    Returns:
        str | List[str]: The formatted text or list of texts
    """
    if system_prompt == "none":
        system_prompt = None
    
    if not use_chat_template:
        if system_prompt is None:
            return text

        if instruction_in_prompt:
            if isinstance(text, list):
                return [f"{system_prompt}\n {t}" for t in text]
            
            return f"{system_prompt}\n {text}"

    if hasattr(tokenizer, 'apply_chat_template'):
        def _apply_template(t: str) -> str:
            """Apply the tokenizer chat template to a single text"""
            messages = []
            if system_prompt is not None and not instruction_in_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            content = f"{system_prompt}\n{t}" if (system_prompt and instruction_in_prompt) else t
            messages.append({"role": "user", "content": content})
            
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                print("!! Tokenizer corrupted !!")
                prefix = f"System: {system_prompt}\n" if system_prompt else ""
                return f"{prefix}User: {t}\nAssistant:"

        if isinstance(text, list):
            return [_apply_template(t) for t in text]
        return _apply_template(text)
    
    return text 

def run_with_cache_hf(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str] | str,
    layer: int,
    model_name: str,
    ft: bool,
    system_prompt: str | None = None,
    instruction_in_prompt: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run a forward pass and cache the residual-stream activations at a given layer

    Args:
        model (torch.nn.Module): The model to run the forward pass on
        tokenizer (Any): The tokenizer for the model
        texts (List[str] | str): The input text or list of texts
        layer (int): Layer number to cache activations from
        model_name (str): Name of the model
        ft (bool): Whether the model is fine-tuned
        system_prompt (str | None): Optional system prompt to use
        instruction_in_prompt (bool): Whether the instruction is in the prompt
    Returns:
        Dict[str, torch.Tensor]: The cached residual-stream activations
    """

    activations: Dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        """Cache the module output as the residual-stream activation"""
        if isinstance(output, tuple):
            output = output[0]
        activations["hook_resid_post"] = output.detach()

    block_name = get_format_block_names(model_name, layer, ft)
    named = dict(model.named_modules())
    if block_name not in named:
        raise KeyError(f"Module not found: {block_name}")
    handle = named[block_name].register_forward_hook(hook_fn)

    formatted_texts = format_with_system_prompt(
        tokenizer, texts, system_prompt, instruction_in_prompt
    )
    inputs = tokenizer(
        formatted_texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()
    return activations

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]] = [],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]] = [],
    **kwargs
):
    """Context manager for temporarily adding hooks to modules

    Args:
        module_forward_pre_hooks (List[Tuple[torch.nn.Module, Callable]]): List of tuples (module, hook) for forward pre-hooks
        module_forward_hooks (List[Tuple[torch.nn.Module, Callable]]): List of tuples (module, hook) for forward hooks
    Yields:
        None
    """
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_direction_ablation_input_pre_hook(direction: Tensor):
    """Get a forward pre-hook function for direction ablation on input activations

    Args:
        direction (Tensor): The direction vector to ablate
    Returns:
        Callable: The hook function for ablation
    """
    def hook_fn(module, input):
        """Ablate the input activation along the direction"""
        nonlocal direction

        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)

        # Projection
        ablated = activation - (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (ablated, *input[1:])
        else:
            return ablated

    return hook_fn


def get_steering_hook(direction: Tensor) :
    """Get a forward hook function for steering the activations in a given direction

    Args:
        direction (Tensor): The direction vector for steering
    Returns:
        Callable: The hook function for steering
    """
    def hook_fn(module, input, output):
        """Steer the output activation along the direction"""
        if isinstance(output, tuple):
            activation = output[0]
            rest = output[1:]
        else:
            activation = output
            rest = None

        dir_cast = direction.to(device=activation.device, dtype=activation.dtype)

        # projection
        proj = (activation * dir_cast).sum(dim=-1, keepdim=True)

        steered = activation + 1.0 * proj * dir_cast

        if rest is not None:
            return (steered, *rest)
        return steered

    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    """Get a forward hook function for direction ablation on output activations

    Args:
        direction (Tensor): The direction vector to ablate
    Returns:
        Callable: The hook function for ablation
    """
    def hook_fn(module, input, output):
        """Ablate the output activation along the direction"""
        nonlocal direction

        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)

        ablated = activation - (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (ablated, *output[1:])
        else:
            return ablated

    return hook_fn
