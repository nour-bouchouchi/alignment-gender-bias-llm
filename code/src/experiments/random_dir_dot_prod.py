import os
import argparse
import torch
import numpy as np

from pathlib import Path

from models.hf import get_model
from utils.cli import str2bool, get_suffix_folder, get_sentence_prompt
from constants import PERSON, CONCEPTS, SCHEMA, SYSTEM_PROMPTS
from utils.directions import compute_lambdas

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

import json


def _safe_load_json(path: Path) -> dict:
    """Load a JSON file, returning an empty dict if it does not exist

    Args:
        path (Path): path to the JSON file
    Returns:
        dict: the parsed JSON content, or an empty dict if the file is missing
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_save_json(path: Path, data) -> None:
    """Write data to a path as JSON atomically via a temporary file and replace

    Args:
        path (Path): destination path for the JSON file
        data (Any): the data to serialize as JSON
    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _merge_layer_dict(existing, incoming, overwrite_existing: bool):
    """Merge two layer-keyed stats dictionaries

    Args:
        existing (dict): current mapping of layer to stats dict
        incoming (dict): new mapping of layer to stats dict to merge in
        overwrite_existing (bool): whether to overwrite layers already present
    Returns:
        dict: the merged mapping of layer to stats dict
    """
    out = dict(existing) if existing is not None else {}
    for layer_k, stats in (incoming or {}).items():
        lk = str(layer_k)
        if lk in out and (not overwrite_existing):
            continue
        out[lk] = stats
    return out


def _merge_concept_dict(existing, incoming, overwrite_existing: bool):
    """Merge two concept-keyed nested stats dictionaries

    Args:
        existing (dict): current mapping of concept to layer-stats dict
        incoming (dict): new mapping of concept to layer-stats dict to merge in
        overwrite_existing (bool): whether to overwrite layers already present
    Returns:
        dict: the merged mapping of concept to layer-stats dict
    """
    out = dict(existing) if existing is not None else {}
    for concept, layers_dict in (incoming or {}).items():
        if concept not in out:
            out[concept] = {}
        out[concept] = _merge_layer_dict(out[concept], layers_dict, overwrite_existing=overwrite_existing)
    return out


def upsert_random_dir_global_json(
    json_path: Path,
    mode_key: str,
    model_name: str,
    results_global,
    results_concept,
    overwrite_existing: bool = False,
):
    """Upsert random-direction global and per-concept results into the global JSON

    Args:
        json_path (Path): path to the global JSON file
        mode_key (str): top-level key identifying the run mode
        model_name (str): the name of the model
        results_global (dict): mapping of layer to global stats
        results_concept (dict): mapping of concept to layer-stats dict
        overwrite_existing (bool): whether to overwrite entries already present
    Returns:
        dict: the full updated JSON data structure
    """
    data = _safe_load_json(json_path)

    if mode_key not in data:
        data[mode_key] = {}
    if model_name not in data[mode_key]:
        data[mode_key][model_name] = {}

    incoming_global = {str(layer): stats for layer, stats in (results_global or {}).items()}
    incoming_concept = {
        concept: {str(layer): stats for layer, stats in layers.items()}
        for concept, layers in (results_concept or {}).items()
    }

    existing_block = data[mode_key][model_name]
    existing_global = existing_block.get("global", {})
    existing_concept = existing_block.get("concept", {})

    merged_global = _merge_layer_dict(existing_global, incoming_global, overwrite_existing=overwrite_existing)
    merged_concept = _merge_concept_dict(existing_concept, incoming_concept, overwrite_existing=overwrite_existing)

    data[mode_key][model_name] = {
        "global": merged_global,
        "concept": merged_concept,
    }

    _atomic_save_json(json_path, data)
    return data


def compute_random_dir_layerwise_global_dict(
    model,
    tokenizer,
    model_name: str,
    ft: bool,
    list_layers,
    list_concepts,
    system_prompt=None,
    instruction_in_prompt: bool = False,
    n_dirs: int = 200,
    base_seed: int = 42,
    normalize_by_norms: bool = True,
):
    """Compute layerwise random-direction polarization stats globally and per concept

    Args:
        model (Any): the language model
        tokenizer (Any): the tokenizer for the model
        model_name (str): the name of the model
        ft (bool): whether the model is fine-tuned
        list_layers (list): the layers to process
        list_concepts (list): the concepts to process
        system_prompt (str | None): optional system prompt to use
        instruction_in_prompt (bool): whether the instruction is in the prompt
        n_dirs (int): number of random directions to sample
        base_seed (int): base seed for random direction generation
        normalize_by_norms (bool): whether to normalize scores by mean activation norm
    Returns:
        tuple: global layer stats and per-concept layer stats
    """
    results_global = {}
    results_concept = {}

    for layer in list_layers:
        print(f"L{layer}")
        items_to_tensors_global = {}
        all_neutral_global = []

        for concept in list_concepts:
            if concept not in CONCEPTS:
                print(f"[WARN] concept '{concept}' missing from CONCEPTS, skip.")
                continue
            if concept not in SCHEMA:
                print(f"[WARN] concept '{concept}' missing from SCHEMA, skip.")
                continue

            items_to_tensors_C = {}
            all_neutral_C = []

            for entity in CONCEPTS[concept]:
                if concept.startswith('ruted_'):
                    sentences = [get_sentence_prompt(concept, "", entity, idx=i, dot=True) for i in range(len(PERSON["neutral"]))]
                else : 
                    sentences = [get_sentence_prompt(concept, persona, entity, dot=True) for persona in PERSON["neutral"]]
                acts = [
                    compute_lambdas(
                        s, model, tokenizer, layer,
                        model_name, ft,
                        system_prompt=system_prompt,
                        instruction_in_prompt=instruction_in_prompt
                    )
                    for s in sentences
                ]
                tensors = [a[0].detach().to(device).float() for a in acts if a is not None]
                if not tensors:
                    continue

                # global
                items_to_tensors_global[str(entity)] = tensors
                all_neutral_global.extend(tensors)

                # concept
                items_to_tensors_C[str(entity)] = tensors
                all_neutral_C.extend(tensors)

            if concept not in results_concept:
                results_concept[concept] = {}

            if not all_neutral_C or not items_to_tensors_C:
                results_concept[concept][layer] = {"mean": np.nan, "p025": np.nan, "p50": np.nan, "p975": np.nan}
            else:
                dim = all_neutral_C[0].numel()

                if normalize_by_norms:
                    denom = float(torch.mean(torch.stack([t.norm() for t in all_neutral_C])).detach().cpu())
                    if denom <= 0:
                        denom = np.nan
                else:
                    denom = 1.0

                seed = np.abs(hash((model_name, int(ft), int(layer), str(concept), int(base_seed)))) % (2**32)
                rng = np.random.default_rng(seed)

                vals = []
                for _ in range(n_dirs):
                    v = rng.normal(size=(dim,)).astype(np.float32)
                    n = np.linalg.norm(v)
                    if n == 0:
                        v[0] = 1.0
                        n = 1.0
                    v_t = torch.from_numpy(v / n).to(device)

                    mu_vals = []
                    for _, X_list in items_to_tensors_C.items():
                        X = torch.stack(X_list, dim=0)
                        mu = (X @ v_t).mean()
                        mu_vals.append(float(mu.detach().cpu()))

                    if mu_vals:
                        numer = float(np.std(mu_vals, ddof=0))
                        score = numer / denom if (normalize_by_norms and denom and not np.isnan(denom)) else numer
                        vals.append(score)

                arr = np.asarray(vals, dtype=float)
                if arr.size == 0 or np.all(np.isnan(arr)):
                    results_concept[concept][layer] = {"mean": np.nan, "p025": np.nan, "p50": np.nan, "p975": np.nan}
                else:
                    results_concept[concept][layer] = {
                        "mean": float(np.nanmean(arr)),
                        "p025": float(np.nanpercentile(arr, 2.5)),
                        "p50":  float(np.nanpercentile(arr, 50.0)),
                        "p975": float(np.nanpercentile(arr, 97.5)),
                    }

        # stats random dir GLOBAL (for all concepts) for this layer
        if not all_neutral_global or not items_to_tensors_global:
            results_global[layer] = {"mean": np.nan, "p025": np.nan, "p50": np.nan, "p975": np.nan}
        else:
            dim = all_neutral_global[0].numel()

            if normalize_by_norms:
                denom_g = float(torch.mean(torch.stack([t.norm() for t in all_neutral_global])).detach().cpu())
                if denom_g <= 0:
                    denom_g = np.nan
            else:
                denom_g = 1.0

            seed = np.abs(hash((model_name, int(ft), int(layer), "GLOBAL", int(base_seed)))) % (2**32)
            rng = np.random.default_rng(seed)

            vals = []
            for _ in range(n_dirs):
                v = rng.normal(size=(dim,)).astype(np.float32)
                n = np.linalg.norm(v)
                if n == 0:
                    v[0] = 1.0
                    n = 1.0
                v_t = torch.from_numpy(v / n).to(device)

                mu_vals = []
                for _, X_list in items_to_tensors_global.items():
                    X = torch.stack(X_list, dim=0)
                    mu = (X @ v_t).mean()
                    mu_vals.append(float(mu.detach().cpu()))

                if mu_vals:
                    numer = float(np.std(mu_vals, ddof=0))
                    score = numer / denom_g if (normalize_by_norms and denom_g and not np.isnan(denom_g)) else numer
                    vals.append(score)

            arr = np.asarray(vals, dtype=float)
            if arr.size == 0 or np.all(np.isnan(arr)):
                results_global[layer] = {"mean": np.nan, "p025": np.nan, "p50": np.nan, "p975": np.nan}
            else:
                results_global[layer] = {
                    "mean": float(np.nanmean(arr)),
                    "p025": float(np.nanpercentile(arr, 2.5)),
                    "p50":  float(np.nanpercentile(arr, 50.0)),
                    "p975": float(np.nanpercentile(arr, 97.5)),
                }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results_global, results_concept



def main(args: argparse.Namespace) -> None:
    """Compute the random-direction reference distribution of latent polarization and store it in the global JSON

    Args:
        args (argparse.Namespace): parsed command-line arguments
    Returns:
        None
    """
    print("====== Random Direction (GLOBAL JSON) ======")

    if args.use_model_ft_dpo and args.use_model_ft_sft:
        raise ValueError("Cannot use both DPO and LoRA fine-tuning at the same time.")

    ft = args.use_model_ft_dpo or args.use_model_ft_sft

    # Resolve the system-prompt key to its actual instruction text (None if "none"/unset)
    system_prompt = SYSTEM_PROMPTS.get(args.system_prompt_key, None)

    suffix_folder = get_suffix_folder(
        args.instruction_in_prompt,
        args.system_prompt_key,
        args.model_name,
        args.use_model_ft_dpo,
        args.use_model_ft_sft
    )

    folder = ROOT / "results" / "dot_prod" 
    os.makedirs(folder, exist_ok=True)

    json_path = folder / "random_dir_GLOBAL.json"

    print(f"[INFO] output json: {json_path}")
    print(f"[INFO] mode_key={args.mode_key} | overwrite_existing={args.overwrite_existing}")

    model, tokenizer = get_model(args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft)
    model.eval()

    results_global, results_concept = compute_random_dir_layerwise_global_dict(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        ft=ft,
        list_layers=args.list_layers,
        list_concepts=args.list_concepts,
        system_prompt=system_prompt,
        instruction_in_prompt=args.instruction_in_prompt,
        n_dirs=args.n_dirs,
        base_seed=args.base_seed,
        normalize_by_norms=args.normalize_by_norms,
    )

    upsert_random_dir_global_json(
        json_path=json_path,
        mode_key=args.mode_key,
        model_name=args.model_name,
        results_global=results_global,
        results_concept=results_concept,
        overwrite_existing=args.overwrite_existing,
    )

    print("[OK] Done.")


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


    parser.add_argument('--mode_key', type=str, required=True,
                        help="Clé dans random_dir_GLOBAL.json (ex: base, sft, instruction_in_prompt_general, ...)")
    parser.add_argument('--overwrite_existing', type=str2bool, default=False,
                        help="Si true, remplace les (concept, layer) déjà présents; sinon, skip ceux qui existent.")
    parser.add_argument('--n_dirs', type=int, default=200)
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--normalize_by_norms', type=str2bool, default=True)

    args = parser.parse_args()
    main(args)
