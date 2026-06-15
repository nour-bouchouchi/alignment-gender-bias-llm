import os
import shutil
from pathlib import Path
import argparse

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))

def collect_plots(overwrite: bool = False, symlink: bool = False, model: str | None = None) -> None:
    """Copy (or symlink) every .png under results/ into results/plots/, preserving the relative structure

    Args:
        overwrite (bool): whether to overwrite files that already exist in plots/
        symlink (bool): whether to create symbolic links instead of copying
        model (str | None): only keep PNGs whose path contains this model, or None for all
    Returns:
        None
    """
    results_dir = ROOT / "results"
    plots_root = results_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    for png in results_dir.rglob("*.png"):
        try:
            png.relative_to(plots_root)
            continue
        except ValueError:
            pass

        if model and model not in str(png):
            continue

        rel = png.relative_to(results_dir)
        dst = plots_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            if not overwrite:
                continue
            dst.unlink()

        if symlink:
            try:
                dst.symlink_to(png)
            except FileExistsError:
                pass
        else:
            shutil.copy2(png, dst)

def main() -> None:
    """CLI entry point that parses options and runs plot collection

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Écraser les fichiers déjà existants dans plots/")
    parser.add_argument("--symlink", action="store_true", help="Créer des liens symboliques au lieu de copier")
    parser.add_argument("--model", type=str, default=None, help="Filtrer uniquement les PNG dont le chemin contient ce modèle")
    args = parser.parse_args()

    collect_plots(overwrite=args.overwrite, symlink=args.symlink, model=args.model)
    print("Plots collected in: ", ROOT / "results" / "plots")

if __name__ == "__main__":
    main()
