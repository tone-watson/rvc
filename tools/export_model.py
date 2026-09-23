#!/usr/bin/env python3
"""Export trained RVC model to ~/models/rvc/voice/<name>/"""

import argparse
import shutil
import sys
from pathlib import Path

RVC_DIR = Path("/srv/farm/code/rvc")
WEIGHTS_DIR = RVC_DIR / "assets/weights"
LOGS_DIR = RVC_DIR / "logs"
OUTPUT_BASE = Path.home() / "models/rvc/voice"


def find_latest_pth(model_name: str) -> Path | None:
    """Find the latest .pth file for a model (highest epoch)."""
    candidates = list(WEIGHTS_DIR.glob(f"{model_name}*.pth"))
    if not candidates:
        return None

    # Sort by epoch number (e.g., charm-v2_e180_s3600.pth)
    def get_epoch(p: Path) -> int:
        name = p.stem
        if "_e" in name:
            try:
                return int(name.split("_e")[1].split("_")[0])
            except (ValueError, IndexError):
                return 0
        return 0

    return max(candidates, key=get_epoch)


def find_index(model_name: str) -> Path | None:
    """Find the added index file for a model."""
    model_log_dir = LOGS_DIR / model_name
    if not model_log_dir.exists():
        return None

    candidates = list(model_log_dir.glob("added_*.index"))
    if candidates:
        return candidates[0]
    return None


def list_models():
    """List available trained models."""
    models = set()

    # Check weights directory
    for pth in WEIGHTS_DIR.glob("*.pth"):
        name = pth.stem.split("_e")[0]  # Remove epoch suffix
        models.add(name)

    if models:
        print("Available models:")
        for m in sorted(models):
            print(f"  - {m}")
    else:
        print("No trained models found in assets/weights/")


def export_model(model_name: str, output_name: str = None):
    """Export model files to ~/models/rvc/voice/<name>/"""
    output_name = output_name or model_name

    # Find files
    pth_file = find_latest_pth(model_name)
    index_file = find_index(model_name)

    if not pth_file:
        print(f"Error: No .pth file found for '{model_name}'")
        print("Available models:")
        list_models()
        sys.exit(1)

    if not index_file:
        print(f"Warning: No .index file found for '{model_name}'")
        print("You may need to run 'Train feature index' in the UI first.")

    # Create output directory
    output_dir = OUTPUT_BASE / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"Exporting '{model_name}' to {output_dir}/")

    dest_pth = output_dir / f"{output_name}.pth"
    shutil.copy2(pth_file, dest_pth)
    print(f"  ✓ {pth_file.name} → {dest_pth.name}")

    if index_file:
        dest_index = output_dir / f"{output_name}.index"
        shutil.copy2(index_file, dest_index)
        print(f"  ✓ {index_file.name} → {dest_index.name}")

    print(f"\nDone! Model exported to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export trained RVC model")
    parser.add_argument("model", nargs="?", help="Model name to export (e.g., charm-v2)")
    parser.add_argument("-o", "--output", help="Output folder name (defaults to model name)")
    parser.add_argument("-l", "--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list or not args.model:
        list_models()
        if not args.model and not args.list:
            print("\nUsage: rvc_export.py <model-name> [-o output-name]")
        sys.exit(0)

    export_model(args.model, args.output)


if __name__ == "__main__":
    main()
