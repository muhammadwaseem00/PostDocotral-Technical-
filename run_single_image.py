"""
Task 3: CLI to run integrated pipeline on one image.
Usage: python run_single_image.py --image path.png [--vlm] [--output result.json]
"""
import argparse
import json
from pathlib import Path

from inference_pipeline import load_pipeline, run_inference
import os
os.environ["HF_TOKEN"] = "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"


def main():
    ap = argparse.ArgumentParser(description="Task 3: CNN + optional VLM on one chest X-ray image")
    ap.add_argument("--image", required=True, help="Path to chest X-ray image")
    ap.add_argument("--vlm", action="store_true", help="Also run VLM to generate text report")
    ap.add_argument("--output", "-o", default=None, help="Save result JSON to this path")
    ap.add_argument("--prompt", default=None, help="Custom prompt for VLM (default: clinical)")
    args = ap.parse_args()

    path = Path(args.image)
    if not path.exists():
        raise SystemExit(f"Image not found: {path}")

    print("Loading pipeline (CNN" + (" + VLM" if args.vlm else "") + ")...")
    pipe = load_pipeline(use_vlm=args.vlm)
    print("Running inference...")
    result = run_inference(pipe, str(path), run_vlm=args.vlm, vlm_prompt=args.prompt or None)

    print(f"CNN: {result['cnn_pred']} (prob={result['cnn_prob']:.4f})")
    if "vlm_report" in result:
        print("VLM report (excerpt):", result["vlm_report"][:200] + "..." if len(result["vlm_report"]) > 200 else result["vlm_report"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path.resolve()}")

    return result


if __name__ == "__main__":
    main()
