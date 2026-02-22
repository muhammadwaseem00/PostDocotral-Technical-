"""
Read all .pth model files, evaluate on test set, and plot combined ROC comparison.
Run from chleng/ directory. Searches for .pth files in current dir and parent.
"""
import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc

from config import Config
from dataset import get_dataloaders
from model import get_model
from models import MODEL_REGISTRY
from utils import calculate_metrics


def find_pth_files(search_dirs=None):
    """Find all .pth checkpoint files, excluding venv/site-packages."""
    if search_dirs is None:
        search_dirs = [Path(".").resolve(), Path(".").resolve().parent]
    found = []
    exclude = {"venv", "site-packages", "__pycache__", ".git"}
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.rglob("*.pth"):
            parts = f.parts
            if any(x in parts for x in exclude):
                continue
            found.append(f)
    return list(set(found))


def infer_model_name_from_path(pth_path):
    """Try to infer model name from filename, e.g. resnet18_pneumonia.pth -> resnet18."""
    stem = Path(pth_path).stem.lower()
    # Prefer exact match or prefix match (longest first)
    for name in sorted(MODEL_REGISTRY.keys(), key=len, reverse=True):
        if stem == name or stem.startswith(name + "_"):
            return name
    # Common pattern: {model}_pneumonia
    match = re.match(r"^([a-z0-9_]+)_(?:pneumonia|model|checkpoint)?$", stem)
    if match:
        cand = match.group(1)
        if cand in MODEL_REGISTRY:
            return cand
    return None


def load_and_evaluate(pth_path, model_name, test_loader):
    """Load model from .pth and return (y_true, y_prob, metrics)."""
    Config.model_name = model_name
    model = get_model().to(Config.device)
    state = torch.load(pth_path, map_location=Config.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(Config.device)
            labels = labels.float().squeeze(-1).to(Config.device)
            outputs = model(images).squeeze()
            all_logits.append(outputs)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    y_prob = torch.sigmoid(all_logits).cpu().numpy()
    y_true = all_labels.cpu().numpy().ravel()
    metrics = calculate_metrics(all_labels, all_logits)
    return y_true, y_prob, metrics


def main():
    print("Discovering .pth files...")
    script_dir = Path(__file__).resolve().parent
    pth_files = find_pth_files([script_dir, script_dir.parent])
    if not pth_files:
        print("No .pth files found. Using default model paths in script dir.")
        script_dir = Path(__file__).resolve().parent
        pth_files = [
            script_dir / f"{name}_pneumonia.pth"
            for name in ["resnet18", "resnet34", "resnet50", "efficientnet_b0"]
            if (script_dir / f"{name}_pneumonia.pth").exists()
        ]
        if not pth_files:
            print("No models found. Train models first (e.g. resnet18_pneumonia.pth).")
            return

    print("Loading test data...")
    _, _, test_loader = get_dataloaders(Config.batch_size, Config.num_workers)

    results = {}
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pth_files), 1)))

    for idx, pth_path in enumerate(sorted(pth_files, key=lambda p: p.name)):
        pth_path = Path(pth_path)
        if not pth_path.exists():
            print(f"  Skip (not found): {pth_path}")
            continue

        model_name = infer_model_name_from_path(pth_path)
        if model_name is None:
            print(f"  Skip (unknown architecture): {pth_path.name}")
            continue

        print(f"\nEvaluating {pth_path.name} (model: {model_name})...")
        try:
            y_true, y_prob, metrics = load_and_evaluate(pth_path, model_name, test_loader)
        except Exception as e:
            print(f"  Error loading {pth_path.name}: {e}")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        label = f"{model_name} ({pth_path.name}) AUC={roc_auc:.3f}"
        results[label] = {"AUC": roc_auc, **metrics}
        plt.plot(fpr, tpr, lw=2, color=colors[idx % len(colors)], label=label)

    if not results:
        print("No models could be evaluated.")
        return

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (All .pth Models)")
    plt.legend(loc="lower right", fontsize=8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    out_path = "combined_roc_all_pth.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("\n===== MODEL COMPARISON =====")
    for name, m in results.items():
        print(f"\n{name}")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
