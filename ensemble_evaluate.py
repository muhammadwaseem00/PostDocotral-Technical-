"""
Ensemble (fusion) evaluation: combine two trained models by averaging their
predicted probabilities. Use this to get a single stronger predictor without
training a new model.
Example: python ensemble_evaluate.py resnet34 resnet18
"""
import sys
import torch
from dataset import get_dataloaders
from models import get_model, MODEL_REGISTRY
from utils import calculate_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model_name: str, path: str = None):
    """Load model by name and optional path. Path defaults to {model_name}_pneumonia.pth."""
    path = path or f"{model_name}_pneumonia.pth"
    model = get_model(model_name).to(DEVICE)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    if len(sys.argv) < 3:
        print("Usage: python ensemble_evaluate.py <model1> <model2> [path1] [path2]")
        print("Example: python ensemble_evaluate.py resnet34 resnet18")
        print("Available:", list(MODEL_REGISTRY.keys()))
        sys.exit(1)
    name1, name2 = sys.argv[1].lower(), sys.argv[2].lower()
    path1 = sys.argv[3] if len(sys.argv) > 3 else None
    path2 = sys.argv[4] if len(sys.argv) > 4 else None

    if name1 not in MODEL_REGISTRY or name2 not in MODEL_REGISTRY:
        print("Unknown model. Available:", list(MODEL_REGISTRY.keys()))
        sys.exit(1)

    _, _, test_loader = get_dataloaders(64, 2)
    m1 = load_checkpoint(name1, path1)
    m2 = load_checkpoint(name2, path2)

    all_logits1 = []
    all_logits2 = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.float().squeeze(-1)
            all_logits1.append(m1(images).squeeze().cpu())
            all_logits2.append(m2(images).squeeze().cpu())
            all_labels.append(labels)

    logits1 = torch.cat(all_logits1)
    logits2 = torch.cat(all_logits2)
    labels = torch.cat(all_labels).to(DEVICE)
    # Average probabilities (fusion)
    prob1 = torch.sigmoid(logits1).to(DEVICE)
    prob2 = torch.sigmoid(logits2).to(DEVICE)
    avg_prob = (prob1 + prob2) / 2
    p = avg_prob.clamp(1e-6, 1 - 1e-6)
    ensemble_logits = (p / (1 - p)).log().to(DEVICE)

    metrics = calculate_metrics(labels, ensemble_logits)
    print(f"Ensemble ({name1} + {name2}):")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
