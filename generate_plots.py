"""
Generate visualizations: ROC curve, confusion matrix, sample predictions.
Run after train.py and evaluate.py. Saves plots to current directory.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from config import Config
from dataset import get_dataloaders
from model import get_model
from utils import calculate_metrics


def plot_roc_curve(y_true, y_prob, save_path="roc_curve.png"):
    """Plot ROC curve and save to file."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot confusion matrix and save to file."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_predictions(y_true, y_prob, y_pred, images, n_samples=10, save_path="sample_predictions.png"):
    """Plot sample correct and incorrect predictions."""
    # Denormalize images: was normalized with mean=0.5, std=0.5 -> x = x_norm * 0.5 + 0.5
    images_disp = np.clip(images * 0.5 + 0.5, 0, 1)

    # Find correct and incorrect indices
    correct = np.where(y_pred == y_true)[0]
    incorrect = np.where(y_pred != y_true)[0]

    n_each = n_samples // 2
    indices = []
    if len(correct) >= n_each:
        indices.extend(np.random.choice(correct, n_each, replace=False))
    else:
        indices.extend(correct)
    if len(incorrect) >= n_each:
        indices.extend(np.random.choice(incorrect, n_each, replace=False))
    else:
        indices.extend(incorrect)

    if len(indices) == 0:
        indices = np.arange(min(n_samples, len(y_true)))

    n_show = min(len(indices), n_samples)
    cols = 5
    rows = max(1, (n_show + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, idx in enumerate(indices[:n_show]):
        ax = axes[i]
        img = images_disp[idx].squeeze()
        ax.imshow(img, cmap='gray')
        label_str = 'Pneumonia' if y_true[idx] == 1 else 'Normal'
        pred_str = 'Pneumonia' if y_pred[idx] == 1 else 'Normal'
        prob = y_prob[idx]
        color = 'green' if y_pred[idx] == y_true[idx] else 'red'
        ax.set_title(f'True: {label_str}\nPred: {pred_str} ({prob:.2f})', fontsize=8, color=color)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Sample Predictions (green=correct, red=incorrect)', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Loading model and running inference...")
    _, _, test_loader = get_dataloaders(Config.batch_size, Config.num_workers)

    model = get_model().to(Config.device)
    model.load_state_dict(torch.load(Config.model_save_path, map_location=Config.device))
    model.eval()

    all_logits = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(Config.device)
            labels_sq = labels.float().squeeze(-1).to(Config.device)
            outputs = model(images_gpu).squeeze()
            all_logits.append(outputs)
            all_labels.append(labels_sq)
            all_images.append(images.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_images = torch.cat(all_images)

    y_prob = torch.sigmoid(all_logits).cpu().numpy()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = all_labels.cpu().numpy().ravel()
    images_np = all_images.numpy()

    metrics = calculate_metrics(all_labels, all_logits)
    print("Metrics:", metrics)

    # Set seed for reproducible sample selection
    np.random.seed(42)

    plot_roc_curve(y_true, y_prob)
    plot_confusion_matrix(y_true, y_pred)
    plot_sample_predictions(y_true, y_prob, y_pred, images_np)

    print("Done. Plots saved: roc_curve.png, confusion_matrix.png, sample_predictions.png")


if __name__ == "__main__":
    main()
