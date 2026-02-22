import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred_logits):

    y_prob = torch.sigmoid(y_pred_logits).cpu().detach().numpy()
    y_pred = (y_prob > 0.5).astype(int)

    y_true = y_true.cpu().numpy().ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }

    return metrics
