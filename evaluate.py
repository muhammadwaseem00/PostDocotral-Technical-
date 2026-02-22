import torch
from config import Config
from dataset import get_dataloaders
from model import get_model
from utils import calculate_metrics

def evaluate():

    _, _, test_loader = get_dataloaders(
        Config.batch_size,
        Config.num_workers
    )

    model = get_model().to(Config.device)
    model.load_state_dict(torch.load(Config.model_save_path, map_location=Config.device))
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

    metrics = calculate_metrics(all_labels, all_logits)

    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    evaluate()
