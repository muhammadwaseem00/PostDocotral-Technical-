import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from dataset import get_dataloaders
from model import get_model
from utils import calculate_metrics

def train():

    torch.manual_seed(Config.seed)

    train_loader, val_loader, _ = get_dataloaders(
        Config.batch_size,
        Config.num_workers
    )

    model = get_model().to(Config.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=Config.lr,
                            weight_decay=Config.weight_decay)

    best_val_auc = 0
    epochs_without_improvement = 0
    patience = getattr(Config, 'early_stopping_patience', 5)

    for epoch in range(Config.num_epochs):

        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(Config.device)
            labels = labels.float().squeeze(-1).to(Config.device)

            outputs = model(images).squeeze()

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {running_loss/len(train_loader)}")

        # Validation
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.device)
                labels = labels.float().squeeze(-1).to(Config.device)

                outputs = model(images).squeeze()

                all_logits.append(outputs)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        metrics = calculate_metrics(all_labels, all_logits)

        print(f"Validation AUC: {metrics['auc']}")

        if metrics['auc'] > best_val_auc:
            best_val_auc = metrics['auc']
            epochs_without_improvement = 0
            save_path = Config.model_save_path
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to: {save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

    print("Training complete.")

if __name__ == "__main__":
    train()
