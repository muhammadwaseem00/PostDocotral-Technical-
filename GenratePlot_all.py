def main():
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(Config.batch_size, Config.num_workers)

    model_names = ["resnet18", "resnet34", "resnet50", "efficientnet_b0"]

    results = {}

    plt.figure(figsize=(7, 6))

    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")

        Config.model_name = model_name
        Config.model_save_path = f"{model_name}_pneumonia.pth"

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

        y_prob = torch.sigmoid(all_logits).cpu().numpy()
        y_true = all_labels.cpu().numpy().ravel()
        y_pred = (y_prob > 0.5).astype(int)

        # Metrics
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        metrics = calculate_metrics(all_labels, all_logits)

        results[model_name] = {
            "AUC": roc_auc,
            **metrics
        }

        # Plot ROC
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={roc_auc:.3f})")

    # Diagonal line
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_roc_curve.png", dpi=150)
    plt.close()

    print("\n===== MODEL COMPARISON =====")
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    print("\nSaved: combined_roc_curve.png")
