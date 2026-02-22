import torch

class Config:
    seed = 42
    batch_size = 64
    num_epochs = 25
    lr = 1e-3
    weight_decay = 5e-4  # stronger regularization to reduce overfitting
    num_workers = 2
    early_stopping_patience = 5  # stop if val AUC does not improve for this many epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection: resnet18, resnet34, resnet50, efficientnet_b0, vgg16, custom_cnn, pneumonia_net
    # pneumonia_net = custom residual model (no pretrained). Use ensemble_evaluate.py for fusion of two models.
    model_name = "pneumonia_net"
    model_save_path = f"{model_name}_pneumonia.pth"
