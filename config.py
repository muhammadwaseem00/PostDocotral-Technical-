"""Task 3: paths and settings for integrated inference pipeline."""
import os
from pathlib import Path

# Hugging Face token for MedGemma (set here, or env HF_TOKEN, or --token when running)
# Get token: https://huggingface.co/settings/tokens  |  Accept terms: https://huggingface.co/google/medgemma-4b-it
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# CNN checkpoint: first path that exists is used. Set FUSION=1 to use two-model ensemble.
CNN_MODEL_PATHS = [
    Path("/content/resnet34_pneumonia.pth"),
    Path("/content/pneumonia_net_pneumonia.pth"),
    Path("/content/resnet18_pneumonia.pth"),
    Path("/content/efficientnet_b0_pneumonia.pth"),
    Path("resnet34_pneumonia.pth"),
    Path("pneumonia_net_pneumonia.pth"),
    Path("resnet18_pneumonia.pth"),
    Path("efficientnet_b0_pneumonia.pth"),
    Path(__file__).resolve().parent.parent / "Pneumonia_Task1" / "venv" / "Scripts" / "chleng" / "resnet34_pneumonia.pth",
    Path(__file__).resolve().parent.parent / "Pneumonia_Task1" / "venv" / "Scripts" / "chleng" / "pneumonia_net_pneumonia.pth",
    Path(__file__).resolve().parent.parent / "Pneumonia_Task1" / "venv" / "Scripts" / "chleng" / "resnet18_pneumonia.pth",
]
FUSION_PATHS = [
    (Path("/content/resnet34_pneumonia.pth"), Path("/content/resnet18_pneumonia.pth")),
    (Path("resnet34_pneumonia.pth"), Path("resnet18_pneumonia.pth")),
]

VLM_MODEL_ID = "google/medgemma-4b-it"
IMAGE_SIZE_CNN = 28   # PneumoniaMNIST size
IMAGE_SIZE_VLM = 224  # MedGemma input
