"""
Task 3: Integrated inference pipeline — CNN + optional VLM.
Postdoctoral Technical Challenge - Alfaisal University.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional

from config import CNN_MODEL_PATHS, VLM_MODEL_ID, IMAGE_SIZE_CNN, IMAGE_SIZE_VLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional fusion paths: (path1, path2) for two-model ensemble
try:
    from config import FUSION_PATHS
except ImportError:
    FUSION_PATHS = []


def _get_pneumonia_net():
    """PneumoniaNet: same architecture as Task 1 (custom residual for 28x28)."""
    import torch.nn.functional as F
    class ResidualBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride),
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.downsample = nn.Identity()

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            shortcut = self.downsample(x)
            if shortcut.shape[2:] != out.shape[2:]:
                shortcut = F.adaptive_avg_pool2d(shortcut, out.shape[2:])
            return torch.relu(out + shortcut)

    class PneumoniaNet(nn.Module):
        def __init__(self, in_ch: int = 1, base_ch: int = 32, dropout: float = 0.35):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, base_ch, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            self.layer1 = self._make_layer(base_ch, base_ch, 2, stride=1)
            self.layer2 = self._make_layer(base_ch, base_ch * 2, 2, stride=2)
            self.layer3 = self._make_layer(base_ch * 2, base_ch * 4, 2, stride=2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(base_ch * 4, 1)

        def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int):
            layers = [ResidualBlock(in_ch, out_ch, stride)]
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_ch, out_ch, 1))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.drop(x)
            return self.fc(x)
    return PneumoniaNet()


class FusionWrapper(nn.Module):
    """Wraps two models and returns logit from average of their probabilities."""
    def __init__(self, m1: nn.Module, m2: nn.Module):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        logit1 = self.m1(x).squeeze()
        logit2 = self.m2(x).squeeze()
        p1 = torch.sigmoid(logit1)
        p2 = torch.sigmoid(logit2)
        p = (p1 + p2).clamp(1e-6, 1 - 1e-6) / 2
        return (p / (1 - p)).log()


def _arch_from_name(name: str) -> str:
    name = name.lower()
    if "resnet34" in name:
        return "resnet34"
    if "efficientnet_b0" in name or "efficientnet" in name:
        return "efficientnet_b0"
    if "pneumonia_net" in name:
        return "pneumonia_net"
    return "resnet18"


def _get_cnn_model(arch: str):
    """ResNet18/34, EfficientNet-B0, or PneumoniaNet for 1-channel 28x28, binary output."""
    from torchvision import models
    if arch == "resnet34":
        model = models.resnet34(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride, padding=old_conv.padding, bias=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
    elif arch == "pneumonia_net":
        model = _get_pneumonia_net()
    else:
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_cnn():
    """Load Task 1 CNN from first available path. Set FUSION=1 to use two-model ensemble."""
    use_fusion = os.environ.get("FUSION", "").strip() == "1"
    fusion_pair = next((p for p in FUSION_PATHS if p[0].exists() and p[1].exists()), None) if use_fusion and FUSION_PATHS else None

    if use_fusion and fusion_pair:
        path1, path2 = fusion_pair
        m1 = _get_cnn_model(_arch_from_name(path1.name))
        m2 = _get_cnn_model(_arch_from_name(path2.name))
        for path, model in [(path1, m1), (path2, m2)]:
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        m1.to(DEVICE)
        m2.to(DEVICE)
        m1.eval()
        m2.eval()
        model = FusionWrapper(m1, m2)
        model.to(DEVICE)
        model.eval()
        print(f"Loaded fusion ensemble: {path1.name} + {path2.name}")
        return model

    cnn_path = next((p for p in CNN_MODEL_PATHS if p.exists()), None)
    if not cnn_path:
        raise FileNotFoundError(
            f"No CNN checkpoint found. Tried: {[str(p) for p in CNN_MODEL_PATHS]}. "
            "Place resnet34_pneumonia.pth, pneumonia_net_pneumonia.pth, resnet18_pneumonia.pth, or efficientnet_b0_pneumonia.pth in /content/ or current dir."
        )
    arch = _arch_from_name(cnn_path.name)
    model = _get_cnn_model(arch)
    state = torch.load(cnn_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def load_vlm():
    """Load MedGemma pipeline; returns a function (pil_image, prompt) -> report text."""
    import os
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from config import HF_TOKEN
            token = HF_TOKEN or ""
        except Exception:
            pass
    if token:
        from huggingface_hub import login
        login(token=token)
        print("Logged in to Hugging Face (MedGemma).")
    from transformers import pipeline
    print("Loading MedGemma (first time may take 2–5 min, please wait)...")
    dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
    try:
        pipe = pipeline(
            "image-text-to-text",
            model=VLM_MODEL_ID,
            dtype=dtype,
            device=0 if DEVICE.type == "cuda" else -1,
        )
    except TypeError:
        pipe = pipeline(
            "image-text-to-text",
            model=VLM_MODEL_ID,
            torch_dtype=dtype,
            device=0 if DEVICE.type == "cuda" else -1,
        )
    default_prompt = (
        "You are an expert radiologist. Analyze this chest X-ray and report: "
        "lung fields, cardiac silhouette. Note any findings suggesting pneumonia or normal appearance."
    )

    def generate(img: Image.Image, prompt: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt or default_prompt}, {"type": "image", "image": img}]},
        ]
        out = pipe(messages, max_new_tokens=256, do_sample=False)
        # Handle different pipeline output formats
        gen = out[0].get("generated_text", out[0]) if out else ""
        if isinstance(gen, str):
            return gen.strip()
        if isinstance(gen, list) and len(gen) > 0:
            last = gen[-1]
            if isinstance(last, dict) and "content" in last:
                return str(last["content"]).strip()
            return str(last).strip()
        return str(gen).strip()

    return generate


def preprocess_for_cnn(image: Image.Image) -> torch.Tensor:
    """Resize to 28x28 grayscale, normalize; tensor (1,1,28,28)."""
    img = image.convert("L").resize((IMAGE_SIZE_CNN, IMAGE_SIZE_CNN), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    return t.to(DEVICE)


def preprocess_for_vlm(image: Image.Image) -> Image.Image:
    """Resize to 224x224 RGB for MedGemma."""
    img = image.convert("L").resize((IMAGE_SIZE_VLM, IMAGE_SIZE_VLM), Image.BILINEAR)
    return img.convert("RGB")


def load_pipeline(use_vlm: bool = False) -> Dict[str, Any]:
    """Load CNN and optionally VLM. Returns dict with 'cnn_model' and optionally 'vlm_generate'."""
    pipeline_dict = {"cnn_model": load_cnn()}
    if use_vlm:
        try:
            pipeline_dict["vlm_generate"] = load_vlm()
            print("VLM (MedGemma) loaded successfully.")
        except Exception as e:
            print(f"Warning: VLM could not be loaded ({e}). Run without --vlm or check HF_TOKEN and MedGemma access.")
    return pipeline_dict


def run_inference(
    pipeline: Dict[str, Any],
    image_input,  # path string or PIL Image
    run_vlm: bool = True,
    vlm_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run CNN + optional VLM on one image.
    image_input: path to image file or PIL.Image.
    Returns dict with cnn_pred, cnn_prob, and optionally vlm_report.
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    else:
        image = image_input

    cnn_model = pipeline["cnn_model"]
    x = preprocess_for_cnn(image)
    with torch.no_grad():
        logit = cnn_model(x).squeeze().item()
    prob = 1.0 / (1.0 + np.exp(-logit))
    pred = "Pneumonia" if prob >= 0.5 else "Normal"

    out = {"cnn_pred": pred, "cnn_prob": round(float(prob), 4)}

    if run_vlm:
        if "vlm_generate" in pipeline:
            try:
                pil_vlm = preprocess_for_vlm(image)
                out["vlm_report"] = pipeline["vlm_generate"](pil_vlm, vlm_prompt)
            except Exception as e:
                out["vlm_report"] = f"[VLM error] {type(e).__name__}: {str(e)}"
        else:
            out["vlm_report"] = "[VLM not available] Load failed (check HF token and MedGemma access)."

    return out
