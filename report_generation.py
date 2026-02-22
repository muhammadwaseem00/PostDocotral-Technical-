"""
Task 2: Medical Report Generation using Visual Language Model
Postdoctoral Technical Challenge - Alfaisal University

Pipeline: chest X-ray image -> VLM (MedGemma) -> natural language report.
Integrates with Task 1 CNN to identify normal, pneumonia, and misclassified samples.
Colab-ready: upload to Colab, run cells or execute as script.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# --- Configuration ---
# COLAB: Upload your Task 1 trained model to /content/
#        e.g. pneumonia_net_pneumonia.pth or resnet34_pneumonia.pth
#        The script looks in /content/ first, then current directory.
# Hugging Face token (required for MedGemma): set here or set env HF_TOKEN
# Get token: https://huggingface.co/settings/tokens  |  Accept terms: https://huggingface.co/google/medgemma-4b-it
HF_TOKEN = os.environ.get("HF_TOKEN", "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")  # Or set directly: HF_TOKEN = "hf_xxxxxxxxxxxx"

MODEL_ID = "google/medgemma-4b-it"
# Colab: upload your .pth here → /content/ ; script checks /content/ first
COLAB_DIR = Path("/content")
CNN_MODEL_PATHS = [
    COLAB_DIR / "pneumonia_net_pneumonia.pth",
    COLAB_DIR / "resnet34_pneumonia.pth",
    COLAB_DIR / "resnet18_pneumonia.pth",
    COLAB_DIR / "efficientnet_b0_pneumonia.pth",
    Path("pneumonia_net_pneumonia.pth"),
    Path("resnet34_pneumonia.pth"),
    Path("resnet18_pneumonia.pth"),
    Path("efficientnet_b0_pneumonia.pth"),
    Path(__file__).resolve().parent / "pneumonia_net_pneumonia.pth",
    Path(__file__).resolve().parent / "resnet34_pneumonia.pth",
]
FUSION_PATHS = [
    (COLAB_DIR / "resnet34_pneumonia.pth", COLAB_DIR / "resnet18_pneumonia.pth"),
    (Path("resnet34_pneumonia.pth"), Path("resnet18_pneumonia.pth")),
]


def _find_uploaded_cnn_path():
    """On Colab, if no standard path exists, use any *_pneumonia.pth in /content/."""
    for p in CNN_MODEL_PATHS:
        if p.exists():
            return None
    if COLAB_DIR.exists():
        for f in COLAB_DIR.glob("*_pneumonia.pth"):
            return f
    return None
# Output: task2_outputs/ in cwd, and on Colab also /content/ for easy finding
OUTPUT_DIR = Path.cwd() / "task2_outputs"
SAMPLE_IMAGES_DIR = OUTPUT_DIR / "sample_images"
REPORTS_JSON = OUTPUT_DIR / "generated_reports.json"
# Colab: copy also here so you can find it in Files sidebar under /content/
REPORTS_JSON_COLAB = Path("/content/generated_reports.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224  # MedGemma preprocessing

# --- Prompting strategies to test ---
PROMPTING_STRATEGIES = {
    "minimal": "Describe this chest X-ray.",
    "detailed": "Provide a detailed radiological report of this chest X-ray. Describe lung fields, heart size, mediastinum, and any abnormalities.",
    "structured": "As an expert radiologist, generate a structured report: (1) Technique/Finding, (2) Comparison if applicable, (3) Impression. Describe this chest X-ray.",
    "clinical": "You are an expert radiologist. Analyze this chest X-ray and report: lung fields (consolidation, opacity, clarity), cardiac silhouette, bony structures. Note any findings suggesting pneumonia or normal appearance.",
    "open_ended": "What do you observe in this chest X-ray? List all relevant clinical findings.",
}


@dataclass
class SampleInfo:
    idx: int
    label: int  # 0=normal, 1=pneumonia
    cnn_pred: int
    cnn_prob: float
    cnn_correct: bool
    image_path: str


def get_pneumoniamnist_data():
    """Load PneumoniaMNIST test set (no heavy transform for raw images)."""
    import medmnist
    from medmnist import INFO
    from torch.utils.data import DataLoader

    info = INFO["pneumoniamnist"]
    DataClass = getattr(medmnist, info["python_class"])
    # Minimal transform - we need raw pixel values for PIL
    from torchvision import transforms
    t = transforms.Compose([transforms.ToTensor()])
    test_ds = DataClass(split="test", transform=t, download=True)
    return test_ds


def tensor_to_pil_for_vlm(tensor: torch.Tensor, size: int = IMAGE_SIZE) -> Image.Image:
    """Convert PneumoniaMNIST tensor (1,28,28) to RGB PIL for MedGemma."""
    arr = tensor.squeeze().numpy()
    # Denormalize if needed (ToTensor gives [0,1])
    arr = (arr * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
    pil = Image.fromarray(arr, mode="L")
    pil = pil.resize((size, size), Image.BILINEAR)
    pil_rgb = pil.convert("RGB")
    return pil_rgb


def _get_pneumonia_net():
    """PneumoniaNet: same architecture as Task 1 (custom residual for 28x28)."""
    import torch.nn as nn
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


def _get_cnn_model(arch: str):
    """Build ResNet18, ResNet34, EfficientNet-B0, or PneumoniaNet for 1-channel input, binary output."""
    from torchvision import models
    import torch.nn as nn
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


def _arch_from_name(name: str) -> str:
    name = name.lower()
    if "resnet34" in name:
        return "resnet34"
    if "efficientnet_b0" in name or "efficientnet" in name:
        return "efficientnet_b0"
    if "pneumonia_net" in name:
        return "pneumonia_net"
    return "resnet18"


def load_cnn_and_predict(test_ds):
    """Load Task 1 CNN and get predictions on test set. Set FUSION=1 to use two-model ensemble."""
    from torch.utils.data import DataLoader

    use_fusion = os.environ.get("FUSION", "").strip() == "1"
    fusion_pair = next((p for p in FUSION_PATHS if p[0].exists() and p[1].exists()), None) if use_fusion else None

    if use_fusion and fusion_pair:
        path1, path2 = fusion_pair
        m1 = _get_cnn_model(_arch_from_name(path1.name))
        m2 = _get_cnn_model(_arch_from_name(path2.name))
        for path, model in [(path1, m1), (path2, m2)]:
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        m1.eval()
        m2.eval()
        print(f"Loaded fusion ensemble: {path1.name} + {path2.name}")
        loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        preds, probs, labels = [], [], []
        with torch.no_grad():
            for x, y in loader:
                p1 = torch.sigmoid(m1(x).squeeze()).numpy()
                p2 = torch.sigmoid(m2(x).squeeze()).numpy()
                p = (p1 + p2) / 2
                preds.extend((p >= 0.5).astype(int))
                probs.extend(p)
                labels.extend(y.numpy().ravel().astype(int))
        return np.array(preds), np.array(probs), np.array(labels)

    paths_to_try = list(CNN_MODEL_PATHS)
    uploaded = _find_uploaded_cnn_path()
    if uploaded is not None:
        paths_to_try.insert(0, uploaded)
    cnn_path = next((p for p in paths_to_try if p.exists()), None)
    if not cnn_path:
        print("Warning: CNN model not found. Upload your trained model (e.g. pneumonia_net_pneumonia.pth) to /content/ in Colab.")
        model = _get_cnn_model("resnet18")
    else:
        arch = _arch_from_name(cnn_path.name)
        model = _get_cnn_model(arch)
        try:
            state = torch.load(cnn_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            print(f"Loaded CNN from {cnn_path} (architecture: {arch})")
        except (RuntimeError, OSError) as e:
            size_mb = cnn_path.stat().st_size / (1024 * 1024) if cnn_path.exists() else 0
            print(
                f"Warning: CNN checkpoint failed to load ({cnn_path}, {size_mb:.1f} MB). "
                "Using random-initialized model so VLM can still run."
            )
            print(f"  Error: {e}")

    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x).squeeze()
            p = torch.sigmoid(logits).numpy()
            preds.extend((p >= 0.5).astype(int))
            probs.extend(p)
            labels.extend(y.numpy().ravel().astype(int))
    return np.array(preds), np.array(probs), np.array(labels)


def select_samples(
    test_ds, preds, probs, labels, n_normal=4, n_pneumonia=4, n_misclassified=4
) -> List[SampleInfo]:
    """Select representative samples: normal, pneumonia, and CNN misclassified."""
    correct = (preds == labels)
    normal_idx = np.where(labels == 0)[0]
    pneumonia_idx = np.where(labels == 1)[0]
    wrong_idx = np.where(~correct)[0]

    samples = []
    for idx in np.random.choice(normal_idx, min(n_normal, len(normal_idx)), replace=False):
        samples.append(
            SampleInfo(
                idx=int(idx),
                label=int(labels[idx]),
                cnn_pred=int(preds[idx]),
                cnn_prob=float(probs[idx]),
                cnn_correct=bool(correct[idx]),
                image_path="",
            )
        )
    for idx in np.random.choice(pneumonia_idx, min(n_pneumonia, len(pneumonia_idx)), replace=False):
        samples.append(
            SampleInfo(
                idx=int(idx),
                label=int(labels[idx]),
                cnn_pred=int(preds[idx]),
                cnn_prob=float(probs[idx]),
                cnn_correct=bool(correct[idx]),
                image_path="",
            )
        )
    for idx in np.random.choice(wrong_idx, min(n_misclassified, len(wrong_idx)), replace=False):
        samples.append(
            SampleInfo(
                idx=int(idx),
                label=int(labels[idx]),
                cnn_pred=int(preds[idx]),
                cnn_prob=float(probs[idx]),
                cnn_correct=False,
                image_path="",
            )
        )
    np.random.shuffle(samples)
    return samples[: max(n_normal + n_pneumonia + n_misclassified, 10)]


def load_vlm_and_generate(
    model_id: str = MODEL_ID,
    max_new_tokens: int = 256,
    use_pipeline: bool = True,
):
    """Load MedGemma and return a generate(img, prompt) -> str function."""
    try:
        from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        raise ImportError("pip install transformers>=4.50.0 accelerate")

    if use_pipeline:
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device=0 if DEVICE == "cuda" else -1,
        )

        def generate(img: Image.Image, prompt: str) -> str:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img}]},
            ]
            out = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
            return out[0]["generated_text"][-1]["content"].strip()

    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)

        def generate(img: Image.Image, prompt: str) -> str:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img}]},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.decode(gen[0][input_len:], skip_special_tokens=True)
            return decoded.strip()

    return generate


def main(
    n_samples: int = 12,
    strategies_to_test: Optional[List[str]] = None,
    skip_vlm: bool = False,
):
    """
    Main pipeline: load data -> CNN predictions -> select samples -> VLM reports.
    VLM is required for Task 2 deliverables. Use skip_vlm only for debugging.
    """
    import traceback
    results = []
    strategies = strategies_to_test or list(PROMPTING_STRATEGIES.keys())

    def write_json(data=None):
        data = data if data is not None else results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        p = REPORTS_JSON.resolve()
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved reports to: {p}")
        # Also save in /content/ on Colab so you see it in the file browser
        try:
            with open(REPORTS_JSON_COLAB, "w") as f:
                json.dump(data, f, indent=2)
            print(f"COPY IN /content/:  {REPORTS_JSON_COLAB}")
        except OSError:
            pass

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {OUTPUT_DIR.resolve()}")
        print(f"Reports will be saved to: {REPORTS_JSON.resolve()}")

        np.random.seed(42)
        torch.manual_seed(42)

        print("Loading PneumoniaMNIST test set...")
        test_ds = get_pneumoniamnist_data()
        print("Running Task 1 CNN for predictions...")
        preds, probs, labels = load_cnn_and_predict(test_ds)
        print(f"CNN accuracy: {(preds == labels).mean():.4f}")

        print("Selecting representative samples...")
        samples = select_samples(
            test_ds, preds, probs, labels,
            n_normal=4, n_pneumonia=4, n_misclassified=4,
        )
        samples = samples[:n_samples]

        for s in samples:
            img, _ = test_ds[s.idx]
            pil = tensor_to_pil_for_vlm(img)
            path = SAMPLE_IMAGES_DIR / f"sample_{s.idx}_gt{s.label}_cnn{s.cnn_pred}.png"
            pil.save(path)
            s.image_path = str(path)

        for s in samples:
            results.append({
                "idx": s.idx,
                "ground_truth": "Pneumonia" if s.label == 1 else "Normal",
                "cnn_pred": "Pneumonia" if s.cnn_pred == 1 else "Normal",
                "cnn_prob": s.cnn_prob,
                "cnn_correct": s.cnn_correct,
                "image_path": s.image_path,
                "reports": {},
            })

        if skip_vlm:
            for r in results:
                r["reports"] = {k: "[VLM skipped]" for k in strategies}
            write_json()
            raise SystemExit("Task 2 requires VLM. Run without --skip-vlm.")

        print("Loading MedGemma (requires transformers>=4.50, GPU, HF login)...")
        hf_token = os.environ.get("HF_TOKEN", "") or HF_TOKEN
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            print("Logged in to Hugging Face.")
        try:
            generate_fn = load_vlm_and_generate(use_pipeline=True)
        except Exception as e:
            for r in results:
                r["reports"] = {k: f"[VLM failed to load: {e}]" for k in strategies}
            write_json()
            raise SystemExit(
                f"VLM failed: {e}\n"
                "Fix: (1) Enable GPU  (2) pip install -U transformers>=4.50  (3) huggingface_hub.login()"
            ) from e

        for i, s in enumerate(samples):
            img_tensor, _ = test_ds[s.idx]
            pil_img = tensor_to_pil_for_vlm(img_tensor)
            for strat_name in strategies:
                prompt = PROMPTING_STRATEGIES.get(strat_name, strat_name)
                try:
                    report = generate_fn(pil_img, prompt)
                    results[i]["reports"][strat_name] = report
                except Exception as e:
                    results[i]["reports"][strat_name] = f"[VLM Error: {e}]"
            print(f"  Generated reports for sample {i+1}/{len(samples)}")

        write_json()
        return results

    except SystemExit:
        raise
    except Exception as e:
        err_data = [{"error": str(e), "traceback": traceback.format_exc()}]
        write_json(err_data)
        raise


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--skip-vlm", action="store_true", help="Not supported: Task 2 requires VLM; will exit")
    ap.add_argument("--token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    args = ap.parse_args()
    if args.token:
        os.environ["HF_TOKEN"] = args.token
    main(n_samples=args.n_samples, strategies_to_test=args.strategies, skip_vlm=args.skip_vlm)
    print("---")
    print("JSON file is here: /content/generated_reports.json  (Colab: open Files panel, click 'content', look for generated_reports.json)")
    print("Or here:", REPORTS_JSON.resolve())
