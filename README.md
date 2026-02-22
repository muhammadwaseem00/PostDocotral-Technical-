# Task 3: Integrated Inference Pipeline

**Postdoctoral Technical Challenge - Alfaisal University**

Single entry point that combines **Task 1 (CNN)** and **Task 2 (VLM)** for chest X-ray analysis: given an image, returns CNN prediction (Normal/Pneumonia + probability) and optional VLM text report.

---

## Overview

| Component | Description |
|-----------|-------------|
| **inference_pipeline.py** | Load CNN + optional VLM; run on one image; return combined result |
| **run_single_image.py** | CLI: process one image and print or save JSON |
| **app.py** | Optional FastAPI server: `POST /predict` with image file |

---

## Requirements

- Trained CNN: `resnet34_pneumonia.pth` or `resnet18_pneumonia.pth` (from Task 1)
- For VLM: GPU, `transformers>=4.50`, Hugging Face login (same as Task 2)

---

## Setup

```bash
cd Pneumonia_Task3
pip install -r requirements.txt
```

Put your trained model in one of the paths listed in `config.py` (e.g. `/content/resnet34_pneumonia.pth` on Colab).

---

## Usage

### 1. Process one image (CLI)

**Default (Colab/GPU):** Runs CNN + VLM and writes full JSON (`cnn_pred`, `cnn_prob`, `vlm_report`).  
**Hugging Face:** For VLM (MedGemma) you must pass your token (same as Task 2). Accept terms: https://huggingface.co/google/medgemma-4b-it

```bash
# Full output with VLM — pass your HF token (Colab)
python run_single_image.py --image path/to/chest_xray.png --token "hf_YourTokenHere" -o result.json

# Or set token in config.py (HF_TOKEN = "hf_...") or env: export HF_TOKEN=hf_...
python run_single_image.py --image path/to/chest_xray.png -o result.json

# CNN only (no VLM, no token needed)
python run_single_image.py --image path/to/chest_xray.png --no-vlm -o result.json
```

### 2. Use pipeline in code

```python
from inference_pipeline import load_pipeline, run_inference

pipe = load_pipeline(use_vlm=False)  # or use_vlm=True
out = run_inference(pipe, "path/to/image.png")
print(out["cnn_pred"], out["cnn_prob"])
# With VLM: out["vlm_report"]
```

### 3. Run API server (optional)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# POST /predict with form-data: file=image.png
```

---

## Output

JSON written to `--output` (or printed) contains:

- **cnn_pred**: "Normal" or "Pneumonia"
- **cnn_prob**: float in [0, 1]
- **vlm_report**: (when VLM runs, i.e. default on GPU) text report from MedGemma

Example full output: `{"cnn_pred": "Normal", "cnn_prob": 0.0487, "vlm_report": "Okay, I've reviewed the chest X-ray. ..."}`
