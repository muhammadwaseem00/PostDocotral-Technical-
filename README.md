# Task 2: Medical Report Generation using Visual Language Model

**Postdoctoral Technical Challenge - Alfaisal University**

**Environment:** This task was implemented and run on **Google Colab** (GPU runtime, Hugging Face login for MedGemma).

---

## Objective

Use an open-source visual language model (VLM) to generate medical reports from chest X-ray images. The pipeline integrates with Task 1's CNN to select representative samples (normal, pneumonia, and misclassified) and produces natural language radiological descriptions.

---

## Components

| Component | Description |
|-----------|-------------|
| **VLM** | MedGemma 4B (Google) — medical VLM for chest X-ray analysis |
| **Report generation** | Image → preprocess → VLM → natural language report |
| **CNN integration** | Task 1 model used to identify misclassified samples for qualitative analysis |
| **Prompting strategies** | Minimal, detailed, structured, clinical, open-ended |

---

## Setup

```bash
pip install -r requirements.txt
```

**Hugging Face:**  
1. Accept MedGemma terms: https://huggingface.co/google/medgemma-4b-it (click "Access repository" when logged in).  
2. Create a token: https://huggingface.co/settings/tokens → Create new token (Read).  
3. In Colab: paste the token in the notebook login cell (`HF_TOKEN = "hf_..."`) or run:  
   `!python report_generation.py --n-samples 12 --token "hf_YourToken"`

---

## Usage (Google Colab)

1. Upload to `/content/`:
   - **report_generation.py** (this task's script)
   - **Your Task 1 trained model**, e.g. `pneumonia_net_pneumonia.pth` or `resnet34_pneumonia.pth` or `resnet18_pneumonia.pth`
2. Enable **GPU**: Runtime → Change runtime type → T4 GPU.
3. Install and add your token:
   ```python
   !pip install transformers>=4.50 accelerate torch torchvision medmnist Pillow huggingface_hub
   ```
   In the next cell, paste your Hugging Face token (from https://huggingface.co/settings/tokens):
   ```python
   HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"  # paste your token
   from huggingface_hub import login
   login(token=HF_TOKEN)
   ```
4. Run the pipeline:
   ```python
   !python report_generation.py --n-samples 12
   ```
   Or pass the token on the command line: `!python report_generation.py --n-samples 12 --token "hf_xxx"`
5. Outputs: `/content/generated_reports.json` and `/content/task2_outputs/sample_images/`.

---

## Deliverables

| File | Description |
|------|-------------|
| `report_generation.py` | Main pipeline script |
| `task2 report generation.md` | Full report: model justification, prompting strategies, qualitative analysis |
| `task2_outputs/generated_reports.json` | Sample reports (after running pipeline) |
| `task2_outputs/sample_images/` | Saved sample chest X-ray images |
