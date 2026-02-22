# Task 2: Medical Report Generation using Visual Language Model

**Postdoctoral Technical Challenge - Alfaisal University**

**Implementation environment:** All experiments and report generation were carried out on **Google Colab** (GPU runtime, T4). The CNN model (`resnet34_pneumonia.pth`) and `report_generation.py` were uploaded to `/content/`; MedGemma was loaded via Hugging Face after `huggingface_hub.login()`. Outputs: `generated_reports.json` and `task2_outputs/sample_images/` on Colab.

---

## 1. Model Selection Justification

### MedGemma (Selected)

We use **MedGemma 4B (instruction-tuned)**, Google's open-source medical visual language model available on Hugging Face (`google/medgemma-4b-it`).

**Rationale:**
- **Medical-specific pre-training**: MedGemma's image encoder (SigLIP) and LLM are pre-trained on de-identified medical data, including chest X-rays, dermatology, ophthalmology, and histopathology (Sellergren et al., arXiv:2507.05201).
- **Chest X-ray expertise**: The model is explicitly designed for radiology tasks and medical image interpretation.
- **Open-source and reproducible**: Available via Hugging Face with clear licensing (Health AI Developer Foundation terms).
- **Colab-compatible**: The 4B variant fits on consumer GPUs (~8GB VRAM), enabling Colab execution.

### Alternatives Considered

| Model | Pros | Cons |
|-------|------|------|
| **LLaVA-Med** | General medical VLM | Less focused on radiology; older architecture |
| **BLIP-2** | Lightweight, general VLM | Not medically fine-tuned |
| **MedGemma 27B** | Higher capacity | Requires significant GPU memory; overkill for this task |

---

## 2. Prompting Strategies Tested

We evaluated five prompting strategies to guide report generation:

| Strategy | Prompt | Effect on Output |
|----------|--------|------------------|
| **minimal** | "Describe this chest X-ray." | Short, generic descriptions; sometimes vague |
| **detailed** | "Provide a detailed radiological report... Describe lung fields, heart size, mediastinum, and any abnormalities." | More structured; encourages anatomical coverage |
| **structured** | "As an expert radiologist, generate a structured report: (1) Technique/Finding, (2) Comparison, (3) Impression." | Best format for clinical-style reports; clearest structure |
| **clinical** | "You are an expert radiologist. Analyze this chest X-ray and report: lung fields, cardiac silhouette, bony structures. Note any findings suggesting pneumonia or normal appearance." | Most aligned with pneumonia task; emphasizes pathology |
| **open_ended** | "What do you observe in this chest X-ray? List all relevant clinical findings." | Flexible; output varies in structure |

**Recommendation:** The **clinical** and **structured** prompts produced the most useful reports for pneumonia detection and align best with radiological reporting conventions.

---

## 3. Sample Generated Reports with Corresponding Images

Sample outputs are saved to `task2_outputs/sample_images/` and full reports to `task2_outputs/generated_reports.json`. Below are representative examples (clinical prompt).

### Sample 41: Misclassified — GT Normal, CNN Pneumonia (0.75)
- **Image:** `task2_outputs/sample_images/sample_41_gt0_cnn1.png`
- **VLM Report (clinical):** *"Okay, I've reviewed the chest X-ray. **Lung Fields:** The lung fields appear relatively clear. There is no obvious consolidation, opacity, or infiltrates. ..."*  
- **Interpretation:** VLM describes clear lungs, aligning with **ground truth (Normal)**; CNN incorrectly predicted Pneumonia.

### Sample 470: Misclassified — GT Normal, CNN Pneumonia (0.75)
- **Image:** `task2_outputs/sample_images/sample_470_gt0_cnn1.png`
- **VLM Report (clinical):** *"The image appears to be a frontal chest X-ray. The lung fields are relatively clear, with no obvious consolidation, opacities, or infiltrates. ..."*  
- **Interpretation:** VLM agrees with GT (normal); CNN false positive.

### Sample 144: Misclassified — GT Normal, CNN Pneumonia (0.77)
- **Image:** `task2_outputs/sample_images/sample_144_gt0_cnn1.png`
- **VLM Report (clinical):** *"**Lung Fields:** The lung fields appear clear bilaterally. There is no evidence of consolidation, opacity, or infiltrates. ..."*  
- **Interpretation:** Again, VLM text supports Normal; CNN overcalled Pneumonia.

*At least 10 samples (mix of normal, pneumonia, and misclassified) are generated; full text for all strategies is in `generated_reports.json`.*

---

## 4. Qualitative Analysis: VLM vs. Ground Truth and CNN

### Comparison Framework

| Aspect | Ground Truth | Task 1 CNN | VLM (MedGemma) |
|--------|--------------|------------|----------------|
| **Output type** | Binary (Normal/Pneumonia) | Binary + probability | Free-text radiological description |
| **Interpretability** | N/A | Black-box | Natural language explanations |
| **Complementarity** | — | High accuracy, no explanation | Explanatory text; can mention findings that CNN may miss or misweight |

### Observations

1. **Alignment with ground truth**: The VLM’s descriptions often align with the true label when it mentions “no focal consolidation” (normal) or “opacity/consolidation” (pneumonia). However, PneumoniaMNIST’s low resolution (28×28) limits anatomical detail, which can lead to generic or uncertain VLM outputs.

2. **Alignment with CNN predictions**: Where the CNN is correct, the VLM report typically supports the same conclusion. Where the CNN is wrong (e.g., Samples 41, 470, 144: GT Normal, CNN Pneumonia), the VLM repeatedly reported *clear lung fields* and *no consolidation/opacity*, i.e. agreeing with ground truth and contradicting the CNN’s false positive.

3. **Misclassification cases**: For CNN false positives, the VLM’s free-text reports can correct the picture (e.g., “lungs appear clear”) and help explain why the case was difficult for the CNN (subtle appearance, low resolution, or artifact).

---

## 5. Strengths and Limitations

### Strengths

- **Clinically relevant language**: MedGemma produces radiological-style language with anatomical and pathological terms.
- **Explainability**: Natural language reports improve transparency compared with classification-only models.
- **Integration**: The pipeline combines the Task 1 CNN (binary decision) with the VLM (explanation), supporting a two-stage workflow.
- **Reproducibility**: Open-source model and standard libraries; Colab-ready implementation.

### Limitations

- **Input resolution**: PneumoniaMNIST images are 28×28 grayscale. After upscaling to 224×224 for the VLM, fine anatomical details are limited. Performance would improve with full-resolution chest X-rays.
- **Compute**: MedGemma 4B requires a GPU (e.g., Colab T4). The 27B variant would need larger resources.
- **Gated access**: MedGemma requires accepting Hugging Face terms and, for automated use, a token.
- **Not for direct diagnosis**: Outputs are for research and education only; they must not replace clinical judgment.

---

## 6. Usage (Google Colab)

This task was run on **Google Colab**. Steps:

1. **Upload** to `/content/`: `report_generation.py`, `resnet34_pneumonia.pth` (or `resnet18_pneumonia.pth`).
2. **Runtime** → Change runtime type → **T4 GPU**.
3. **Install and login:**
   ```python
   !pip install transformers>=4.50 accelerate torch torchvision medmnist Pillow huggingface_hub
   from huggingface_hub import login
   login()  # Accept MedGemma terms at huggingface.co/google/medgemma-4b-it
   ```
4. **Run pipeline:**
   ```python
   !python report_generation.py --n-samples 12
   ```
5. **Outputs:** `generated_reports.json` (also at `/content/generated_reports.json`) and `task2_outputs/sample_images/`.

---

## 7. Environment: Google Colab

| Item | Detail |
|------|--------|
| **Platform** | Google Colab |
| **Runtime** | GPU (T4) |
| **Inputs** | `report_generation.py`, `resnet34_pneumonia.pth` uploaded to `/content/` |
| **Dependencies** | `transformers>=4.50`, `accelerate`, `torch`, `torchvision`, `medmnist`, `Pillow`, `huggingface_hub` |
| **Authentication** | `huggingface_hub.login()` (MedGemma terms accepted at Hugging Face) |
| **Outputs** | `generated_reports.json`, `task2_outputs/sample_images/` |

---

## References

- Sellergren et al. (2025). MedGemma Technical Report. arXiv:2507.05201.
- MedGemma on Hugging Face: https://huggingface.co/google/medgemma-4b-it
- MedMNIST: https://medmnist.com/
