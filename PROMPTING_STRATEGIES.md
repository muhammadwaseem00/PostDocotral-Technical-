# Prompting Strategies for Medical Report Generation

**Environment:** Google Colab. Tested with MedGemma on chest X-ray images (PneumoniaMNIST).

| Strategy | Prompt | Notes |
|----------|--------|-------|
| **minimal** | Describe this chest X-ray. | Short; may be vague |
| **detailed** | Provide a detailed radiological report of this chest X-ray. Describe lung fields, heart size, mediastinum, and any abnormalities. | More anatomical coverage |
| **structured** | As an expert radiologist, generate a structured report: (1) Technique/Finding, (2) Comparison if applicable, (3) Impression. Describe this chest X-ray. | Best format consistency |
| **clinical** | You are an expert radiologist. Analyze this chest X-ray and report: lung fields (consolidation, opacity, clarity), cardiac silhouette, bony structures. Note any findings suggesting pneumonia or normal appearance. | Best for pneumonia task |
| **open_ended** | What do you observe in this chest X-ray? List all relevant clinical findings. | Flexible; variable structure |

**Default:** The pipeline uses the `clinical` prompt by default. Use `--strategies minimal detailed clinical` to test specific strategies.
