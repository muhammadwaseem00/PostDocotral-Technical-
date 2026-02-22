"""
Task 3: Optional FastAPI server for inference.
Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000
POST /predict with form-data: file=<image file>
"""
from pathlib import Path
import tempfile
import os

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn python-multipart")

from inference_pipeline import load_pipeline, run_inference

app = FastAPI(title="Task 3: Pneumonia + Report API", version="1.0")

# Load pipeline once at startup (CNN only by default; set USE_VLM=1 for VLM)
USE_VLM = os.environ.get("USE_VLM", "0").strip().lower() in ("1", "true", "yes")
_pipeline = None


@app.on_event("startup")
def startup():
    global _pipeline
    try:
        _pipeline = load_pipeline(use_vlm=USE_VLM)
    except Exception as e:
        print(f"Pipeline load failed: {e}")
        _pipeline = None


@app.get("/")
def root():
    return {"message": "POST /predict with image file", "vlm_enabled": USE_VLM}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    suffix = Path(file.filename or "img").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = run_inference(_pipeline, tmp_path, run_vlm=USE_VLM)
        return JSONResponse(result)
    finally:
        os.unlink(tmp_path)
