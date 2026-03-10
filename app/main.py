from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import sys
import os
import importlib.util

# ────────────────────────────────────────────────
# Dynamically load model_loader.py (bypass import path issues)
# ────────────────────────────────────────────────
module_name = "model_loader"
module_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")

spec = importlib.util.spec_from_file_location(module_name, module_path)

if spec is None:
    raise ImportError(f"Could not find or load module at: {module_path}")

model_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_loader_module)

# Now access the class
ChurnPredictor = model_loader_module.ChurnPredictor

# ────────────────────────────────────────────────
# FastAPI app
# ────────────────────────────────────────────────

app = FastAPI(title="High-Risk Customer Churn Predictor")

# Instantiate the predictor once at startup
predictor = ChurnPredictor()

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]   # list of records (batch prediction)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        results = predictor.predict(request.data)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/single")
def predict_single(item: Dict[str, Any]):
    try:
        results = predictor.predict(item)
        return {"prediction": results[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Single prediction failed: {str(e)}")