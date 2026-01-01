import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightning.model import MNISTClassifier

app = FastAPI(title="MNIST Classifier API", version="1.0.0")

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "../checkpoints/mnist_classifier/epochepoch=08-val_accval_acc=0.9886.ckpt")
model = None


class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model():
    global model
    try:
        model = MNISTClassifier.load_from_checkpoint(MODEL_PATH)
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(status="healthy", model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image.
    Expects a grayscale or RGB image that will be converted to 28x28 MNIST format.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize((28, 28))
        
        # Convert to tensor and normalize (MNIST normalization)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        # Normalize using MNIST stats
        mean = 0.1307
        std = 0.3081
        img_tensor = (img_tensor - mean) / std
        
        # Inference
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = probabilities.argmax(dim=1).item()
            probs_list = probabilities.squeeze().tolist()
        
        return PredictionResponse(
            prediction=int(prediction),
            probabilities=probs_list
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MNIST Classifier API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with image file)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
