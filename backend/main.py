from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated
import pandas as pd
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model file not found")

app = FastAPI( version="1.0")

class CropInput(BaseModel):
    Nitrogen: Annotated[float, Field(..., description='Nitrogen in kg/ha')]
    Phosphorus: Annotated[float, Field(..., description='Phosphorous in kg/ha')]
    Potassium: Annotated[float, Field(..., description='Potassium in kg/ha')]
    Temperature: Annotated[float, Field(..., gt=0, lt=100, description='Temperature in Celsius')]
    Humidity: Annotated[float, Field(..., description='Humidity in %')]
    pH_Value: Annotated[float, Field(..., gt=0, lt=14, description='pH value of soil')]
    Rainfall: Annotated[float, Field(..., description='Rainfall in mm')]

@app.get("/")
def home():
    return {"message": "Welcome to Crop Prediction API"}

@app.post("/predict")
def predict(data: CropInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([data.dict()])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {"predicted_crop": str(prediction)}
