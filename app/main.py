import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

app = FastAPI(title="Iris Classifier")

try:
    model = joblib.load("model/model.pkl")
except FileNotFoundError:
    model = None


class PredictRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("features must have exactly 4 values")
        return v


class PredictResponse(BaseModel):
    cls: int
    label: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array(req.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    return PredictResponse(cls=prediction, label=IRIS_LABELS[prediction])
