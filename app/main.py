import os
import pickle
from contextlib import asynccontextmanager
from typing import Annotated
import logging
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
ml_models = {}


class IrisData(BaseModel):
    sepal_length: float = Field(
        default=1.1, gt=0, lt=10, description="Sepal length is in range (0,10)"
    )
    sepal_width: float = Field(default=3.1, gt=0, lt=10)
    petal_length: float = Field(default=2.1, gt=0, lt=10)
    petal_width: float = Field(default=4.1, gt=0, lt=10)

def load_model(path):
    try:
        if not path:
            raise ValueError("Model path is None")
        model = joblib.load(path)
        if model is None:
            raise ValueError(f"Loaded model from {path} is None")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logistic_path = os.getenv("LOGISTIC_MODEL", "models/logistic_model.pkl")
    rf_path = os.getenv("RF_MODEL", "models/rf_model.pkl")

    if not logistic_path or not rf_path:
        logger.error(f"Missing environment variables: LOGISTIC_MODEL={logistic_path}, RF_MODEL={rf_path}")
        raise ValueError("Model paths not set in environment variables")

    ml_models["logistic_model"] = load_model(logistic_path)
    ml_models["rf_model"] = load_model(rf_path)
    
    logger.debug(f"Loaded models: {ml_models.keys()}")
    logger.debug(f"rf_model: {ml_models['rf_model']}")
    
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()
    logger.debug("Models cleared")


# Create a FastAPI instance
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    return {"available_models": list(ml_models.keys())}


@app.post("/predict/{model_name}")
async def predict(
    model_name: Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")],
    iris_data: IrisData,
):
    input_data = [
        [
            iris_data.sepal_length,
            iris_data.sepal_width,
            iris_data.petal_length,
            iris_data.petal_width,
        ]
    ]

    if model_name not in ml_models.keys():
        raise HTTPException(status_code=404, detail="Model not found.")

    model = ml_models[model_name]
    prediction = model.predict(input_data)

    return {"model": model_name, "prediction": int(prediction[0])}
