import asyncio
import pickle
import time
from contextlib import asynccontextmanager
from typing import Annotated

from auth import require_api_key
from config import LOGISTIC_MODEL, RF_MODEL
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path
from pydantic import BaseModel, Field


class IrisData(BaseModel):
    sepal_length: float = Field(
        default=1.1, gt=0, lt=10, description="Sepal length is in range (0,10)"
    )
    sepal_width: float = Field(default=3.1, gt=0, lt=10)
    petal_length: float = Field(default=2.1, gt=0, lt=10)
    petal_width: float = Field(default=4.1, gt=0, lt=10)


ml_models = {}  # Global dictionary to hold the models.


def load_model(path: str):
    model = None
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models when the app starts
    ml_models["logistic_model"] = load_model(LOGISTIC_MODEL)
    ml_models["rf_model"] = load_model(RF_MODEL)

    yield

    # This code will be executed after the application finishes handling requests, right before the shutdown
    # Clean up the ML models and release the resources
    ml_models.clear()


# Create a FastAPI instance
app = FastAPI(lifespan=lifespan)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def list_models():
    print(LOGISTIC_MODEL)
    print(RF_MODEL)
    return {"available_models": list(ml_models.keys())}


@app.post("/predict/{model_name}")
async def predict(
    model_name: Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")],
    iris: IrisData,
    background_tasks: BackgroundTasks,
):
    # await asyncio.sleep(5) # Mimic heavy workload.

    input_data = [
        [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]
    ]

    if model_name not in ml_models.keys():
        raise HTTPException(status_code=404, detail="Model not found")

    ml_model = ml_models[model_name]
    prediction = ml_model.predict(input_data)

    background_tasks.add_task(
        log_prediction,
        {"model": model_name, "features": iris.model_dump(), "prediction": prediction},
    )

    return {"model": model_name, "prediction": int(prediction[0])}


def log_prediction(data: dict):
    time.sleep(5)  # mimic heavy work.
    print(f"Logging prediction: {data}")


@app.post("/predict_secure/{model_name}")
async def predict_secure(
    model_name: Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")],
    iris: IrisData,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
):
    # await asyncio.sleep(5) # Mimic heavy workload.

    input_data = [
        [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]
    ]

    if model_name not in ml_models.keys():
        raise HTTPException(status_code=404, detail="Model not found")

    ml_model = ml_models[model_name]
    prediction = ml_model.predict(input_data)

    background_tasks.add_task(
        log_prediction,
        {"model": model_name, "features": iris.model_dump(), "prediction": prediction},
    )

    return {"model": model_name, "prediction": int(prediction[0])}
