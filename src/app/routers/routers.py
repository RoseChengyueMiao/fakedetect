import uuid
from logging import getLogger
from typing import Any, Dict, List
import json
from fastapi import APIRouter
import pandas as pd
from src.ml.prediction import Data, classifier
from src.ml.predict_fake import load_models_and_run_inference

logger = getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}


@router.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "data_type": "string",
        "data_structure": "(1)",
        "input_language": "japanese",
        "prediction_type": "float32",
        "prediction_structure": "(1)",
        "prediction_sample": [0.97093159],
    }


@router.get("/label")
def label() -> Dict[int, str]:
    with open('../../../models/label.json', "r") as f:
        label= json.load(f)
    return label



@router.post("/predict")
def predict(data: str) -> Dict[str, List[float]]:
    job_id = str(uuid.uuid4())
    prediction = load_models_and_run_inference(pd.DataFrame({"context": [data]}))
    prediction_list =prediction
    logger.info(f"{job_id}: {prediction_list}")
    return {"prediction": prediction_list}


