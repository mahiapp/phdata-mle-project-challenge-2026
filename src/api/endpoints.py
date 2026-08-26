import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from utils import loader
from utils.imputation import impute_home_features

router = APIRouter()
logger = logging.getLogger(__name__)

class HomeFeatures(BaseModel):
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    sqft_living: Optional[float] = None
    sqft_lot: Optional[float] = None
    floors: Optional[float] = None
    sqft_above: Optional[float] = None
    sqft_basement: Optional[float] = None
    zipcode: str


@router.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration.
    Returns 200 if API is ready to accept requests.
    """
    if loader.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "healthy"}


@router.post("/predict")
async def predict(home_features: HomeFeatures):
    input_data = impute_home_features(home_features.dict())

    demographic_info = loader.demographics[
        loader.demographics["zipcode"] == home_features.zipcode
    ].drop(columns="zipcode").reset_index(drop=True)

    if demographic_info.empty:
        raise HTTPException(status_code=404, detail=f"Unknown zipcode: {home_features.zipcode}")

    input_data = pd.concat([input_data, demographic_info], axis=1)
    input_data = input_data[loader.model_features]
    prediction = loader.model.predict(input_data)

    logger.info("zipcode=%s predicted_price=%s", home_features.zipcode, prediction[0])
    return {"predicted_price": prediction[0]}
