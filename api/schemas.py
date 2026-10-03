from pydantic import BaseModel, Field
from typing import List

class HousePricePredictionRequest(BaseModel):
    sqft: float = Field(...,gt=0, description="sq ft of house")
    bedrooms: int = Field(..., ge=1, description="number of Bedrooms")
    bathrooms: float = Field(...,gt=0, description="number of bathrooms")
    location: str = Field(...,description="Location (urban,suburban, rural)")
    year_built: int = Field(...,ge=1970,le=2026,description="year the house was built")
    condition: str = Field(...,description="Condition of the house")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: List[float]
    features_importance: dict
    prediction_time: str