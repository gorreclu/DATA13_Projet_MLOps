"""Pydantic schemas for the FastAPI serving API."""

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """Input features for house price prediction."""

    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=15000, description="Living space (sqft)")
    sqft_lot: int = Field(..., ge=100, description="Lot size (sqft)")
    floors: float = Field(..., ge=1, le=4, description="Number of floors")
    waterfront: int = Field(0, ge=0, le=1, description="Waterfront property (0/1)")
    view: int = Field(0, ge=0, le=4, description="View quality (0-4)")
    condition: int = Field(3, ge=1, le=5, description="Condition (1-5)")
    grade: int = Field(7, ge=1, le=13, description="Construction grade (1-13)")
    sqft_above: int = Field(..., ge=0, description="Above ground sqft")
    sqft_basement: int = Field(0, ge=0, description="Basement sqft")
    yr_built: int = Field(..., ge=1900, le=2030, description="Year built")
    yr_renovated: int = Field(0, ge=0, description="Year renovated (0 if never)")
    zipcode: int = Field(..., description="ZIP code")
    lat: float = Field(..., description="Latitude")
    long: float = Field(..., description="Longitude")
    sqft_living15: int = Field(..., ge=0, description="Avg living sqft of 15 neighbors")
    sqft_lot15: int = Field(..., ge=0, description="Avg lot sqft of 15 neighbors")
    year_sold: int = Field(2015, ge=2014, le=2030, description="Year of sale")
    month_sold: int = Field(6, ge=1, le=12, description="Month of sale")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bedrooms": 3,
                    "bathrooms": 2.5,
                    "sqft_living": 2000,
                    "sqft_lot": 5000,
                    "floors": 2.0,
                    "waterfront": 0,
                    "view": 0,
                    "condition": 3,
                    "grade": 8,
                    "sqft_above": 1500,
                    "sqft_basement": 500,
                    "yr_built": 1990,
                    "yr_renovated": 0,
                    "zipcode": 98103,
                    "lat": 47.6516,
                    "long": -122.3480,
                    "sqft_living15": 1800,
                    "sqft_lot15": 5000,
                    "year_sold": 2015,
                    "month_sold": 6,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response with predicted price."""

    predicted_price: float = Field(..., description="Predicted price in USD")
    formatted_price: str = Field(..., description="Formatted price string")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    mlflow_uri: str
