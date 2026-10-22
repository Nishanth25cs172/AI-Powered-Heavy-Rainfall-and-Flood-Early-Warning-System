"""
SIH26071 – Backend API Request & Response Data Schemas
Defines structured input schemas and type validation models for Rainfall, Flood, and Compound Risk endpoints.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class RainfallPredictionRequest(BaseModel):
    avg_temp: float = Field(..., description="Average atmospheric temperature (°C)")
    wind_speed: float = Field(..., description="Surface wind speed (km/h)")
    air_pressure: float = Field(..., description="Barometric pressure (hPa)")
    latitude: float = Field(..., description="Station Latitude (°N)")
    longitude: float = Field(..., description="Station Longitude (°E)")
    elevation: float = Field(..., description="Terrain elevation above sea level (m)")

class FloodPredictionRequest(BaseModel):
    rainfall_mm: float = Field(..., alias="Rainfall (mm)", description="Cumulative 24h rainfall (mm)")
    water_level_m: float = Field(..., alias="Water Level (m)", description="River stage level (m)")
    river_discharge: float = Field(..., alias="River Discharge (m³/s)", description="River volumetric discharge rate (m³/s)")
    elevation: float = Field(..., alias="Elevation (m)", description="Terrain elevation (m)")
    latitude: float = Field(..., alias="Latitude", description="Latitude (°N)")
    longitude: float = Field(..., alias="Longitude", description="Longitude (°E)")
    land_cover: str = Field(..., alias="Land Cover", description="Land use / land cover category")
    soil_type: str = Field(..., alias="Soil Type", description="Dominant soil classification")

class CompoundRiskRequest(BaseModel):
    avg_temp: Optional[float] = 25.0
    wind_speed: Optional[float] = 10.0
    air_pressure: Optional[float] = 1013.0
    rainfall_mm: Optional[float] = 50.0
    water_level_m: Optional[float] = 5.0
    river_discharge: Optional[float] = 1000.0
    elevation: Optional[float] = 50.0
    latitude: Optional[float] = 20.0
    longitude: Optional[float] = 78.0
    land_cover: Optional[str] = "Urban"
    soil_type: Optional[str] = "Clay"
