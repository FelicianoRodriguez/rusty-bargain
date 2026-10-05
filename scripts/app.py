"""
app.py
-------
API REST con FastAPI que expone el modelo Linear Regression para inferencia.

Endpoints:
    GET /       -> health chack
    GET /       -> metadatos del modelo
    POST/       -> predice el precio del vehículo
    POST/       -> predicción para múltiples vehículos
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from inference import (
    VehicleFeatures,
    PricePrediction,
    get_metadata,
    predict,
    predict_batch
)

# ──────────────────────────────────────────────────────────────
# 1. SCHEMAS (Pydantic)
# ──────────────────────────────────────────────────────────────

class VehicleRequest(BaseModel):
    vehicle_type: str = Field(..., description="Tipo de vehículo.")
    registration_year: int = Field(..., ge=1980, le=2025, description="Fecha de registro.")
    gearbox: str = Field(..., description="Tipo de caja de cambios.")
    power: int = Field(..., ge=30, le=500, description="Potencia en HP.")
    model: str = Field(..., description="Modelo del vehículo.")
    mileage: int = Field(..., description="Millas recorridas.")
    registration_month: int = Field(..., description="Mes de registro.")
    fuel_type: str = Field(..., description="Tipo de combustible.")
    brand: str = Field(..., description="Marca del vehículo.")
    not_repaired: bool = Field(..., description="¿El vehículo ya fue raparado?")

    model_config = {
        "json_schema_extra": {
            "example": {
                "vehicle_type": "small",
                "registration_year": 2001,
                "gearbox": "manual",
                "power": 75,
                "model": "golf",
                "mileage": 150000,
                "registration_month": 6,
                "fuel_type": "petrol",
                "brand": "volkswagen",
                "not_repaired": False
            }
        }
    }

class BatchRequest(BaseModel):
    vehicles: list[VehicleRequest] = Field(..., min_length=1, max_length=500)

    model_config = {
        "json_schema_extra": {
            "example": {
                "vehicles": [
                    {
                        "vehicle_type": "small",
                        "registration_year": 2001,
                        "gearbox": "manual",
                        "power": 75,
                        "model": "golf",
                        "mileage": 150000,
                        "registration_month": 6,
                        "fuel_type": "petrol",
                        "brand": "volkswagen",
                        "not_repaired": False
                    },
                    {
                        "vehicle_type": "small",
                        "registration_year": 2008,
                        "gearbox": "manual",
                        "power": 68,
                        "model": "fabia",
                        "mileage": 90000,
                        "registration_month": 7,
                        "fuel_type": "gasoline",
                        "brand": "skoda",
                        "not_repaired": False
                    }
                ]
            }
        }
    }

class PredictionResponse(BaseModel):
    price: float = Field(..., description="Precio estimado del vehículo.")

class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]

class ModelInfoResponse(BaseModel):
    model: str
    features: list[str]
    mse: float
    mae: float
    rmse: float
    training_samples: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"

# ──────────────────────────────────────────────────────────────
# 2. LIFESPAN (carga el modelo al iniciar)
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-carga el modelo para que el primer request no sea lento
    try:
        get_metadata()
        print("Modelo Linear Regression cargado correctamente")
    except FileNotFoundError as e:
        print(f"    {e}")
    yield
    # Shutdown (nada que cerrar)

# ──────────────────────────────────────────────────────────────
# 3. APLICACIÓN FASTAPI
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Linear Regression Price Vehicle estimation API",
    description=("API para estimar el precio de vehículos"),
    version="1.0.0",
    lifespan=lifespan
)

# ──────────────────────────────────────────────────────────────
# 4. ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Verifica que la API y el modelo estén operativos."""
    try:
        get_metadata()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(status="ok", model_loaded=model_loaded)

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """
    Devuelve los metadatos del modelo entrenado:
    """
    try:
        meta = get_metadata()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return meta

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_price(body: VehicleFeatures):
    """
    Predice el precio del vehículo.
    """
    vehicle = VehicleFeatures(
        vehicle_type=body.vehicle_type,
        registration_year=body.registration_year,
        gearbox=body.gearbox,
        power=body.power,
        model=body.model,
        mileage=body.mileage,
        registration_month=body.registration_month,
        fuel_type=body.fuel_type,
        brand=body.brand,
        not_repaired=body.not_repaired
    )

    try:
        result: PricePrediction = predict(vehicle)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(
        price=result.price
    )

@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_price_batch(body: BatchRequest):
    """
    Predice el precio de múltiples vehículos en un sola llamada (máx. 500).
    """
    vehicles = [
        VehicleFeatures(
            vehicle_type=v.vehicle_type,
            registration_year=v.registration_year,
            gearbox=v.gearbox,
            power=v.power,
            model=v.model,
            mileage=v.mileage,
            registration_month=v.registration_month,
            fuel_type=v.fuel_type,
            brand=v.brand,
            not_repaired=v.not_repaired
        )
        for v in body.vehicles
    ]

    try:
        results = predict_batch(vehicles)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return BatchResponse(
        count=len(results),
        predictions=[
            PredictionResponse(price=r.price)
            for r in results
        ]
    )