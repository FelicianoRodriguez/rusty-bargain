"""
inference.py
----------------
Carga el modelo linear regressor entrenado y expone funciones de inferencia
para ser consumida por app.py (FastAPI) u otros módulos.

Uso directo (CLI de prueba):
    python scripts/inference.py --vehicle-type small --gearbox manual --model golf --fuel-type petrol --brand
 wolkswagen --not-repaired no --registration-year 2001 --power 75 --mileage 150000 --registration-month 6     
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE RUTAS
# ──────────────────────────────────────────────────────────────

MODEL_DIR = os.getenv("MODEL_DIR", "model")
ONEHOT_PATH = os.path.join(MODEL_DIR, "onehot.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "linear_regressor.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ──────────────────────────────────────────────────────────────
# 2. TIPOS DE DATOS
# ──────────────────────────────────────────────────────────────

@dataclass
class VehicleFeatures:
    vehicle_type: str
    registration_year: int
    gearbox: str
    power: int
    model: str
    mileage: int
    registration_month: int
    fuel_type: str
    brand: str
    not_repaired: bool

@dataclass
class PricePrediction:
    price: int

# ──────────────────────────────────────────────────────────────
# 3. CARGA LAZY DEL MODELO (singleton)
# ──────────────────────────────────────────────────────────────

_onehot = None
_model = None
_metadata = None

def _load_artifacts():
    """Carga los artefactos del modelo una sola vez (lazy loading)."""
    global _onehot, _model, _metadata

    if _model is not None:
        return

    for path in (ONEHOT_PATH, MODEL_PATH, META_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artefacto no encontrado: '{path}'. "
                "Ejecuta primero: python training.py"
            )

    with open(ONEHOT_PATH, "rb") as f:
        _onehot = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(META_PATH, "r") as f:
        _metadata = json.load(f)

def get_metadata() -> dict:
    """Devuelve los metadatos del modelo cargado."""
    _load_artifacts()
    return _metadata


# ──────────────────────────────────────────────────────────────
# 4. FUNCIÓN PRINCIPAL DE INFERENCIA
# ──────────────────────────────────────────────────────────────

def predict(vehicle: VehicleFeatures) -> PricePrediction:
    """
    Predice el precio de un vehiculo.

    Parameters
    -------------
    vehicle: VehicleFeatures
        Características del vehiculo que se desea predecir su precio.
    
    Returns
    -------------
    PricePrediction
        - price : precio aproximado del vehiculo.

    """
    _load_artifacts()

    # Validaciones básicas
    if not (1980 <= vehicle.registration_year <= 2025):
        raise ValueError(f"La fecha de registro debe estar entre 1980 y 2025")
    if not (30 <= vehicle.power <= 500):
        raise ValueError(f"La potencia debe estar entre 30 y 500 hp")

    # Preparar vector de features
    X = np.array([[
        vehicle.vehicle_type,
        vehicle.registration_year,
        vehicle.gearbox,
        vehicle.power,
        vehicle.model,
        vehicle.mileage,
        vehicle.registration_month,
        vehicle.fuel_type,
        vehicle.brand,
        vehicle.not_repaired
    ]])

    features = [
        'VehicleType',
        'RegistrationYear',
        'Gearbox',
        'Power',
        'Model',
        'Mileage',
        'RegistrationMonth',
        'FuelType',
        'Brand',
        'NotRepaired'
    ]

    row = pd.DataFrame(X, columns=features)
    print(row)

    numerical_columns = [
        'RegistrationYear',
        'Power',
        'Mileage',
        'RegistrationMonth'
    ]

    categorical_columns = [
        'VehicleType',
        'Gearbox',
        'Model',
        'FuelType',
        'Brand',
        'NotRepaired'
    ]

    # Transformar
    X_encoded = _onehot.transform(row[categorical_columns])
    X_final = pd.concat([
        row[numerical_columns].reset_index(drop=True),
        pd.DataFrame(X_encoded, columns=_onehot.get_feature_names_out(categorical_columns))
    ], axis=1)


    # Predecir precio
    price = _model.predict(X_final)

    return PricePrediction(price=price)

def predict_batch(vehicles: list[VehicleFeatures]) -> list[PricePrediction]:
    """
    Versión batch: predice el cluster para múltiples clientes a la vez.
    """
    _load_artifacts()
    return [predict(c) for c in vehicles]

# ──────────────────────────────────────────────────────────────
# 5. CLI DE PRUEBA
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia Linear Regressor (prueba rápida)")
    parser.add_argument("--vehicle-type", type=str, required=True, help="Tipo de vehículo.")
    parser.add_argument("--gearbox", type=str, required=True, help="Caja de cambios.")
    parser.add_argument("--model", type=str, required=True, help="Modelo del vehículo.")
    parser.add_argument("--fuel-type", type=str, required=True, help="Tipo de gasolina.")
    parser.add_argument("--brand", type=str, required=True, help="Marca del vehículo.")
    parser.add_argument("--not-repaired", type=bool, required=True, help="El vehículo no fue reparado.")
    parser.add_argument("--registration-year", type=int, required=True, help="Año de registro.")
    parser.add_argument("--power", type=int, required=True, help="Potencia.")
    parser.add_argument("--mileage", type=int, required=True, help="Millas recorridas.")
    parser.add_argument("--registration-month", type=int, required=True, help="Mes de registro.")
    args = parser.parse_args()

    vehicle = VehicleFeatures(
        vehicle_type=args.vehicle_type,
        gearbox=args.gearbox,
        model=args.model,
        fuel_type=args.fuel_type,
        brand=args.brand,
        not_repaired=args.not_repaired,
        registration_year=args.registration_year,
        registration_month=args.registration_month,
        power=args.power,
        mileage=args.mileage
    )

    result = predict(vehicle)

    print("\n Resultado de la inferencia:")
    print(f"    Precio: {result.price}")

