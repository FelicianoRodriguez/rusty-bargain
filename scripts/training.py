import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ──────────────────────────────────────────────────────────────
# 1. LECTURA DEL DATASET
# ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Retorna el dataset dividido en conjunto de entrenamiento y prueba.
    """

    df = pd.read_csv("../datasets/car_data.csv")

    # Filtros
    # --------------------------------------------
    # Filtrar registros con precio meno o igual a 0
    filter_price = df['Price'] > 0

    # Solo registros entre 1980 y 2025
    filter_year = df['RegistrationYear'].between(1980, 2025)

    # Solo vehículos con potencia entre 30 y 500
    filter_power = df['Power'].between(30, 500)

    df = df[
        filter_price &
        filter_year &
        filter_power
    ]

    # Nullos
    # -------------------------------------------
    cat_columns_with_nulls = [
        'VehicleType',
        'Gearbox',
        'Model',
        'FuelType',
        'NotRepaired'
    ]

    for column in cat_columns_with_nulls:
        df[column] = df[column].fillna('unknown')

    return df





# ──────────────────────────────────────────────────────────────
# 2. ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────
def train(output_dir: str = "model") -> dict:
    """
    Entrena el pipeline OneHotEncoding -> LinearRegressor y guarda los artefactos.

    Artefactos guardados en output_dir
    - onehot.pkl                : OneHotEncoding ajustado
    - linear_regressor.pkl      : modelo LinearRegressor entrneado
    - metadata.json             : métricas + configuración 

    Returns
    ----------
    dict con las métricas del modelo entrenado
    """
    os.makedirs(output_dir, exist_ok=True)

    # 2.1 Obtener los datos
    print("Cargando datos...")
    df = load_data()

    y = df['Price']

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

    X = df[features]
    print(f"    Shape: {X.shape}")

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

    # 2.2 conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    # 2.3 encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_train_final = pd.concat([
        X_train[numerical_columns].reset_index(drop=True),
        pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    ], axis=1)

    X_test_encoded = encoder.transform(X_test[categorical_columns])
    X_test_final = pd.concat([
        X_test[numerical_columns].reset_index(drop=True),
        pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    ], axis=1)

    # 2.4 linear regressor
    print(f"Entrenando modelo (regressión lineal)")
    model = LinearRegression()
    model.fit(X_train_final, y_train)

    # 2.5 predicciones, conjunto de prueba
    y_pred = model.predict(X_test_final)

    # 2.6 Métricas
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Métricas del modelo")
    print(f"Error cuadratico medio:             {mse}")
    print(f"Error Absoluto Medio:               {mae}")
    print(f"Raiz Error Cuadratico Medio:        {rmse}")

    # 2.7 Persistencia
    onehot_path = os.path.join(output_dir, "onehot.pkl")
    model_path  = os.path.join(output_dir, "linear_regressor.pkl")
    meta_path   = os.path.join(output_dir, "metadata.json")

    with open(onehot_path, "wb") as f:
        pickle.dump(encoder, f)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model": "Linear Regressor",
        "features": features,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "training_samples": len(df)
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artefactos guardados en '{output_dir}':")
    print(f"    - {onehot_path}")
    print(f"    - {model_path}")
    print(f"    - {meta_path}")

    return metadata


# ──────────────────────────────────────────────────────────────
# 3. MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo Linear Regressor")
    parser.add_argument("--output", type=str, default="model", help="Directorio de salida (default: model)")
    args = parser.parse_args()

    train(output_dir=args.output)