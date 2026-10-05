# Linear Regresor Price Predict API

API REST construida con **FastAPI** que expone un modelo lineal
**Linear Regressor** para predecir el precio de un vehículo.


---

## Estructura del proyecto

```
rusty-bargain/
├── training.py       # Entrena y persiste el modelo LinearRegressor
├── inference.py      # Carga el modelo y expone funciones de predicción
├── app.py            # API FastAPI con los endpoints REST
├── Dockerfile        # Imagen Docker multi-stage (mínima)
└── model/            # Generado automáticamente al entrenar
    ├── linear_regressor.pkl
    ├── onehot.pkl
    └── metadata.json
```

---

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) ≥ 24.x

---

## Gestión del contenedor

```bash
docker build -t rusty_bargain .
```

> El modelo se entrena **durante el build**, por lo que la imagen ya
incluye los artefactos listos para servir.

-----

## 2. Correr el contenedor
```bash
docker run -d \
    --name rusty_bargain \
    -p 8000:8000 \
    rusty_bargain
```