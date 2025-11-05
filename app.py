from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

app = FastAPI(
    title="API de Predicción de Precios",
    description="Predice el precio futuro de una acción usando regresión lineal.",
    version="1.0.0"
)

# Modelo de entrada
class InputData(BaseModel):
    ticker: str = "AAPL"  # Símbolo de la acción
    dias: int = 180       # Días de historial para el entrenamiento

# Ruta raíz
@app.get("/")
def home():
    return {
        "mensaje": "✅ Bienvenido a la API de predicción de precios.",
        "endpoints_disponibles": ["/predecir_precio (POST)", "/docs"]
    }

# Endpoint principal
@app.post("/predecir_precio")
def predecir_precio(data: InputData):
    try:
        ticker = data.ticker.upper()
        dias = data.dias

        # Descargar datos históricos de Yahoo Finance
        df = yf.download(ticker, period=f"{int(dias/30)}mo", interval="1d")
        if df.empty:
            return {"resultado": f"No se encontraron datos para {ticker}."}

        df = df.reset_index()
        df = df[['Date', 'Close']]
        df['Day'] = np.arange(len(df))

        # Entrenar modelo simple
        X = df[['Day']]
        y = df['Close']

        model = LinearRegression()
        model.fit(X, y)

        # Guardar modelo y datos
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/modelo_precio.pkl")
        df.to_csv("model/data_procesada.csv", index=False)

        # Predicción para el día siguiente
        next_day = np.array([[len(df) + 1]])
        prediccion = model.predict(next_day).item()

        return {
            "ticker": ticker,
            "dias_usados": dias,
            "prediccion_usd": round(prediccion, 2)
        }

    except Exception as e:
        return {"error": str(e)}
