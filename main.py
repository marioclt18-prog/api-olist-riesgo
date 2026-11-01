from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Optional

app = FastAPI(title="API de Riesgo de Mala Reseña - Olist")

# Cargamos el modelo y el umbral al iniciar la API
modelo = joblib.load("modelo_riesgo.joblib")
with open("umbral.txt", "r") as f:
    UMBRAL = float(f.read())

# Definimos el "contrato" de datos usando Pydantic
class PedidoInput(BaseModel):
    order_id: str
    order_purchase_timestamp: str
    order_approved_at: Optional[str] = None
    order_delivered_carrier_date: Optional[str] = None
    order_delivered_customer_date: str
    order_estimated_delivery_date: str
    estado_cliente: str
    n_items: int
    n_vendedores: int
    precio_total: float = Field(..., ge=0) # No precios negativos (responde 422)
    flete_total: float = Field(..., ge=0)
    peso_total_g: float = Field(..., ge=0)
    fotos_promedio: float = Field(..., ge=0)
    categoria: str
    estado_vendedor: str
    tipo_pago: str
    cuotas_max: int

@app.get("/")
def inicio():
    return {"mensaje": "La API de Olist está viva y coleando"}

@app.post("/riesgo")
def predecir_riesgo(pedido: PedidoInput):
    try:
        # Convertimos el JSON a un DataFrame (como el que espera el modelo)
        datos = pd.DataFrame([pedido.model_dump()])
        
        # Predecimos la probabilidad
        prob = modelo.predict_proba(datos)[0][1]
        
        # Evaluamos si supera el umbral
        es_riesgo_alto = bool(prob >= UMBRAL)
        
        return {
            "probabilidad": round(float(prob), 4),
            "riesgo_alto": es_riesgo_alto,
            "umbral_usado": round(UMBRAL, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))