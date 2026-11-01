from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Optional
from preparacion import crear_variables

app = FastAPI(title="API de Riesgo de Mala Reseña - Olist")

# Cargamos las 3 piezas
preprocesador = joblib.load("preprocesador.joblib")
modelo_xgb = joblib.load("modelo_xgb.joblib")
with open("umbral.txt", "r") as f:
    UMBRAL = float(f.read())

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
    precio_total: float = Field(..., ge=0)
    flete_total: float = Field(..., ge=0)
    peso_total_g: float = Field(..., ge=0)
    fotos_promedio: float = Field(..., ge=0)
    categoria: str
    estado_vendedor: str
    tipo_pago: str
    cuotas_max: int

@app.get("/")
def inicio():
    return {"mensaje": "La API de Olist está viva"}

@app.post("/riesgo")
def predecir_riesgo(pedido: PedidoInput):
    try:
        # 1. Convertimos a DataFrame
        datos = pd.DataFrame([pedido.model_dump()])
        
        # 2. Aplicamos nuestra función de limpieza
        datos_limpios = crear_variables(datos)
        
        # 3. Pasamos por el preprocesador (Escalador + OneHotEncoder)
        datos_procesados = preprocesador.transform(datos_limpios)
        
        # 4. Predecimos con el modelo puro
        prob = modelo_xgb.predict_proba(datos_procesados)[0][1]
        es_riesgo_alto = bool(prob >= UMBRAL)
        
        return {
            "probabilidad": round(float(prob), 4),
            "riesgo_alto": es_riesgo_alto,
            "umbral_usado": round(UMBRAL, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))