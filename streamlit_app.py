import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Olist - Riesgo de Mala Reseña", layout="wide")

st.title("🛒 Olist: Del pedido a la reseña")
st.write("Esta app consume la API para predecir el riesgo de una mala reseña.")

# Menú lateral
menu = st.sidebar.selectbox("Navegación", ["Riesgo de un pedido", "Comentarios (Deep Learning)", "Ficha del Modelo"])

if menu == "Riesgo de un pedido":
    st.header("Evaluar Riesgo de un Pedido")
    
    # Formulario
    col1, col2 = st.columns(2)
    with col1:
        estado = st.selectbox("Estado del Cliente", ["SP", "RJ", "MG", "RS", "PR"])
        categoria = st.text_input("Categoría del producto", "toys")
        precio = st.number_input("Precio Total (R$)", value=100.0)
    with col2:
        flete = st.number_input("Flete (R$)", value=20.0)
        cuotas = st.number_input("Cuotas máximas", value=1)
    
    if st.button("Predecir Riesgo 🚀"):
        # Construimos el JSON con el contrato exacto
        pedido = {
            "order_id": "test-123", "order_purchase_timestamp": "2018-01-14 14:33:31",
            "order_approved_at": "2018-01-14 14:48:30", "order_delivered_carrier_date": "2018-01-16 12:36:48",
            "order_delivered_customer_date": "2018-01-22 13:19:16", "order_estimated_delivery_date": "2018-01-22 00:00:00",
            "estado_cliente": estado, "n_items": 1, "n_vendedores": 1, "precio_total": precio, "flete_total": flete,
            "peso_total_g": 500.0, "fotos_promedio": 2.0, "categoria": categoria, "estado_vendedor": estado,
            "tipo_pago": "credit_card", "cuotas_max": cuotas
        }
        
        try:
            with st.spinner("Consultando la API..."):
                # AQUÍ VA TU URL DE RENDER
                url = "https://api-olist-riesgo-aw9s.onrender.com/riesgo"
                res = requests.post(url, json=pedido, timeout=60)
                
                if res.status_code == 200:
                    data = res.json()
                    prob = data['probabilidad']
                    st.metric(label="Probabilidad de Mala Reseña", value=f"{prob*100:.1f}%")
                    
                    if data['riesgo_alto']:
                        st.error("🚨 **ALTO RIESGO:** Llamar al cliente para resolver problemas.")
                    else:
                        st.success("✅ **RIESGO BAJO:** No es necesario llamar.")
                else:
                    st.error(f"Error en la API: {res.text}")
        except Exception as e:
            st.error(f"Error de conexión: {e}")

elif menu == "Comentarios (Deep Learning)":
    st.header("Análisis de Sentimiento")
    st.write("El modelo preentrenado (XLM-RoBERTa) analizó 1000 comentarios en portugués.")
    st.info("📊 Resultados: 70.5% de coincidencia con las estrellas. Se encontraron 295 desacuerdos (ej. clientes que ponen 5 estrellas pero se quejan del retraso).")

elif menu == "Ficha del Modelo":
    st.header("Ficha del Modelo")
    st.markdown("""
    - **Modelo:** XGBoost (versión 2.1.1).
    - **Métrica principal:** 64.94% de precisión en el 10% de mayor riesgo.
    - **Uso:** Priorizar llamadas de atención al cliente.
    - **Límites:** Puede fallar en periodos atípicos (como el paro de camioneros de mayo 2018).
    """)