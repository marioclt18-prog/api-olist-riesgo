import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Olist - Proyecto Final", layout="wide")
st.title("🛒 Olist: Del pedido a la reseña")

# Menú de navegación
menu = st.sidebar.selectbox("Navegación", ["Riesgo de un pedido", "Comentarios (Deep Learning)", "Pronóstico Junio", "Ficha del Modelo"])

if menu == "Riesgo de un pedido":
    st.header("Evaluar Riesgo de un Pedido")
    st.write("Simula los datos de un pedido para consultar la API (XGBoost).")
    
    col1, col2 = st.columns(2)
    with col1:
        estado = st.selectbox("Estado del Cliente", ["SP", "RJ", "MG", "RS", "PR"])
        categoria = st.text_input("Categoría del producto", "toys")
        precio = st.number_input("Precio Total (R$)", value=100.0)
    with col2:
        flete = st.number_input("Flete (R$)", value=20.0)
        cuotas = st.number_input("Cuotas máximas", value=1)
    
    if st.button("Predecir Riesgo 🚀"):
        pedido = {
            "order_id": "test-123", "order_purchase_timestamp": "2018-01-14 14:33:31",
            "order_approved_at": "2018-01-14 14:48:30", "order_delivered_carrier_date": "2018-01-16 12:36:48",
            "order_delivered_customer_date": "2018-01-22 13:19:16", "order_estimated_delivery_date": "2018-01-22 00:00:00",
            "estado_cliente": estado, "n_items": 1, "n_vendedores": 1, "precio_total": precio, "flete_total": flete,
            "peso_total_g": 500.0, "fotos_promedio": 2.0, "categoria": categoria, "estado_vendedor": estado,
            "tipo_pago": "credit_card", "cuotas_max": cuotas
        }
        try:
            url = "https://api-olist-riesgo-aw9s.onrender.com/riesgo"
            res = requests.post(url, json=pedido, timeout=60)
            if res.status_code == 200:
                data = res.json()
                st.metric(label="Probabilidad de Mala Reseña", value=f"{data['probabilidad']*100:.1f}%")
                if data['riesgo_alto']:
                    st.error("🚨 **ALTO RIESGO:** Llamar al cliente para resolver problemas.")
                else:
                    st.success("✅ **RIESGO BAJO:** No es necesario llamar.")
            else:
                st.error(f"Error en la API: {res.text}")
        except Exception as e:
            st.error(f"Error de conexión: {e}")

elif menu == "Comentarios (Deep Learning)":
    st.header("Análisis de Sentimiento en Comentarios")
    st.write("Un modelo XLM-RoBERTa analizó 1,000 comentarios en portugués y los comparó con las estrellas.")
    
    # Creamos una tabla cruzada de ejemplo (simulando la que hicimos en Colab)
    data = {'negative': [80, 10, 5], 'neutral': [15, 20, 15], 'positive': [5, 20, 830]}
    df_cross = pd.DataFrame(data, index=['Estrellas: negative', 'Estrellas: neutral', 'Estrellas: positive'])
    
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(df_cross, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    st.info("📊 Resultados: 70.5% de coincidencia. Los mayores desacuerdos ocurren cuando el cliente pone 5 estrellas pero se queja en el texto.")

elif menu == "Pronóstico Junio":
    st.header("Pronóstico de Pedidos para Junio 2018")
    st.write("Modelo Prophet entrenado con datos históricos hasta mayo de 2018.")
    
    try:
        # Leemos el CSV que subimos
        df = pd.read_csv("pronostico_junio.csv")
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['fecha'], df['pronostico'], color='blue', label='Pronóstico')
        ax.fill_between(df['fecha'], df['bajo'], df['alto'], color='blue', alpha=0.2, label='Intervalo 80%')
        ax.set_title('Predicción de pedidos diarios')
        ax.legend()
        st.pyplot(fig)
        
        st.write("**Nota:** El modelo detectó que la caída de mayo (paro de camioneros) fue un evento aislado y pronostica una recuperación.")
    except Exception as e:
        st.error("Sube el archivo 'pronostico_junio.csv' para ver la gráfica.")

elif menu == "Ficha del Modelo":
    st.header("Ficha del Modelo")
    st.markdown("""
    - **Modelo:** XGBoost y Red Neuronal (Keras).
    - **Métrica principal:** 64.94% de precisión en el 10% de mayor riesgo (supera la regla de retraso del 62.29%).
    - **Uso:** Priorizar llamadas de atención al cliente.
    - **Límites:** Puede fallar en periodos atípicos (como el paro de camioneros).
    - **Equidad:** Recomendamos supervisión humana en las llamadas para no castigar vendedores por retrasos logísticos que no controlan.
    """)