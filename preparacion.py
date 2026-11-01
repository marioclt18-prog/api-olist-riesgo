import pandas as pd
import numpy as np

def crear_variables(df):
    df = df.copy()
    
    df['peso_total_g'] = df['peso_total_g'].fillna(df['peso_total_g'].median())
    df['fotos_promedio'] = df['fotos_promedio'].fillna(0)
    df['categoria'] = df['categoria'].fillna('desconocida')
    df['tipo_pago'] = df['tipo_pago'].fillna('desconocido')
    df['cuotas_max'] = df['cuotas_max'].fillna(1)
    
    df['order_approved_at'] = df['order_approved_at'].fillna(df['order_purchase_timestamp'])
    df['order_delivered_carrier_date'] = df['order_delivered_carrier_date'].fillna(df['order_purchase_timestamp'])

    cols_fechas = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                   'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in cols_fechas:
        df[col] = pd.to_datetime(df[col])

    df['dias_retraso'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['dias_envio'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['dias_preparacion'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days

    df['dias_retraso'] = df['dias_retraso'].fillna(0)
    df['dias_envio'] = df['dias_envio'].fillna(0)
    df['dias_preparacion'] = df['dias_preparacion'].fillna(0)

    return df[['estado_cliente', 'n_items', 'n_vendedores', 'precio_total', 'flete_total', 'peso_total_g', 
               'fotos_promedio', 'categoria', 'estado_vendedor', 'tipo_pago', 'cuotas_max', 
               'dias_retraso', 'dias_envio', 'dias_preparacion']]
