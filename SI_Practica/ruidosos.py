import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "incidentes.db"

def main():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Elimina la tabla si ya existía para evitar errores de columnas faltantes
    cursor.execute("DROP TABLE IF EXISTS metricas")

    # Crea la tabla metricas con las columnas solicitadas
    cursor.execute("""
        CREATE TABLE metricas (
            id_metricas INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_calculo TEXT,
            nombre_metrica TEXT,
            dato REAL
        )
    """)

    # Lee las tablas necesarias
    df_incidencia = pd.read_sql_query("SELECT * FROM incidencia_ticket", conn)
    df_contacto = pd.read_sql_query("SELECT * FROM contacto", conn)

    # Convierte a datetime para calcular diferencias
    df_incidencia['fecha_apertura'] = pd.to_datetime(df_incidencia['fecha_apertura'])
    df_incidencia['fecha_cierre'] = pd.to_datetime(df_incidencia['fecha_cierre'])

    # 1) Número de muestras totales
    num_muestras = len(df_incidencia)

    # 2) Media y std de incidentes con valoración >= 5
    inc_val5 = df_incidencia[df_incidencia['satisfaccion_cliente'] >= 5]
    media_val5 = inc_val5['satisfaccion_cliente'].mean()
    std_val5 = inc_val5['satisfaccion_cliente'].std()

    # 3) Media y std del número de incidentes por cliente
    inc_por_cliente = df_incidencia.groupby('id_cliente').size()
    media_incid_x_cliente = inc_por_cliente.mean()
    std_incid_x_cliente = inc_por_cliente.std()

    # 4) Media y std de horas totales por incidente
    horas_por_ticket = df_contacto.groupby('id_ticket')['tiempo'].sum()
    media_horas_incid = horas_por_ticket.mean()
    std_horas_incid = horas_por_ticket.std()

    # 5) Mínimo y máximo de horas totales por empleado
    horas_por_empleado = df_contacto.groupby('id_emp')['tiempo'].sum()
    min_horas_emp = horas_por_empleado.min()
    max_horas_emp = horas_por_empleado.max()

    # 6) Mínimo y máximo del tiempo entre apertura y cierre
    df_incidencia['tiempo_resolucion_horas'] = (
        df_incidencia['fecha_cierre'] - df_incidencia['fecha_apertura']
    ).dt.total_seconds() / 3600
    min_tiempo = df_incidencia['tiempo_resolucion_horas'].min()
    max_tiempo = df_incidencia['tiempo_resolucion_horas'].max()

    # 7) Mínimo y máximo del número de incidentes atendidos por cada empleado
    inc_por_empleado = df_contacto.groupby('id_emp')['id_ticket'].nunique()
    min_incid_emp = inc_por_empleado.min()
    max_incid_emp = inc_por_empleado.max()

    # Preparamos el diccionario de métricas
    metrics = {
        "num_muestras": num_muestras,
        "media_val5": media_val5,
        "std_val5": std_val5,
        "media_incid_x_cliente": media_incid_x_cliente,
        "std_incid_x_cliente": std_incid_x_cliente,
        "media_horas_incid": media_horas_incid,
        "std_horas_incid": std_horas_incid,
        "min_horas_emp": min_horas_emp,
        "max_horas_emp": max_horas_emp,
        "min_tiempo": min_tiempo,
        "max_tiempo": max_tiempo,
        "min_incid_emp": min_incid_emp,
        "max_incid_emp": max_incid_emp
    }

    fecha_calculo = datetime.now().isoformat()

    # Insertamos cada métrica como un registro individual
    for nombre_metrica, valor in metrics.items():
        if pd.isna(valor):
            valor = 0
        cursor.execute("""
            INSERT INTO metricas (fecha_calculo, nombre_metrica, dato)
            VALUES (?, ?, ?)
        """, (fecha_calculo, nombre_metrica, float(valor)))

    conn.commit()
    conn.close()

    print("He guardado las métricas en la base de datos. Fin.")

if __name__ == "__main__":
    main()
