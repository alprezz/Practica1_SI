from flask import Flask
from etl_process import run_etl, DB_NAME

import os
import sqlite3
import pandas as pd

app = Flask(__name__)



if not os.path.exists(DB_NAME):
    run_etl("datos.json")

# Retorna la información de tickets y contactos.
def get_full_tickets_df():
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT 
            t.id_ticket,
            t.fecha_apertura,
            t.fecha_cierre,
            t.es_mantenimiento,
            t.satisfaccion_cliente,
            t.id_inci,
            t.id_cliente,
            c.id_emp,
            c.fecha AS fecha_contacto,
            c.tiempo
        FROM incidencia_ticket t
        LEFT JOIN contacto c ON t.id_ticket = c.id_ticket
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['fecha_apertura'] = pd.to_datetime(df['fecha_apertura'])
    df['fecha_cierre']   = pd.to_datetime(df['fecha_cierre'])
    df['fecha_contacto'] = pd.to_datetime(df['fecha_contacto'], errors='coerce')
    df['duracion'] = (df['fecha_cierre'] - df['fecha_apertura']).dt.days
    df['tiempo'] = df['tiempo'].fillna(0).astype(float)
    return df

def calculate_metrics():
    df = get_full_tickets_df()

    total_time_by_ticket = df.groupby('id_ticket')['tiempo'].sum().reset_index(name='total_tiempo')
    tickets = df.drop_duplicates(subset=['id_ticket']).copy()
    tickets = tickets.merge(total_time_by_ticket, on='id_ticket', how='left')

    metrics = {}
    metrics['total_tickets'] = len(tickets)

    tickets['satisfaccion_ok'] = tickets['satisfaccion_cliente'] >= 5
    client_counts_ok = tickets.groupby('id_cliente')['satisfaccion_ok'].sum()
    metrics['incidents_satisfied_mean'] = round(client_counts_ok.mean(), 2) if len(client_counts_ok) else 0
    metrics['incidents_satisfied_std']  = round(client_counts_ok.std(), 2)  if len(client_counts_ok) > 1 else 0

    client_counts = tickets.groupby('id_cliente').size()
    metrics['incidents_per_client_mean'] = round(client_counts.mean(), 2) if len(client_counts) else 0
    metrics['incidents_per_client_std']  = round(client_counts.std(), 2)  if len(client_counts) > 1 else 0

    metrics['incident_total_time_mean'] = round(tickets['total_tiempo'].mean(), 2) if len(tickets) else 0
    metrics['incident_total_time_std']  = round(tickets['total_tiempo'].std(), 2)  if len(tickets) > 1 else 0

    emp_total_time = df.groupby('id_emp')['tiempo'].sum()
    metrics['employee_time_min'] = round(emp_total_time.min(), 2) if len(emp_total_time) else 0
    metrics['employee_time_max'] = round(emp_total_time.max(), 2) if len(emp_total_time) else 0

    metrics['resolution_time_min'] = int(tickets['duracion'].min()) if len(tickets) else 0
    metrics['resolution_time_max'] = int(tickets['duracion'].max()) if len(tickets) else 0

    emp_inci_df = df[['id_emp','id_ticket']].dropna().drop_duplicates()
    emp_inci_counts = emp_inci_df.groupby('id_emp').size()
    metrics['employee_incidents_min'] = int(emp_inci_counts.min()) if len(emp_inci_counts) else 0
    metrics['employee_incidents_max'] = int(emp_inci_counts.max()) if len(emp_inci_counts) else 0

    # Fraude
    fraude_tickets = tickets[tickets['id_inci'] == 5]
    metrics['fraude_ticket_count'] = len(fraude_tickets)
    fraude_df = df[df['id_inci'] == 5]
    fraude_contacts_by_ticket = fraude_df.groupby('id_ticket').size()
    if len(fraude_contacts_by_ticket):
        metrics['fraude_contacts_mean']   = round(fraude_contacts_by_ticket.mean(), 2)
        metrics['fraude_contacts_median'] = round(fraude_contacts_by_ticket.median(), 2)
        metrics['fraude_contacts_var']    = round(fraude_contacts_by_ticket.var(), 2)
        metrics['fraude_contacts_min']    = int(fraude_contacts_by_ticket.min())
        metrics['fraude_contacts_max']    = int(fraude_contacts_by_ticket.max())
    else:
        metrics['fraude_contacts_mean']   = 0
        metrics['fraude_contacts_median'] = 0
        metrics['fraude_contacts_var']    = 0
        metrics['fraude_contacts_min']    = 0
        metrics['fraude_contacts_max']    = 0

    return metrics


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
