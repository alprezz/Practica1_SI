import os
import sqlite3
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for
from etl_process import run_etl, DB_NAME

app = Flask(__name__)


# Inicialización ETL
if not os.path.exists(DB_NAME):
    run_etl("../../Si_P2Grok/datos.json")


def get_full_tickets_df():
    """
    Retorna un DataFrame con la información de tickets + contactos.
    """
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

def get_empleados_df():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT id_emp, nombre, nivel FROM empleado", conn)
    conn.close()
    return df

def get_clientes_df():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT id_cliente, nombre FROM cliente", conn)
    conn.close()
    return df


# Cálculo de métricas generales
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

# Agrupaciones Fraude
def calculate_fraude_groupings():
    """
    Filtra los incidentes de tipo_incidencia = 5 y agrupa por:
      - Empleado
      - Nivel de empleado
      - Cliente
      - Día de la semana (fecha_contacto)
    Para cada grupo calcula:
      - Nº de incidentes (tickets)
      - Nº total de contactos
      - Estadísticas (# contactos por ticket): mediana, media, varianza, min, max
    """
    df = get_full_tickets_df()

    # Solo Fraude
    df_fraude = df[df['id_inci'] == 5].copy()
    if df_fraude.empty:
        # Si no hay ningún ticket de Fraude, devolvemos dict vacío
        return {
            'by_employee': [],
            'by_level': [],
            'by_client': [],
            'by_weekday': []
        }

    # Día de la semana
    df_fraude['weekday'] = df_fraude['fecha_contacto'].dt.day_name().fillna('Desconocido')

    # Unimos nivel de empleado
    emp_df = get_empleados_df()[['id_emp','nivel','nombre']]
    df_fraude = df_fraude.merge(emp_df, on='id_emp', how='left')

    # Por Empleado (id_emp)
    by_employee = do_fraude_stats_by_dimension(df_fraude, 'id_emp')
    # Mapeamos el ID a su nombre
    emp_dict = dict(zip(emp_df['id_emp'], emp_df['nombre']))
    for row in by_employee:
        emp_id = row['group_value']
        row['group_value'] = emp_dict.get(emp_id, f"Emp {emp_id}")

    # Por Nivel
    by_level = do_fraude_stats_by_dimension(df_fraude, 'nivel')
    # (El 'group_value' ya es 1,2,3, no hace falta mapear)

    # Por Cliente
    by_client = do_fraude_stats_by_dimension(df_fraude, 'id_cliente')
    # Mapeamos ID cliente -> nombre
    cli_df = get_clientes_df()
    cli_dict = dict(zip(cli_df['id_cliente'], cli_df['nombre']))
    for row in by_client:
        cid = row['group_value']
        row['group_value'] = cli_dict.get(cid, f"Cliente {cid}")

    # Por día de la semana
    by_weekday = do_fraude_stats_by_dimension(df_fraude, 'weekday')

    return {
        'by_employee': by_employee,
        'by_level': by_level,
        'by_client': by_client,
        'by_weekday': by_weekday
    }

def do_fraude_stats_by_dimension(df_fraude, group_col):

    # Agrupamos por [group_col, id_ticket] y contamos filas => # contactos
    grouped = df_fraude.groupby([group_col, 'id_ticket']).size().reset_index(name='num_contacts')

    results = []
    for group_value, subdf in grouped.groupby(group_col):
        dist = subdf['num_contacts']  # Serie con el # de contactos por ticket
        num_incidents = len(dist)     # Nº de tickets (cada fila = un ticket)
        total_contacts = dist.sum()   # Suma total de contactos
        median_val = dist.median()
        mean_val = dist.mean()
        var_val = dist.var() if len(dist) > 1 else 0
        min_val = dist.min()
        max_val = dist.max()

        results.append({
            'group_value': group_value,
            'num_incidents': int(num_incidents),
            'total_contacts': int(total_contacts),
            'median_contacts': round(median_val, 2),
            'mean_contacts': round(mean_val, 2),
            'var_contacts': round(var_val, 2),
            'min_contacts': int(min_val),
            'max_contacts': int(max_val)
        })

    return results


# Generar gráficos
def generate_charts():
    df = get_full_tickets_df()
    total_time_by_ticket = df.groupby('id_ticket')['tiempo'].sum().reset_index(name='total_tiempo')
    tickets = df.drop_duplicates(subset=['id_ticket']).copy()
    tickets = tickets.merge(total_time_by_ticket, on='id_ticket', how='left')

    chart_folder = os.path.join('static', 'charts')
    os.makedirs(chart_folder, exist_ok=True)

    # Gráfico 1
    group = tickets.groupby('es_mantenimiento')['duracion'].mean()
    plt.figure()
    group.plot(kind='bar', color=['#007bff','#ffc107'])
    plt.title('Tiempo promedio (días) por mantenimiento')
    plt.xlabel('Es Mantenimiento (0=No, 1=Sí)')
    plt.ylabel('Tiempo Promedio (días)')
    chart1_filename = 'charts/chart1.png'
    chart1_path = os.path.join('static', chart1_filename)
    plt.tight_layout()
    plt.savefig(chart1_path)
    plt.close()

    # Gráfico 2
    groups = tickets.groupby('id_inci')
    box_data = []
    labels = []
    for tipo, df_tipo in groups:
        box_data.append(df_tipo['duracion'].values)
        labels.append(str(tipo))
    plt.figure()
    plt.boxplot(box_data, whis=[5, 90], labels=labels)
    plt.title('Boxplot tiempos de resolución (por tipo_incidencia)')
    plt.xlabel('Tipo de Incidencia')
    plt.ylabel('Duración (días)')
    chart2_filename = 'charts/chart2.png'
    chart2_path = os.path.join('static', chart2_filename)
    plt.tight_layout()
    plt.savefig(chart2_path)
    plt.close()

    # Gráfico 3 (Top 5 clientes críticos)
    crit_df = tickets[(tickets['es_mantenimiento'] == 1) & (tickets['id_inci'] != 1)]
    crit_counts = crit_df.groupby('id_cliente').size().sort_values(ascending=False).head(5)
    conn = sqlite3.connect(DB_NAME)
    cli_df = pd.read_sql_query("SELECT id_cliente, nombre FROM cliente", conn)
    conn.close()
    cli_dict = dict(zip(cli_df['id_cliente'], cli_df['nombre']))
    crit_counts.index = crit_counts.index.map(lambda x: cli_dict.get(x, f"Cliente {x}"))
    plt.figure()
    crit_counts.plot(kind='bar', color='#dc3545')
    plt.title('Top 5 clientes críticos')
    plt.xlabel('Cliente')
    plt.ylabel('Nº incidencias críticas')
    chart3_filename = 'charts/chart3.png'
    chart3_path = os.path.join('static', chart3_filename)
    plt.tight_layout()
    plt.savefig(chart3_path)
    plt.close()

    # Gráfico 4 (Actuaciones por empleado)
    emp_contact_counts = df[df['id_emp'].notna()].groupby('id_emp').size()
    conn = sqlite3.connect(DB_NAME)
    emp_df = pd.read_sql_query("SELECT id_emp, nombre FROM empleado", conn)
    conn.close()
    emp_dict = dict(zip(emp_df['id_emp'], emp_df['nombre']))
    emp_contact_counts.index = emp_contact_counts.index.map(lambda x: emp_dict.get(x, f"Emp {x}"))
    plt.figure()
    emp_contact_counts.plot(kind='bar', color='#17a2b8')
    plt.title('Total actuaciones por empleado')
    plt.xlabel('Empleado')
    plt.ylabel('Nº actuaciones')
    chart4_filename = 'charts/chart4.png'
    chart4_path = os.path.join('static', chart4_filename)
    plt.tight_layout()
    plt.savefig(chart4_path)
    plt.close()

    # Gráfico 5 (Actuaciones por día de la semana)
    df['weekday'] = df['fecha_contacto'].dt.day_name()
    weekday_counts = df['weekday'].value_counts()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_counts = weekday_counts.reindex(order).dropna()
    plt.figure()
    weekday_counts.plot(kind='bar', color='#6f42c1')
    plt.title('Actuaciones por día de la semana')
    plt.xlabel('Día')
    plt.ylabel('Nº actuaciones')
    chart5_filename = 'charts/chart5.png'
    chart5_path = os.path.join('static', chart5_filename)
    plt.tight_layout()
    plt.savefig(chart5_path)
    plt.close()

    return {
        'chart1': chart1_filename,
        'chart2': chart2_filename,
        'chart3': chart3_filename,
        'chart4': chart4_filename,
        'chart5': chart5_filename
    }


# Rutas Flask
@app.route('/')
def index():
    metrics = calculate_metrics()
    charts = generate_charts()

    # NUEVO: cálculo de agrupaciones para Fraude
    fraude_groupings = calculate_fraude_groupings()

    return render_template('index.html',
                           metrics=metrics,
                           charts=charts,
                           fraude_groupings=fraude_groupings)

@app.route('/add_incidente', methods=['GET','POST'])
def add_incidente():
    if request.method == 'POST':
        # Recogemos datos del incidente
        cliente = request.form.get('cliente')
        fecha_apertura = request.form.get('fecha_apertura')
        fecha_cierre   = request.form.get('fecha_cierre')
        es_mant = 1 if request.form.get('es_mantenimiento') == 'true' else 0
        satisfaccion = int(request.form.get('satisfaccion_cliente'))
        tipo_inci = request.form.get('tipo_incidencia')

        # Recogemos datos del contacto
        id_emp = request.form.get('id_emp')
        fecha_contacto = request.form.get('fecha_contacto')
        tiempo_contacto = float(request.form.get('tiempo_contacto', 0))

        # Insertar en la BD
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO incidencia_ticket
            (fecha_apertura, fecha_cierre, es_mantenimiento, satisfaccion_cliente, id_inci, id_cliente)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (fecha_apertura, fecha_cierre, es_mant, satisfaccion, tipo_inci, cliente))
        id_ticket = cur.lastrowid

        # Insertar contacto
        cur.execute("""
            INSERT INTO contacto (id_ticket, id_emp, fecha, tiempo)
            VALUES (?, ?, ?, ?)
        """, (id_ticket, id_emp, fecha_contacto, tiempo_contacto))

        conn.commit()
        conn.close()

        return redirect(url_for('index'))

    else:
        conn = sqlite3.connect(DB_NAME)
        clientes = pd.read_sql_query("SELECT id_cliente, nombre FROM cliente", conn).to_dict('records')
        tipos = pd.read_sql_query("SELECT id_inci, nombre FROM tipo_incidencia", conn).to_dict('records')
        empleados = pd.read_sql_query("SELECT id_emp, nombre FROM empleado", conn).to_dict('records')
        conn.close()

        return render_template('add_incidente.html',
                               clientes=clientes,
                               tipos_incidentes=tipos,
                               empleados=empleados)

#PRACTICA 2
@app.route('/top_clientes/<int:x>')
def top_clientes(x):

    df = get_full_tickets_df()

    # Agrupar por cliente y contar incidencias
    client_incidents = df.groupby('id_cliente').size().sort_values(ascending=False).head(x)

    # Obtener nombres de clientes
    clients_df = get_clientes_df()
    client_dict = dict(zip(clients_df['id_cliente'], clients_df['nombre']))
    client_incidents.index = client_incidents.index.map(lambda x: client_dict.get(x, f"Cliente {x}"))

    # Convertir a lista de diccionarios para la plantilla
    top_clients_list = [
        {'nombre': client_name, 'incidencias': int(count)}
        for client_name, count in client_incidents.items()
    ]

    return render_template('top_clientes.html',
                           top_clientes=top_clients_list,
                           x=x)


@app.route('/top_tiempos_incidencias/<int:x>')
def top_tiempos_incidencias(x):

    df = get_full_tickets_df()

    # Calcular el tiempo promedio de resolución por tipo de incidencia
    incident_times = df.groupby('id_inci')['duracion'].mean().sort_values(ascending=False).head(x)

    # Obtener nombres de tipos de incidencias
    conn = sqlite3.connect(DB_NAME)
    incident_types = pd.read_sql_query("SELECT id_inci, nombre FROM tipo_incidencia", conn)
    conn.close()
    incident_dict = dict(zip(incident_types['id_inci'], incident_types['nombre']))
    incident_times.index = incident_times.index.map(lambda x: incident_dict.get(x, f"Tipo {x}"))

    # Convertir a lista de diccionarios para la plantilla
    top_incidents_list = [
        {'tipo': incident_name, 'dias_promedio': round(time, 2)}
        for incident_name, time in incident_times.items()
    ]

    return render_template('top_tiempos_incidencias.html',
                           top_incidencias=top_incidents_list,
                           x=x)


@app.route('/top_reportes/<int:x>', defaults={'mostrar_empleados': 'no'})
@app.route('/top_reportes/<int:x>/<mostrar_empleados>')
def top_reportes(x, mostrar_empleados):
    """
    Muestra el top X de clientes con más incidencias reportadas y, opcionalmente,
    el top X de empleados con más tiempo empleado en resolución de incidencias
    """
    df = get_full_tickets_df()

    # --- Top X Clientes con más incidencias ---
    # Agrupar por cliente y contar incidencias
    client_incidents = df.groupby('id_cliente').size().sort_values(ascending=False).head(x)

    # Obtener nombres de clientes
    clients_df = get_clientes_df()
    client_dict = dict(zip(clients_df['id_cliente'], clients_df['nombre']))
    client_incidents.index = client_incidents.index.map(lambda x: client_dict.get(x, f"Cliente {x}"))

    # Convertir a lista de diccionarios
    top_clientes_list = [
        {'nombre': client_name, 'incidencias': int(count)}
        for client_name, count in client_incidents.items()
    ]

    # --- Top X Empleados con más tiempo (si se solicita) ---
    top_empleados_list = None
    if mostrar_empleados.lower() == 'si':
        # Agrupar por empleado y sumar el tiempo total empleado (en horas)
        employee_times = df.groupby('id_emp')['tiempo'].sum().sort_values(ascending=False).head(x)

        # Obtener nombres de empleados
        empleados_df = get_empleados_df()
        employee_dict = dict(zip(empleados_df['id_emp'], empleados_df['nombre']))
        employee_times.index = employee_times.index.map(lambda x: employee_dict.get(x, f"Empleado {x}"))

        # Convertir a lista de diccionarios
        top_empleados_list = [
            {'nombre': emp_name, 'horas': round(hours, 2)}
            for emp_name, hours in employee_times.items()
        ]

    return render_template('top_reportes.html',
                           top_clientes=top_clientes_list,
                           top_empleados=top_empleados_list,
                           x=x,
                           mostrar_empleados=(mostrar_empleados.lower() == 'si'))
if __name__ == '__main__':
    app.run(debug=True)
