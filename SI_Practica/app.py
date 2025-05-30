import json
import os
import sqlite3
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import io
import joblib
import logging
import requests
import re
from flask import Flask, render_template, request, redirect, url_for, send_file
from requests.adapters import HTTPAdapter
from tenacity import wait_fixed, stop_after_attempt, retry_if_exception_type, retry
from urllib3 import Retry
from etl_process import run_etl, DB_NAME
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

matplotlib.use("Agg")
app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización ETL
if not os.path.exists(DB_NAME):
    run_etl("datos.json")


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
      - N.º de incidentes (tickets)
      - N.º total de contactos
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
    plt.boxplot(box_data, whis=[5, 90], tick_labels=labels)  # Changed 'labels' to 'tick_labels'
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
@app.route('/top_clientes/<int:x>', defaults={'x': 5})
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


@app.route('/top_tiempos_incidencias/<int:x>', defaults={'x': 5})
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

# Ejercicio 3

def cveinfo(cve):
    customheaders = {
        "User-Agent": "Some script trying to be nice :)"
    }
    try:
        res = requests.get("http://cve.circl.lu/api/cve/%s" % (cve.upper()), headers=customheaders)
        if res.status_code == 200:
            reply = res.json()
            if len(reply):
                # Buscar la descripción en inglés
                description = next(
                    (d.get("value") for d in reply["containers"]["cna"].get("descriptions", []) if
                     d.get("lang") == "en"),
                    "No hay descripción disponible"
                )
                return {
                    "cve": cve.upper(),
                    "summary": description,
                    "published": reply["cveMetadata"].get("datePublished", "Fecha no disponible")
                }
        return {
            "success": False,
            "reason": "expected HTTP 200 status code but got %d instead for requesturl" % (res.status_code)
        }
    except Exception as ex:
        return {
            "success": False,
            "exception": str(ex)  # Cambio a str(ex) en lugar de ex.message
        }

def cverecent(maxcves=0):
    customheaders = {
        "User-Agent": "Some script trying to be nice :)"
    }
    try:
        res = requests.get("http://cve.circl.lu/api/last", headers=customheaders)
        if res.status_code == 200:
            reply = res.json()  # Usar directamente res.json()
            cves = list()
            for node in reply:
                if "REJECT" not in node.get("summary", ""):
                    if node.get("id", "").startswith("CVE"):
                        cves.append(node.get("id", ""))
            return {
                "success": True,
                "cves": cves if maxcves == 0 else cves[:maxcves]
            }
        return {
            "success": False,
            "reason": "expected HTTP 200 status code but got %d instead for requesturl" % (res.status_code)
        }
    except Exception as ex:
        return {
            "success": False,
            "exception": str(ex)
        }


@app.route('/vulnerabilidades')
def mostrar_vulnerabilidades():
    """Vista para mostrar las últimas 10 vulnerabilidades"""
    cves_result = cverecent(10)

    # Verificar si el resultado es exitoso
    if not cves_result.get("success", False):
        error_message = cves_result.get("reason", "Error desconocido al obtener vulnerabilidades")
        if "exception" in cves_result:
            error_message += f": {cves_result['exception']}"
        return render_template('vulnerabilidades.html', error_message=error_message)

    # Obtener la lista de CVEs
    cves = cves_result.get("cves", [])
    vulnerabilidades = []

    for cve in cves:
        resultado = cveinfo(cve)
        if isinstance(resultado, dict) and "cve" in resultado:
            vulnerabilidades.append(resultado)

    return render_template('vulnerabilidades.html', vulnerabilidades=vulnerabilidades)


# Ejercicio 4

def header_footer(canvas, doc):
    # Header: logo y título pequeño
    logo_path = os.path.join('static', 'logo.png')
    if os.path.exists(logo_path):
        canvas.drawImage(logo_path, x=2*cm, y=A4[1]-3*cm, width=3*cm, height=1*cm, preserveAspectRatio=True)
    canvas.setFont('Helvetica-Bold', 12)
    canvas.setFillColor(colors.HexColor('#2E4053'))
    canvas.drawString(6*cm, A4[1]-2.5*cm, "Informe de Incidencias - URJC - Sistemas de Información")

    # Footer: página y fecha
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.grey)
    page_num = f"Página {doc.page}"
    canvas.drawRightString(A4[0]-2*cm, 1.5*cm, page_num)
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    canvas.drawString(2*cm, 1.5*cm, f"Generado el {fecha}")

@app.route('/generate_report')
def generate_report():
    metrics = calculate_metrics()
    charts = generate_charts()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=4*cm, bottomMargin=3*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1F618D'), spaceAfter=14)
    subtitle_style = ParagraphStyle('subtitle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2874A6'), spaceAfter=12)
    normal_style = styles['BodyText']

    story = []

    # Título principal
    story.append(Paragraph("Informe de Incidencias", title_style))
    story.append(Spacer(1, 12))

    # Resumen breve (opcional)
    story.append(Paragraph("Este informe presenta un análisis detallado de las incidencias registradas, con métricas clave y gráficos que facilitan la interpretación.", normal_style))
    story.append(Spacer(1, 18))

    # Tabla de métricas con zebra striping
    data = [["Métrica", "Valor"]]
    for key, value in metrics.items():
        key_formatted = key.replace('_', ' ').capitalize()
        data.append([key_formatted, str(value)])

    table = Table(data, hAlign='LEFT', colWidths=[10*cm, 5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980B9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        # Alternar color filas
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    # Zebra striping manual
    for i in range(1, len(data)):
        if i % 2 == 0:
            table.setStyle(TableStyle([('BACKGROUND', (0, i), (-1, i), colors.lavender)]))

    story.append(table)
    story.append(Spacer(1, 24))

    # Sección de gráficos en cuadrícula 2xN
    story.append(Paragraph("Análisis Gráfico", subtitle_style))
    story.append(Spacer(1, 12))

    chart_items = list(charts.items())
    for i in range(0, len(chart_items), 2):
        row = []
        for j in range(2):
            if i + j < len(chart_items):
                chart_name, chart_file = chart_items[i + j]
                img_path = os.path.join('static', chart_file)
                if os.path.exists(img_path):
                    img = Image(img_path, width=8*cm, height=6*cm)
                    # Contenedor con título y gráfico
                    block = [Paragraph(chart_name.replace('_', ' ').capitalize(), normal_style), Spacer(1,6), img]
                    row.append(block)
                else:
                    row.append([Paragraph("Imagen no disponible", normal_style)])
            else:
                row.append('')  # Celda vacía si no hay par
        # Crear tabla para la fila de gráficos
        t = Table([row], colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    # Construir PDF con header y footer
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='informe_incidencias.pdf', mimetype='application/pdf')

# Ejercicio 5
# Función para cargar los modelos
def load_models():
    try:
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        dt_model = joblib.load('models/decision_tree_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        return {'lr': lr_model, 'dt': dt_model, 'rf': rf_model}
    except Exception as e:
        print(f"Error al cargar los modelos: {e}")
        return None


# Ruta para la página de predicción
@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    models = load_models()

    if not models:
        return render_template('error.html', message="No se pudieron cargar los modelos de IA.")

    if request.method == 'POST':
        # Obtener datos del formulario
        cliente = request.form.get('cliente')
        fecha_apertura = request.form.get('fecha_apertura')
        fecha_cierre = request.form.get('fecha_cierre')
        es_mantenimiento = 1 if request.form.get('es_mantenimiento') == 'true' else 0
        satisfaccion_cliente = int(request.form.get('satisfaccion_cliente'))
        tipo_incidencia = int(request.form.get('tipo_incidencia'))
        modelo_seleccionado = request.form.get('modelo')

        # Calcular características
        fecha_apertura_dt = datetime.strptime(fecha_apertura, '%Y-%m-%d')
        fecha_cierre_dt = datetime.strptime(fecha_cierre, '%Y-%m-%d')
        duracion = (fecha_cierre_dt - fecha_apertura_dt).days

        # Crear DataFrame para predicción
        data = {
            'es_mantenimiento': [es_mantenimiento],
            'satisfaccion_cliente': [satisfaccion_cliente],
            'tipo_incidencia': [tipo_incidencia],
            'duracion': [duracion],
            'num_contactos': [1],  # Por defecto
            'tiempo_total': [1.0]  # Por defecto
        }
        df = pd.DataFrame(data)

        # Seleccionar modelo y hacer predicción
        if modelo_seleccionado == 'lr':
            model = models['lr']
            model_name = "Regresión Logística"
            chart_feature = 'lr_feature_importance.png'
            chart_confusion = 'lr_confusion_matrix.png'
        elif modelo_seleccionado == 'dt':
            model = models['dt']
            model_name = "Árbol de Decisión"
            chart_feature = 'dt_feature_importance.png'
            chart_confusion = 'dt_confusion_matrix.png'
            chart_tree = 'decision_tree.png'
        else:  # rf
            model = models['rf']
            model_name = "Random Forest"
            chart_feature = 'rf_feature_importance.png'
            chart_confusion = 'rf_confusion_matrix.png'

        # Predicción
        prediccion = model.predict(df)[0]
        probabilidad = model.predict_proba(df)[0][1]  # Probabilidad de ser crítico

        # Preparar resultados
        resultado = {
            'cliente': cliente,
            'es_mantenimiento': 'Sí' if es_mantenimiento == 1 else 'No',
            'satisfaccion_cliente': satisfaccion_cliente,
            'tipo_incidencia': tipo_incidencia,
            'duracion': duracion,
            'modelo': model_name,
            'es_critico': 'Sí' if prediccion == 1 else 'No',
            'probabilidad': round(probabilidad * 100, 2),
            'chart_feature': f'charts/{chart_feature}',
            'chart_confusion': f'charts/{chart_confusion}'
        }

        if modelo_seleccionado == 'dt':
            resultado['chart_tree'] = f'charts/{chart_tree}'

        return render_template('resultado_prediccion.html', resultado=resultado)

    # Para petición GET, mostrar formulario
    conn = sqlite3.connect(DB_NAME)
    clientes = pd.read_sql_query("SELECT id_cliente, nombre FROM cliente", conn).to_dict('records')
    tipos = pd.read_sql_query("SELECT id_inci, nombre FROM tipo_incidencia", conn).to_dict('records')
    conn.close()

    return render_template('prediccion.html', clientes=clientes, tipos_incidentes=tipos)


if __name__ == '__main__':
    app.run(debug=True)

