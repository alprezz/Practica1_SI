import sqlite3
import json
import random
from datetime import datetime, timedelta

# Nombre del archivo de la base de datos
DB_NAME = "incidentes.db"

def run_etl(json_file_path: str = "datos.json"):
    """
    Ejecuta el proceso ETL para cargar datos desde 'datos.json' a la BD SQLite 'incidentes.db'.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    create_tables(conn)

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # === ALEATORIZAR LA FECHA DE CIERRE ===
    for ticket in data["tickets_emitidos"]:
        fecha_apertura_str = ticket["fecha_apertura"]
        # Convertir fecha_apertura a objeto datetime
        fecha_apertura = datetime.strptime(fecha_apertura_str, "%Y-%m-%d")

        # Generar un número aleatorio de días (p. ej. entre 1 y 10)
        dias_aleatorios = random.randint(1, 10)

        # Calcular la nueva fecha de cierre
        fecha_cierre = fecha_apertura + timedelta(days=dias_aleatorios)

        # Actualizar la fecha de cierre en el ticket
        ticket["fecha_cierre"] = fecha_cierre.strftime("%Y-%m-%d")
    # ======================================

    load_tipos_incidencia(data, conn)
    load_clientes(data, conn)
    load_empleados(data, conn)
    load_incidentes_y_contactos(data, conn)

    conn.close()
    print("Proceso ETL finalizado con éxito.")


def create_tables(conn):
    cursor = conn.cursor()

    # Tabla tipo_incidencia
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tipo_incidencia (
            id_inci INTEGER PRIMARY KEY,
            nombre TEXT
        )
    """)

    # Tabla cliente
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cliente (
            id_cliente INTEGER PRIMARY KEY,
            nombre TEXT,
            telefono TEXT,
            provincia TEXT
        )
    """)

    # Tabla empleado
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS empleado (
            id_emp INTEGER PRIMARY KEY,
            nombre TEXT,
            nivel INTEGER,
            fecha_contrato TEXT
        )
    """)

    # Tabla incidencia_ticket
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidencia_ticket (
            id_ticket INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_apertura TEXT,
            fecha_cierre TEXT,  
            es_mantenimiento BOOLEAN,
            satisfaccion_cliente INTEGER,
            id_inci INTEGER,
            id_cliente INTEGER,
            FOREIGN KEY (id_inci) REFERENCES tipo_incidencia(id_inci),
            FOREIGN KEY (id_cliente) REFERENCES cliente(id_cliente)
        )
    """)

    # Tabla contacto (entidad asociativa)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacto (
            id_contacto INTEGER PRIMARY KEY AUTOINCREMENT,
            id_ticket INTEGER,
            id_emp INTEGER,
            fecha TEXT,
            tiempo REAL,
            FOREIGN KEY (id_ticket) REFERENCES incidencia_ticket(id_ticket),
            FOREIGN KEY (id_emp) REFERENCES empleado(id_emp)
        )
    """)

    conn.commit()
    print("Tablas creadas o verificadas correctamente.")


def load_tipos_incidencia(data, conn):
    cursor = conn.cursor()
    tipos = data.get("tipos_incidentes", [])
    for tipo in tipos:
        id_inci = int(tipo["id_inci"])
        nombre = tipo["nombre"]

        cursor.execute("""
            INSERT OR IGNORE INTO tipo_incidencia (id_inci, nombre)
            VALUES (?, ?)
        """, (id_inci, nombre))

    conn.commit()
    print(f"Se han insertado/actualizado {len(tipos)} tipos de incidencia.")


def load_clientes(data, conn):
    cursor = conn.cursor()
    clientes = data.get("clientes", [])
    for cli in clientes:
        id_cliente = int(cli["id_cli"])
        nombre = cli["nombre"]
        telefono = cli["telefono"]
        provincia = cli["provincia"]

        cursor.execute("""
            INSERT OR IGNORE INTO cliente (id_cliente, nombre, telefono, provincia)
            VALUES (?, ?, ?, ?)
        """, (id_cliente, nombre, telefono, provincia))

    conn.commit()
    print(f"Se han insertado/actualizado {len(clientes)} clientes.")


def load_empleados(data, conn):
    cursor = conn.cursor()
    empleados = data.get("empleados", [])
    for emp in empleados:
        id_emp = int(emp["id_emp"])
        nombre = emp["nombre"]
        nivel = int(emp["nivel"])
        fecha_contrato = emp["fecha_contrato"]

        cursor.execute("""
            INSERT OR IGNORE INTO empleado (id_emp, nombre, nivel, fecha_contrato)
            VALUES (?, ?, ?, ?)
        """, (id_emp, nombre, nivel, fecha_contrato))

    conn.commit()
    print(f"Se han insertado/actualizado {len(empleados)} empleados.")


def load_incidentes_y_contactos(data, conn):
    cursor = conn.cursor()
    tickets = data.get("tickets_emitidos", [])

    for ticket in tickets:
        # Datos principales del incidente
        cliente = int(ticket["cliente"])
        fecha_apertura = ticket["fecha_apertura"]
        fecha_cierre = ticket["fecha_cierre"]
        es_mantenimiento = 1 if ticket["es_mantenimiento"] else 0
        satisfaccion = int(ticket["satisfaccion_cliente"])
        tipo_incidencia = int(ticket["tipo_incidencia"])

        # Insertar el ticket en 'incidencia_ticket'
        cursor.execute("""
            INSERT INTO incidencia_ticket 
            (fecha_apertura, fecha_cierre, es_mantenimiento, satisfaccion_cliente, id_inci, id_cliente)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (fecha_apertura, fecha_cierre, es_mantenimiento, satisfaccion, tipo_incidencia, cliente))

        # Obtener el ID autogenerado del ticket
        id_ticket = cursor.lastrowid

        # Insertar los contactos con empleados
        contactos = ticket.get("contactos_con_empleados", [])
        for c in contactos:
            id_emp = int(c["id_emp"])
            fecha_contacto = c["fecha"]
            tiempo = float(c["tiempo"])

            cursor.execute("""
                INSERT INTO contacto (id_ticket, id_emp, fecha, tiempo)
                VALUES (?, ?, ?, ?)
            """, (id_ticket, id_emp, fecha_contacto, tiempo))

    conn.commit()
    print(f"Se han insertado {len(tickets)} tickets y sus contactos.")


if __name__ == "__main__":
    run_etl("datos.json")
