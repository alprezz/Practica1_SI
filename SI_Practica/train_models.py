import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Cargar datos clasificados
with open('data_clasified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Crear DataFrame con las características relevantes
tickets = data['tickets_emitidos']
df_tickets = []

for ticket in tickets:
    # Datos básicos del ticket
    ticket_data = {
        'cliente': int(ticket['cliente']),
        'es_mantenimiento': 1 if ticket['es_mantenimiento'] else 0,
        'satisfaccion_cliente': ticket['satisfaccion_cliente'],
        'tipo_incidencia': ticket['tipo_incidencia'],
        'es_critico': 1 if ticket['es_critico'] else 0
    }

    # Calcular duración en días
    fecha_apertura = datetime.strptime(ticket['fecha_apertura'], '%Y-%m-%d')
    fecha_cierre = datetime.strptime(ticket['fecha_cierre'], '%Y-%m-%d')
    ticket_data['duracion'] = (fecha_cierre - fecha_apertura).days

    # Características de los contactos
    ticket_data['num_contactos'] = len(ticket['contactos_con_empleados'])
    ticket_data['tiempo_total'] = sum(contacto['tiempo'] for contacto in ticket['contactos_con_empleados'])

    df_tickets.append(ticket_data)

# Crear DataFrame
df = pd.DataFrame(df_tickets)

# Variables independientes y dependiente
X = df[['es_mantenimiento', 'satisfaccion_cliente', 'tipo_incidencia', 'duracion', 'num_contactos', 'tiempo_total']]
y = df['es_critico']

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Entrenamiento del modelo
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluación
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Precisión: {lr_accuracy:.4f}")
print(classification_report(y_test, lr_pred))

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Regresión Logística')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.savefig('static/charts/lr_confusion_matrix.png')

# Visualización de coeficientes
plt.figure(figsize=(10, 6))
coef = pd.Series(lr_model.coef_[0], index=X.columns)
coef.sort_values().plot(kind='barh')
plt.title('Importancia de Características - Regresión Logística')
plt.savefig('static/charts/lr_feature_importance.png')

# Guardar el modelo
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')


# Entrenamiento del modelo
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluación
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Precisión: {dt_accuracy:.4f}")
print(classification_report(y_test, dt_pred))

# Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Matriz de Confusión - Árbol de Decisión')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.savefig('static/charts/dt_confusion_matrix.png')

# Visualización del árbol
plt.figure(figsize=(15, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['No Crítico', 'Crítico'],
          filled=True, rounded=True, fontsize=10)
plt.title('Visualización del Árbol de Decisión')
plt.savefig('static/charts/decision_tree.png')

# Importancia de características
plt.figure(figsize=(10, 6))
feat_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh')
plt.title('Importancia de Características - Árbol de Decisión')
plt.savefig('static/charts/dt_feature_importance.png')

# Guardar el modelo
joblib.dump(dt_model, 'models/decision_tree_model.pkl')

from sklearn.ensemble import RandomForestClassifier

# Entrenamiento del modelo
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluación
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Precisión: {rf_accuracy:.4f}")
print(classification_report(y_test, rf_pred))

# Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.savefig('static/charts/rf_confusion_matrix.png')

# Importancia de características
plt.figure(figsize=(10, 6))
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh')
plt.title('Importancia de Características - Random Forest')
plt.savefig('static/charts/rf_feature_importance.png')

# Guardar el modelo
joblib.dump(rf_model, 'models/random_forest_model.pkl')
