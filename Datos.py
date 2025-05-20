import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos (asumiendo que ya generaste el archivo con el script anterior)
# Si no has generado el archivo, primero ejecuta el código del generador de datos
df = pd.read_csv('datos_credito.csv')

# Exploración básica
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas del dataset:")
print(df.head())

# Distribución de la variable objetivo
print("\nDistribución de la variable objetivo:")
print(df['Credito_Aprobado'].value_counts())
print(df['Credito_Aprobado'].value_counts(normalize=True).round(2))

# Separamos predictores y variable objetivo
X = df.drop('Credito_Aprobado', axis=1)
y = df['Credito_Aprobado']

# Identificar columnas por tipo
columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
columnas_categoricas = X.select_dtypes(include=['object']).columns.tolist()

print("\nColumnas numéricas:", columnas_numericas)
print("Columnas categóricas:", columnas_categoricas)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Preprocesamiento de datos
# 1. Para columnas numéricas: Estandarización
# 2. Para columnas categóricas: One-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), columnas_categoricas)
    ])

# Crear pipeline con preprocesamiento y modelo KNN
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # Probamos con k=5 inicialmente
])

# Entrenar el modelo
print("\nEntrenando modelo KNN...")
knn_pipeline.fit(X_train, y_train)

# Evaluar el modelo
y_pred = knn_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.4f}")

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
print("\nMatriz de confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()

# Encontrar el mejor valor de k
k_range = range(1, 31)
k_scores = []

print("\nBuscando el mejor valor de k...")
for k in k_range:
    knn = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k))
    ])
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    k_scores.append(score)
    print(f"k = {k}, precisión = {score:.4f}")

# Visualizar los resultados de diferentes valores de k
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores)
plt.xlabel('Valor de k')
plt.ylabel('Precisión')
plt.title('Precisión del modelo KNN para diferentes valores de k')
plt.grid(True)
plt.show()

# Obtener el mejor valor de k
mejor_k = k_range[np.argmax(k_scores)]
mejor_score = max(k_scores)
print(f"\nEl mejor valor de k es {mejor_k} con una precisión de {mejor_score:.4f}")

# Reentrenar con el mejor valor de k
mejor_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=mejor_k))
])
mejor_knn.fit(X_train, y_train)

# Evaluar modelo final
y_pred_final = mejor_knn.predict(X_test)
print("\nReporte final de clasificación:")
print(classification_report(y_test, y_pred_final))

# Ejemplo de uso del modelo
print("\n--- Ejemplo de uso del modelo ---")
print("Predecir para un solicitante nuevo:")

# Crear un solicitante de ejemplo
nuevo_solicitante = pd.DataFrame({
    'Edad': [35],
    'Ingresos_Mensuales': [42000],
    'Antiguedad_Laboral': [7],
    'Estado_Civil': ['Casado/a'],
    'Nivel_Educativo': ['Universidad'],
    'Sector_Laboral': ['Privado'],
    'Tiene_Propiedad': ['Sí'],
    'Tiene_Vehiculo': ['Sí'],
    'Saldo_Cuenta': [85000],
    'Deuda_Actual': [15000],
    'Monto_Solicitado': [120000],
    'Historial_Crediticio': ['Bueno']
})

print(nuevo_solicitante)
prediccion = mejor_knn.predict(nuevo_solicitante)
proba = mejor_knn.predict_proba(nuevo_solicitante)

print(f"\nPredicción: {prediccion[0]}")
print(f"Probabilidad: {max(proba[0]):.2f}")

# Solicitante con perfil distinto
print("\nOtro ejemplo con perfil distinto:")
otro_solicitante = pd.DataFrame({
    'Edad': [22],
    'Ingresos_Mensuales': [15000],
    'Antiguedad_Laboral': [1],
    'Estado_Civil': ['Soltero/a'],
    'Nivel_Educativo': ['Bachillerato'],
    'Sector_Laboral': ['Privado'],
    'Tiene_Propiedad': ['No'],
    'Tiene_Vehiculo': ['No'],
    'Saldo_Cuenta': [5000],
    'Deuda_Actual': [8000],
    'Monto_Solicitado': [80000],
    'Historial_Crediticio': ['Limitado']
})

print(otro_solicitante)
prediccion2 = mejor_knn.predict(otro_solicitante)
proba2 = mejor_knn.predict_proba(otro_solicitante)

print(f"\nPredicción: {prediccion2[0]}")
print(f"Probabilidad: {max(proba2[0]):.2f}")