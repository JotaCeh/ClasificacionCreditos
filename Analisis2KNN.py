import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar datos
df = pd.read_csv('datos_credito.csv')

# Exploración básica
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas del dataset:")
print(df.head())

print("\nDistribución de la variable objetivo:")
print(df['Credito_Aprobado'].value_counts())
print(df['Credito_Aprobado'].value_counts(normalize=True).round(2))

X = df.drop('Credito_Aprobado', axis=1)
y = df['Credito_Aprobado']

columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
columnas_categoricas = X.select_dtypes(include=['object']).columns.tolist()

print("\nColumnas numéricas:", columnas_numericas)
print("Columnas categóricas:", columnas_categoricas)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), columnas_categoricas)
    ])

# Buscar el mejor valor de k
k_range = range(1, 31)
k_scores = []

#print("\nBuscando el mejor valor de k...")
for k in k_range:
    knn = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k))
    ])
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    k_scores.append(score)
    print(f"k = {k}, precisión = {score:.4f}")

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
#print("\nReporte final de clasificación:")
#print(classification_report(y_test, y_pred_final))
#print("\nMatriz de confusión:")
#print(confusion_matrix(y_test, y_pred_final))


def solicitar_datos_usuario():
    print("\n--- INGRESO DE DATOS DEL SOLICITANTE ---")
    try:
        edad = int(input("Edad (18-75): "))
        if edad < 18 or edad > 75:
            print("Error: Edad fuera de rango (18-75)")
            return None
        ingresos = float(input("Ingresos mensuales ($): "))
        antiguedad = int(input("Antigüedad laboral (años): "))
        saldo = float(input("Saldo en cuenta ($): "))
        deuda = float(input("Deuda actual ($): "))
        monto = float(input("Monto solicitado ($): "))
    except ValueError:
        print("Error: Por favor ingresa valores numéricos válidos")
        return None

    print("\nEstado civil:")
    print("1. Soltero/a")
    print("2. Casado/a")
    print("3. Divorciado/a")
    print("4. Viudo/a")
    estado_civil_opciones = {
        1: "Soltero/a",
        2: "Casado/a",
        3: "Divorciado/a",
        4: "Viudo/a"
    }
    try:
        opcion_estado = int(input("Seleccione una opción (1-4): "))
        estado_civil = estado_civil_opciones.get(opcion_estado)
        if not estado_civil:
            print("Opción no válida, usando 'Soltero/a' por defecto")
            estado_civil = "Soltero/a"
    except ValueError:
        print("Entrada no válida, usando 'Soltero/a' por defecto")
        estado_civil = "Soltero/a"

    print("\nNivel educativo:")
    print("1. Primaria")
    print("2. Secundaria")
    print("3. Bachillerato")
    print("4. Universidad")
    print("5. Postgrado")
    nivel_educativo_opciones = {
        1: "Primaria",
        2: "Secundaria",
        3: "Bachillerato",
        4: "Universidad",
        5: "Postgrado"
    }
    try:
        opcion_nivel = int(input("Seleccione una opción (1-5): "))
        nivel_educativo = nivel_educativo_opciones.get(opcion_nivel)
        if not nivel_educativo:
            print("Opción no válida, usando 'Bachillerato' por defecto")
            nivel_educativo = "Bachillerato"
    except ValueError:
        print("Entrada no válida, usando 'Bachillerato' por defecto")
        nivel_educativo = "Bachillerato"

    print("\nSector laboral:")
    print("1. Público")
    print("2. Privado")
    print("3. Independiente")
    print("4. Jubilado")
    sector_laboral_opciones = {
        1: "Público",
        2: "Privado",
        3: "Independiente",
        4: "Jubilado"
    }
    try:
        opcion_sector = int(input("Seleccione una opción (1-4): "))
        sector_laboral = sector_laboral_opciones.get(opcion_sector)
        if not sector_laboral:
            print("Opción no válida, usando 'Privado' por defecto")
            sector_laboral = "Privado"
    except ValueError:
        print("Entrada no válida, usando 'Privado' por defecto")
        sector_laboral = "Privado"

    tiene_propiedad = input("\n¿Tiene propiedad? (Sí/No): ").capitalize()
    if tiene_propiedad not in ["Sí", "No", "Si"]:
        print("Entrada no válida, usando 'No' por defecto")
        tiene_propiedad = "No"
    if tiene_propiedad == "Si":
        tiene_propiedad = "Sí"

    tiene_vehiculo = input("¿Tiene vehículo? (Sí/No): ").capitalize()
    if tiene_vehiculo not in ["Sí", "No", "Si"]:
        print("Entrada no válida, usando 'No' por defecto")
        tiene_vehiculo = "No"
    if tiene_vehiculo == "Si":
        tiene_vehiculo = "Sí"

    print("\nHistorial crediticio:")
    print("1. Excelente")
    print("2. Bueno")
    print("3. Regular")
    print("4. Malo")
    print("5. Limitado")
    print("6. Sin historial")
    historial_opciones = {
        1: "Excelente",
        2: "Bueno",
        3: "Regular",
        4: "Malo",
        5: "Limitado",
        6: "Sin historial"
    }
    try:
        opcion_historial = int(input("Seleccione una opción (1-6): "))
        historial_crediticio = historial_opciones.get(opcion_historial)
        if not historial_crediticio:
            print("Opción no válida, usando 'Limitado' por defecto")
            historial_crediticio = "Limitado"
    except ValueError:
        print("Entrada no válida, usando 'Limitado' por defecto")
        historial_crediticio = "Limitado"

    solicitante = pd.DataFrame({
        'Edad': [edad],
        'Ingresos_Mensuales': [ingresos],
        'Antiguedad_Laboral': [antiguedad],
        'Estado_Civil': [estado_civil],
        'Nivel_Educativo': [nivel_educativo],
        'Sector_Laboral': [sector_laboral],
        'Tiene_Propiedad': [tiene_propiedad],
        'Tiene_Vehiculo': [tiene_vehiculo],
        'Saldo_Cuenta': [saldo],
        'Deuda_Actual': [deuda],
        'Monto_Solicitado': [monto],
        'Historial_Crediticio': [historial_crediticio]
    })
    return solicitante

print("\n--- PREDICCIÓN DE APROBACIÓN DE CRÉDITO ---")
print("Ingrese los datos del solicitante:")

solicitante = solicitar_datos_usuario()
if solicitante is None:
    print("No se pudieron obtener los datos correctamente.")
    exit()

print("\nDatos del solicitante:")
print(solicitante)

prediccion = mejor_knn.predict(solicitante)
proba = mejor_knn.predict_proba(solicitante)



print(f"\nPredicción: {prediccion[0]}")
print(f"Probabilidad: {max(proba[0]):.2f}")
