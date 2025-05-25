import streamlit as st
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# --------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# --------------------------------------------------
@st.cache_data
def cargar_datasets():
    # Dataset de regresión: California Housing (precio de viviendas)
    housing = fetch_california_housing()
    X_boston = pd.DataFrame(housing.data, columns=housing.feature_names)
    y_boston = pd.Series(housing.target, name='PRICE')
 
    # Dataset de clasificación: Iris (especies de flores)
    iris = load_iris()
    X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_iris = pd.Series(iris.target, name='SPECIES')

    return X_boston, y_boston, X_iris, y_iris, iris

X_boston, y_boston, X_iris, y_iris, iris = cargar_datasets()

# Dividir en conjuntos de entrenamiento y prueba (80/20)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)
Xi_train, Xi_test, yi_train, yi_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# --------------------------------------------------
# 2. ENTRENAMIENTO DE MODELOS
# --------------------------------------------------
@st.cache_resource
def entrenar_modelos():
    # Modelos de regresión
    # model_lr = LinearRegression()  # Comentado por bajo desempeño
    model_rf_reg = RandomForestRegressor(random_state=42)
    model_gb = GradientBoostingRegressor(random_state=42)
    model_rf_reg.fit(Xb_train, yb_train)
    model_gb.fit(Xb_train, yb_train)

    # Modelos de clasificación
    model_log = LogisticRegression(max_iter=200)
    model_rf_clf = RandomForestClassifier(random_state=42)
    model_log.fit(Xi_train, yi_train)
    model_rf_clf.fit(Xi_train, yi_train)

    return model_rf_reg, model_gb, model_log, model_rf_clf

model_rf_reg, model_gb, model_log, model_rf_clf = entrenar_modelos()

# --------------------------------------------------
# 3. CÁLCULO DE MÉTRICAS COMPARATIVAS
# --------------------------------------------------
@st.cache_data
def calcular_metricas():
    resultados = []

    # Random Forest Regressor
    pred_rf = model_rf_reg.predict(Xb_test)
    resultados.append({
        'Modelo': 'Random Forest Regressor',
        'Tipo': 'Regresión',
        'R2 / Accuracy': r2_score(yb_test, pred_rf),
        'MSE': mean_squared_error(yb_test, pred_rf),
        'Precision': None,
        'Recall': None,
        'F1': None
    })

    # Gradient Boosting Regressor
    pred_gb = model_gb.predict(Xb_test)
    resultados.append({
        'Modelo': 'Gradient Boosting Regressor',
        'Tipo': 'Regresión',
        'R2 / Accuracy': r2_score(yb_test, pred_gb),
        'MSE': mean_squared_error(yb_test, pred_gb),
        'Precision': None,
        'Recall': None,
        'F1': None
    })

    # Regresión Logística
    pred_log = model_log.predict(Xi_test)
    resultados.append({
        'Modelo': 'Regresión Logística',
        'Tipo': 'Clasificación',
        'R2 / Accuracy': accuracy_score(yi_test, pred_log),
        'MSE': None,
        'Precision': precision_score(yi_test, pred_log, average='weighted'),
        'Recall': recall_score(yi_test, pred_log, average='weighted'),
        'F1': f1_score(yi_test, pred_log, average='weighted')
    })

    # Random Forest Classifier
    pred_rf_clf = model_rf_clf.predict(Xi_test)
    resultados.append({
        'Modelo': 'Random Forest Classifier',
        'Tipo': 'Clasificación',
        'R2 / Accuracy': accuracy_score(yi_test, pred_rf_clf),
        'MSE': None,
        'Precision': precision_score(yi_test, pred_rf_clf, average='weighted'),
        'Recall': recall_score(yi_test, pred_rf_clf, average='weighted'),
        'F1': f1_score(yi_test, pred_rf_clf, average='weighted')
    })

    return pd.DataFrame(resultados)

# --------------------------------------------------
# 4. INTERFAZ CON STREAMLIT
# --------------------------------------------------
st.title("Comparativa de Modelos de Machine Learning")

# Mostrar métricas en una tabla
st.subheader("Métricas de Evaluación")
df_metrics = calcular_metricas()
st.dataframe(df_metrics.style.format({
    'R2 / Accuracy': '{:.2f}',
    'MSE': '{:.2f}',
    'Precision': '{:.2f}',
    'Recall': '{:.2f}',
    'F1': '{:.2f}'
}))

# Sidebar: selección de modelo
st.sidebar.header("Predicción con Modelos")
modelo_seleccionado = st.sidebar.selectbox(
    "Elige un modelo:",
    ['Random Forest Regressor', 'Gradient Boosting Regressor', 'Regresión Logística', 'Random Forest Classifier']
)

# Sidebar: selección de dataset para visualización
st.sidebar.header("Visualización de Datos")
dataset_vista = st.sidebar.radio("Selecciona un dataset para visualizar:", ["California Housing", "Iris"])

if dataset_vista == "California Housing":
    st.subheader("Dataset: California Housing")
    st.dataframe(X_boston.head().style.format("{:.2f}"))
elif dataset_vista == "Iris":
    st.subheader("Dataset: Iris")
    st.dataframe(X_iris.head().style.format("{:.2f}"))

# Función para predecir según modelo y entradas

def predecir(modelo, features):
    return modelo.predict([features])[0]


# Inputs dinámicos según tipo de modelo
if modelo_seleccionado in ['Random Forest Regressor', 'Gradient Boosting Regressor']:
    st.sidebar.subheader("Características de la vivienda")
    inputs = [st.sidebar.number_input(f, value=float(X_boston[f].mean())) for f in X_boston.columns]
    if st.sidebar.button("Predecir precio"):
        modelo_obj = model_rf_reg if modelo_seleccionado == 'Random Forest Regressor' else model_gb
        precio = predecir(modelo_obj, inputs)
        #st.write(f"**Precio estimado de la vivienda:** {precio:.2f}")
        precio_real = precio * 100000
        st.write(f"**Precio estimado de la vivienda:** ${precio_real:,.0f}")


else:
    st.sidebar.subheader("Características de la flor")
    inputs = [st.sidebar.number_input(f, value=float(X_iris[f].mean())) for f in X_iris.columns]
    if st.sidebar.button("Predecir especie"):
        modelo_obj = model_log if modelo_seleccionado == 'Regresión Logística' else model_rf_clf
        especie_idx = predecir(modelo_obj, inputs)
        especie = iris.target_names[especie_idx]
        st.write(f"**Especie predicha:** {especie}")
