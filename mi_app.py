import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.impute import SimpleImputer

# Título de la app
st.title("Evaluación y Pruebas de Modelos de Machine Learning - Salud Falabella")

# Cargar datos
# Asegúrate de que la ruta "falabella_salud_limpio.csv" sea correcta
try:
    df_original = pd.read_csv("falabella_salud_limpio.csv")
    df = df_original.copy() # Trabajar con una copia para preprocesamiento
except FileNotFoundError:
    st.error("Error: El archivo 'falabella_salud_limpio.csv' no se encontró. Asegúrate de que esté en el mismo directorio que el script o proporciona la ruta correcta.")
    st.stop()


# --- Preprocesamiento ---
# Columnas a eliminar
cols_to_drop = ['Imagen', 'Vista', 'Nombre_Producto']
df = df.drop(columns=cols_to_drop, errors='ignore')

# Asumimos que 'Descuento' es numérico (ej: 24 para 24%) y no un string como '-24%'
# Si 'Descuento' es un string como "X%", necesitas limpiarlo a numérico primero.
# Ejemplo de limpieza para 'Descuento' si está como '-24%':
# df['Descuento'] = df['Descuento'].str.replace('%', '').astype(float).abs()
# O si es positivo "24%":
# df['Descuento'] = df['Descuento'].str.replace('%', '').astype(float)

# Codificación de etiquetas para columnas categóricas
label_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    # Tratar NaN como una categoría separada o imputar antes si es necesario
    df[col] = df[col].astype(str) # Asegurar que todo sea string antes de LabelEncoder
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Guardar el orden de las columnas después de la codificación y ANTES de la imputación
# Estas son las columnas que el imputer espera, en este orden.
imputer_feature_names_ordered = df.columns.tolist()

# Imputación de valores faltantes (ajustado a todas las columnas en este punto)
imputer = SimpleImputer(strategy='mean')
df_imputed_values = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed_values, columns=imputer_feature_names_ordered, index=df.index)

# Características para el escalador (excluyendo Precio_Actual y Descuento)
X_df_before_scaling = df.drop(columns=['Precio_Actual', 'Descuento'])
# Guardar el orden de las columnas que el scaler espera.
scaler_feature_names_ordered = X_df_before_scaling.columns.tolist()

# Escalado de características
scaler = StandardScaler()
X_scaled_values = scaler.fit_transform(X_df_before_scaling)
X_scaled = pd.DataFrame(X_scaled_values, columns=scaler_feature_names_ordered)

# Definir variables objetivo
# Asegúrate que las columnas objetivo existan y sean del tipo correcto en 'df'
# (df ya está imputado, lo que podría afectar a las 'y' si tenían NaNs y no se manejaron antes)
y_reg1 = df['Precio_Actual']  # Predicción de Precio_Actual
y_reg2 = df['Precio_Ref']     # Predicción de Precio_Ref

# Ajustar la lógica de y_class1 si 'Descuento' no es un porcentaje numérico directo
# Asumiendo que 'Descuento' en df es ahora un número (ej. 24 para 24%)
y_class1 = (df['Descuento'] >= 20).astype(int) # Ejemplo: Descuento es 20% o más

# Ajustar los bins si es necesario según el rango de tus precios
PRICE_BINS = [0, 10000, 30000, 100000, np.inf]
PRICE_LABELS = [0, 1, 2, 3]
y_class2 = pd.cut(df['Precio_Actual'], bins=PRICE_BINS, labels=PRICE_LABELS, right=False)

# --- División de datos ---
# Usar X_scaled (DataFrame) para mantener nombres de columnas si es necesario, aunque train_test_split devuelve arrays
X_train, X_test, y_reg1_train, y_reg1_test = train_test_split(X_scaled, y_reg1, test_size=0.2, random_state=42)
# Para y_reg2, y_class1, y_class2, X_train y X_test son los mismos
_, _, y_reg2_train, y_reg2_test = train_test_split(X_scaled, y_reg2, test_size=0.2, random_state=42)
_, _, y_class1_train, y_class1_test = train_test_split(X_scaled, y_class1, test_size=0.2, random_state=42)
_, _, y_class2_train, y_class2_test = train_test_split(X_scaled, y_class2, test_size=0.2, random_state=42)


# --- Entrenamiento de modelos ---
models = {
    "Linear Regression": LinearRegression().fit(X_train, y_reg1_train),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_reg2_train),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear').fit(X_train, y_class1_train), # Añadido solver para evitar warnings
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_class2_train)
}

# --- Evaluar modelos ---
metrics = []

# Regresión
y_pred_lr = models["Linear Regression"].predict(X_test)
y_pred_rf_reg = models["Random Forest Regressor"].predict(X_test) # Renombrado para evitar conflicto

metrics.append({
    "Modelo": "Linear Regression",
    "Objetivo": "Precio_Actual",
    "Tipo": "Regresión",
    "R2 Score": round(r2_score(y_reg1_test, y_pred_lr), 5),
    "MSE": round(mean_squared_error(y_reg1_test, y_pred_lr), 5),
    "Precisión": "-"
})

metrics.append({
    "Modelo": "Random Forest Regressor",
    "Objetivo": "Precio_Ref",
    "Tipo": "Regresión",
    "R2 Score": round(r2_score(y_reg2_test, y_pred_rf_reg), 5),
    "MSE": round(mean_squared_error(y_reg2_test, y_pred_rf_reg), 5),
    "Precisión": "-"
})

# Clasificación
y_pred_log = models["Logistic Regression"].predict(X_test)
y_pred_rfc_class = models["Random Forest Classifier"].predict(X_test) # Renombrado

metrics.append({
    "Modelo": "Logistic Regression",
    "Objetivo": "Descuento >= 20%",
    "Tipo": "Clasificación Binaria",
    "R2 Score": "-",
    "MSE": "-",
    "Precisión": round(accuracy_score(y_class1_test, y_pred_log), 5)
})

metrics.append({
    "Modelo": "Random Forest Classifier",
    "Objetivo": "Categoría Precio_Actual",
    "Tipo": "Clasificación Multiclase",
    "R2 Score": "-",
    "MSE": "-",
    "Precisión": round(accuracy_score(y_class2_test, y_pred_rfc_class), 5)
})

# Mostrar tabla
metrics_df = pd.DataFrame(metrics)
st.subheader("📊 Resultados de los Modelos Entrenados")
st.dataframe(metrics_df, use_container_width=True)

# Descargar resultados (opcional)
#csv = metrics_df.to_csv(index=False).encode('utf-8')
#st.download_button(
 #   label="⬇️ Descargar Resultados en CSV",
  #  data=csv,
   # file_name="metricas_modelos.csv",
    #mime="text/csv"
#)

# --- Formulario de Predicción ---
st.subheader("🧪 Probar Modelos con Nuevos Datos")

# Columnas que el usuario debe ingresar (las que forman X_scaled)
# Estas son las columnas en 'scaler_feature_names_ordered'
# ['Título', 'Título_URL', 'Vendedor', 'Precio_Anterior', 'Precio_Ref', 'Entrega', 'Retiro']

# Obtener las clases originales para los selectbox de las columnas codificadas
# Esto asume que df_original tiene las columnas antes de cualquier preprocesamiento.
# Y que label_cols son las columnas que efectivamente fueron codificadas.
form_cat_cols = [col for col in scaler_feature_names_ordered if col in label_encoders]
form_num_cols = [col for col in scaler_feature_names_ordered if col not in label_encoders]

with st.form(key='prediction_form'):
    st.write("Por favor, ingresa los datos del producto:")
    
    form_inputs = {}

    # Crear campos de entrada dinámicamente
    # Para columnas categóricas (que fueron codificadas)
    for col_name in form_cat_cols:
        original_col_name = col_name # Asumimos que el nombre es el mismo
        if original_col_name in df_original.columns: # Verificar si la columna existe en el df original
            # Obtener opciones únicas del df original para el selectbox
            options = [''] + sorted(list(df_original[original_col_name].astype(str).unique()))
            form_inputs[col_name] = st.selectbox(f"Selecciona {col_name}:", options=options, key=f"form_{col_name}")
        else:
            # Si la columna no está en df_original (raro, pero por seguridad)
            form_inputs[col_name] = st.text_input(f"Ingresa {col_name} (texto):", key=f"form_{col_name}")


    # Para columnas numéricas
    for col_name in form_num_cols:
        # 'Precio_Anterior', 'Precio_Ref'
        form_inputs[col_name] = st.number_input(f"Ingresa {col_name}:", value=0.0, format="%.2f", key=f"form_{col_name}")

    submit_button = st.form_submit_button(label='Obtener Predicciones')

if submit_button:
    # 1. Crear DataFrame con los datos del formulario
    # Las columnas deben estar en el orden de 'scaler_feature_names_ordered'
    # Pero primero necesitamos aplicarles el preprocesamiento
    
    input_data_raw = {col: [form_inputs[col]] for col in scaler_feature_names_ordered}
    input_df_raw = pd.DataFrame.from_dict(input_data_raw)
    
    # 2. Aplicar preprocesamiento (codificación, imputación, escalado)
    #    Consistente con el preprocesamiento de entrenamiento.
    
    # Copia para procesar
    processed_input_df = input_df_raw.copy()
    
    # 2a. Codificación de etiquetas para las columnas categóricas
    for col_name in form_cat_cols:
        if form_inputs[col_name] == '': # Si el usuario no seleccionó nada (dejó la opción vacía)
            # Manejar como NaN o un valor que el imputer luego pueda manejar
            # O si LabelEncoder no puede manejar NaN, se puede asignar un placeholder si fue entrenado con él
            processed_input_df[col_name] = np.nan # Será imputado
        else:
            try:
                # Usar el label encoder correspondiente
                processed_input_df[col_name] = label_encoders[col_name].transform(processed_input_df[col_name].astype(str))
            except ValueError:
                # Si la categoría es nueva y no fue vista durante el fit del LabelEncoder
                st.warning(f"Categoría '{form_inputs[col_name]}' para '{col_name}' no reconocida. Se imputará.")
                processed_input_df[col_name] = np.nan # Marcar para imputación

    # Asegurar que todas las columnas sean numéricas antes de la imputación global
    for col in processed_input_df.columns:
        processed_input_df[col] = pd.to_numeric(processed_input_df[col], errors='coerce')


    # 2b. Preparación para el Imputer:
    # El imputer se ajustó en 'imputer_feature_names_ordered'
    # ('Título', 'Título_URL', 'Vendedor', 'Precio_Actual', 'Descuento', 'Precio_Anterior', 'Precio_Ref', 'Entrega', 'Retiro')
    # Necesitamos construir un DataFrame con estas columnas para el imputer.
    # Las columnas 'Precio_Actual' y 'Descuento' no vienen del formulario, así que las añadimos como NaN.
    
    df_for_imputer_input = pd.DataFrame(columns=imputer_feature_names_ordered, index=[0])
    
    for col in imputer_feature_names_ordered:
        if col in processed_input_df.columns:
            df_for_imputer_input[col] = processed_input_df[col].values
        else:
            # Columnas como 'Precio_Actual', 'Descuento' que no están en processed_input_df (features X)
            df_for_imputer_input[col] = np.nan
            
    # 2c. Imputación
    try:
        imputed_values_input = imputer.transform(df_for_imputer_input)
        imputed_df_input = pd.DataFrame(imputed_values_input, columns=imputer_feature_names_ordered)
    except Exception as e:
        st.error(f"Error durante la imputación de la entrada: {e}")
        st.stop()

    # 2d. Selección de Características para el Scaler
    # El scaler se ajustó en 'scaler_feature_names_ordered'
    features_for_scaler_input_df = imputed_df_input[scaler_feature_names_ordered]
    
    # 2e. Escalado
    try:
        scaled_input_values = scaler.transform(features_for_scaler_input_df)
        final_input_for_models = pd.DataFrame(scaled_input_values, columns=scaler_feature_names_ordered)
    except Exception as e:
        st.error(f"Error durante el escalado de la entrada: {e}")
        st.stop()

    # 3. Realizar predicciones
    st.markdown("---")
    st.subheader("🔍 Resultados de la Predicción:")

    try:
        pred_price_actual = models["Linear Regression"].predict(final_input_for_models)
        st.write(f"**Predicción Precio Actual (Regresión Lineal):** `${pred_price_actual[0]:,.2f}`")

        pred_price_ref = models["Random Forest Regressor"].predict(final_input_for_models)
        st.write(f"**Predicción Precio Referencia (Random Forest Regressor):** `${pred_price_ref[0]:,.2f}`")
      #  st.caption("_Nota: La predicción de Precio Referencia puede ser similar al valor ingresado debido a cómo fue entrenado el modelo._")

        pred_discount_class = models["Logistic Regression"].predict(final_input_for_models)
        pred_discount_proba = models["Logistic Regression"].predict_proba(final_input_for_models)
        discount_label = "Sí (>= 20%)" if pred_discount_class[0] == 1 else "No (< 20%)"
        st.write(f"**Predicción de Descuento Mayor o Igual al 20% (Regresión Logística):** {discount_label} (Probabilidad SÍ: {pred_discount_proba[0][1]:.2%})")
        
        pred_price_category = models["Random Forest Classifier"].predict(final_input_for_models)
        # Mapear la etiqueta numérica de la categoría de precio a una descripción
        category_map = {
            0: f"{PRICE_BINS[0]:,} - {PRICE_BINS[1]:,}",
            1: f"{PRICE_BINS[1]:,} - {PRICE_BINS[2]:,}",
            2: f"{PRICE_BINS[2]:,} - {PRICE_BINS[3]:,}",
            3: f"> {PRICE_BINS[3]:,}"
        }
        price_category_label = category_map.get(pred_price_category[0], "Categoría Desconocida")
        st.write(f"**Predicción Categoría de Precio Actual (Random Forest Classifier):** {price_category_label}")

    except Exception as e:
        st.error(f"Error al realizar las predicciones: {e}")
        st.dataframe(final_input_for_models) # Muestra los datos que se enviaron al modelo