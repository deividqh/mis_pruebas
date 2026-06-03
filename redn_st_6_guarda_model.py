import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Configuración de página
st.set_page_config(layout="wide", page_title="Actividad - Red Neuronal MNIST")

# ■■■■■ 1. TÍTULO DE LA ACTIVIDAD ■■■■■
st.markdown("### Ejercicio: Entrenamiento de Redes Neuronales (Clasificación MNIST)")

# ■■■■■ Carga de datos global en Caché ■■■■■
@st.cache_data
def cargar_datos():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    
    x_val, y_val = x_test[:-1000], y_test[:-1000]
    x_pred, y_pred = x_test[-1000:], y_test[-1000:]
    return x_train, y_train, x_val, y_val, x_pred, y_pred

x_train, y_train, x_val, y_val, x_pred, y_pred = cargar_datos()

# Inicializamos la memoria de Streamlit
if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = None
    st.session_state.metricas_finales = None
    st.session_state.fig_loss = None  # NUEVO: Para guardar la imagen del gráfico
    st.session_state.fig_acc = None   # NUEVO: Para guardar la imagen del gráfico

# ■■■■■ 2. ENUNCIADO EN COLLAPSE ■■■■■
ENUNCIADO = """En esta actividad vamos a entrenar una Red Neuronal Convolucional (CNN) sencilla para clasificar imágenes de dígitos escritos a mano (del 0 al 9) usando el famoso dataset MNIST.

A través del panel de control inferior, puedes modificar los **hiperparámetros** del algoritmo para ver cómo afectan al aprendizaje:
* **Épocas (Epochs):** Cuántas veces verá la red neuronal el dataset completo.
* **Tamaño del Batch:** Cuántas imágenes procesa a la vez antes de actualizar sus pesos.
* **Mezcla (Shuffle):** El tamaño de la muestra que mezcla los datos para evitar sesgos.
* **Tasa de Aprendizaje (LR):** El tamaño de los "pasos" que da el modelo al corregir sus errores.
* **Filtros Convolucionales:** Son las "gafas" de la red. Extraen bordes y formas.
* **Neuronas Ocultas (Dense):** Es el "cerebro" o capacidad de razonamiento final.
"""

with st.expander("📖 Ver el Enunciado y Exploración de los Datos", expanded=False):
    st.write(ENUNCIADO)
    st.markdown("---")
    st.markdown("#### 📂 Exploración del Dataset MNIST")
    
    col_info1, col_info2, col_img = st.columns([1.5, 1.5, 1])
    with col_info1:
        st.info(f"**Set de Entrenamiento:**\n\nImágenes: {x_train.shape}\nEtiquetas: {y_train.shape}")
        st.info(f"**Set de Validación:**\n\nImágenes: {x_val.shape}\nEtiquetas: {y_val.shape}")
    with col_info2:
        st.warning(f"**Set de Reserva (Para Predicciones):**\n\nImágenes: {x_pred.shape}\nEtiquetas: {y_pred.shape}")
    with col_img:
        st.markdown("**Visor del Set de Reserva:**")
        idx_visor = st.number_input("Selecciona un índice (0 - 999):", min_value=0, max_value=len(x_pred)-1, value=0)
        imagen_mostrar = x_pred[idx_visor].numpy() if hasattr(x_pred[idx_visor], 'numpy') else x_pred[idx_visor]
        st.image(imagen_mostrar, width=120, caption=f"Etiqueta real: {y_pred[idx_visor]}")

# ■■■■■ 3. PANEL DE CONTROLES ■■■■■
st.markdown("### 🎛️ Panel de Simulación y Control de Hiperparámetros")
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        EPOCHS = st.slider("Número de Épocas:", min_value=1, max_value=30, value=5)
        LEARNING_RATE = st.number_input("Tasa de Aprendizaje (LR):", min_value=0.00001, max_value=1.0, value=0.001, step=0.001, format="%.5f")
    with c2:
        BATCH_SIZE = st.selectbox("Tamaño del Batch:", [16, 32, 64, 128, 256], index=1)
        NEURONAS = st.slider("Neuronas Ocultas (Dense):", min_value=32, max_value=256, value=128, step=32)
    with c3:
        FILTROS_CNN = st.selectbox("Filtros Convolucionales:", [16, 32, 64], index=1)
        SHUFFLE_SIZE = st.number_input("Mezcla (Shuffle):", min_value=1000, max_value=60000, value=10000, step=1000)
    with c4:
        st.write("") 
        st.write("") 
        iniciar_entrenamiento = st.button("🚀 Iniciar Entrenamiento", type="primary", width='stretch')

# ■■■■■ 4. PROCESAMIENTO Y ENTRENAMIENTO ■■■■■
if iniciar_entrenamiento:
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

    class MyModel(Model):
        def __init__(self, filtros, neuronas):
            super().__init__()
            self.filtros = filtros     
            self.neuronas = neuronas   
            self.conv1 = Conv2D(filtros, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(neuronas, activation='relu')
            self.d2 = Dense(10)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

        # 2. Implementamos get_config para permitir la serialización al guardar
        def get_config(self):
            config = super().get_config()
            config.update({
                "filtros": self.filtros,
                "neuronas": self.neuronas,
            })
            return config

    model = MyModel(filtros=FILTROS_CNN, neuronas=NEURONAS)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        val_loss(t_loss)
        val_accuracy(labels, predictions)

    st.markdown("---")
    
    # Contenedores temporales para la animación (se borrarán al acabar)
    zona_animacion = st.empty()
    with zona_animacion.container():
        st.markdown("### ⏳ Entrenamiento Dinámico del Modelo")
        txt_estado = st.empty()
        barra_progreso = st.progress(0.0)
        col_grafico1, col_grafico2 = st.columns(2)
        grafico_perdida = col_grafico1.empty()
        grafico_precision = col_grafico2.empty()

    hist_epochs, hist_train_loss, hist_val_loss, hist_train_acc, hist_val_acc = [], [], [], [], []

    for epoch in range(EPOCHS):
        porcentaje_actual = int(((epoch + 1) / EPOCHS) * 100)
        txt_estado.info(f"**Procesando Época {epoch + 1} de {EPOCHS} ({porcentaje_actual}%)...**")
        
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        for images, labels in train_ds:
            train_step(images, labels)
        for val_images, val_labels in val_ds:
            test_step(val_images, val_labels)

        hist_epochs.append(epoch + 1)
        hist_train_loss.append(train_loss.result().numpy())
        hist_val_loss.append(val_loss.result().numpy())
        hist_train_acc.append(train_accuracy.result().numpy() * 100)
        hist_val_acc.append(val_accuracy.result().numpy() * 100)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=hist_epochs, y=hist_train_loss, mode='lines+markers', name='Train Loss', line=dict(color='#1f77b4', width=3, shape='spline')))
        fig_loss.add_trace(go.Scatter(x=hist_epochs, y=hist_val_loss, mode='lines+markers', name='Val Loss', line=dict(color='#ff7f0e', width=3, shape='spline')))
        fig_loss.update_layout(title='Evolución de la Pérdida', xaxis_title='Época', yaxis_title='Loss', template='plotly_white', margin=dict(t=40, b=0))
        grafico_perdida.plotly_chart(fig_loss, width='stretch')

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=hist_epochs, y=hist_train_acc, mode='lines+markers', name='Train Accuracy', line=dict(color='#2ca02c', width=3, shape='spline')))
        fig_acc.add_trace(go.Scatter(x=hist_epochs, y=hist_val_acc, mode='lines+markers', name='Val Accuracy', line=dict(color='#d62728', width=3, shape='spline')))
        fig_acc.update_layout(title='Evolución de la Exactitud (%)', xaxis_title='Época', yaxis_title='Precisión (%)', template='plotly_white', margin=dict(t=40, b=0))
        grafico_precision.plotly_chart(fig_acc, width='stretch')

        barra_progreso.progress((epoch + 1) / EPOCHS)

    # Limpiamos la zona de animación porque los gráficos definitivos irán abajo
    zona_animacion.empty()
    st.success(f"✅ ¡Entrenamiento completado exitosamente! (100%)")
    
    # GUARDAMOS TODO EN MEMORIA PARA QUE NO SE BORRE
    st.session_state.modelo_entrenado = model
    st.session_state.fig_loss = fig_loss
    st.session_state.fig_acc = fig_acc
    st.session_state.metricas_finales = {
        'fin_train_acc': hist_train_acc[-1],
        'fin_val_acc': hist_val_acc[-1],
        'fin_train_loss': hist_train_loss[-1],
        'fin_val_loss': hist_val_loss[-1]
    }


# ■■■■■ 5. PANEL INFERIOR (RESULTADOS PERSISTENTES) ■■■■■
# Esta parte siempre se dibujará si hay un modelo guardado, aunque toques el predictor
if st.session_state.modelo_entrenado is not None:
    
    st.markdown("---")
    st.markdown("### 📉 Gráficos Históricos del Entrenamiento")
    # Dibujamos las imágenes guardadas en memoria
    col_g1, col_g2 = st.columns(2)

    # • Streamlit es muy estricto con los identificadores. Cuando dibujas un gráfico de Plotly (plotly_chart), 
    # él intenta asignarle un ID único automáticamente. Si en la sección 5 intentas dibujar el gráfico que
    # guardaste en session_state, Streamlit cree que estás intentando "duplicar" el mismo gráfico que ya 
    # dibujaste en la sección 4 (durante el entrenamiento).
    # • Al poner key="loss_final" y key="acc_final", le estás dando un nombre único a ese elemento en la página.
    # Aunque el objeto fig_loss sea el mismo que el que se usó durante el entrenamiento, Streamlit 
    # ahora lo trata como un elemento nuevo y diferente gracias a su "llave" única, 
    # evitando el conflicto de duplicidad.
    col_g1.plotly_chart(st.session_state.fig_loss, width='stretch', key="loss_final")
    col_g2.plotly_chart(st.session_state.fig_acc, width='stretch', key="acc_final")

    st.markdown("---")
    st.markdown("### 📊 Métricas Finales de Rendimiento")

    mf = st.session_state.metricas_finales
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precisión Final (Training)", f"{mf['fin_train_acc']:.2f} %", delta="Datos conocidos")
    m2.metric("Precisión Final (Validación)", f"{mf['fin_val_acc']:.2f} %", delta="Datos invisibles")
    m3.metric("Pérdida Final (Train Loss)", f"{mf['fin_train_loss']:.4f}", delta="Menor entropía es mejor", delta_color="inverse")
    m4.metric("Pérdida Final (Val Loss)", f"{mf['fin_val_loss']:.4f}", delta="Menor entropía es mejor", delta_color="inverse")

    st.markdown("---")
    st.markdown("### 🔮 Motor de Predicción Interactivo")
    st.write("Usa el modelo ya entrenado para clasificar imágenes del **set de reserva**.")
    
    idx_pred = st.slider("Selecciona el índice de la imagen a predecir:", 0, len(x_pred)-1, 0)
    
    col_img_pred, col_resultados, col_graf_prob = st.columns([1, 1.5, 2])
    
    img_tensor = x_pred[idx_pred]
    label_real = y_pred[idx_pred]
    
    with col_img_pred:
        imagen_mostrar_pred = img_tensor.numpy() if hasattr(img_tensor, 'numpy') else img_tensor
        st.image(imagen_mostrar_pred, width=150, caption=f"Imagen Real (Dígito: {label_real})")
        
    with col_resultados:
        img_batch = tf.expand_dims(img_tensor, 0) 
        logits_pred = st.session_state.modelo_entrenado(img_batch, training=False)
        
        probabilidades = tf.nn.softmax(logits_pred).numpy()[0]
        clase_predicha = np.argmax(probabilidades)
        confianza = probabilidades[clase_predicha] * 100
        
        if clase_predicha == label_real:
            st.success(f"### ✅ Predicción: {clase_predicha}")
            st.write("¡El modelo ha acertado!")
        else:
            st.error(f"### ❌ Predicción: {clase_predicha}")
            st.write(f"El modelo ha fallado. El número real era **{label_real}**.")
            
        st.metric("Confianza del modelo", f"{confianza:.2f}%")
        
    with col_graf_prob:
        clases_x = [str(i) for i in range(10)]
        fig_probs = go.Figure(data=[go.Bar(
            x=clases_x, 
            y=probabilidades,
            marker_color=['#2ca02c' if i == clase_predicha else '#1f77b4' for i in range(10)]
        )])
        fig_probs.update_layout(
            title="Distribución de Probabilidades (Softmax)", 
            xaxis_title="Dígitos Posibles", 
            yaxis_title="Probabilidad", 
            template="plotly_white", 
            margin=dict(t=40, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_probs, width='stretch')

    # ■■■■■ 6. EXPORTAR INFORME ■■■■■
    st.markdown("---")
    informe = f"""========================================
INFORME DE ENTRENAMIENTO: RED NEURONAL MNIST
========================================
Configuración del Algoritmo:
- Épocas: {EPOCHS}
- Tamaño del Batch: {BATCH_SIZE}
- Tasa de Aprendizaje (LR): {LEARNING_RATE}
- Filtros Convolucionales: {FILTROS_CNN}
- Neuronas Ocultas: {NEURONAS}
- Tamaño de Mezcla (Shuffle): {SHUFFLE_SIZE}

Resultados Finales:
- Precisión en Entrenamiento: {mf['fin_train_acc']:.2f} %
- Precisión en Validación: {mf['fin_val_acc']:.2f} %
- Pérdida (Train Loss): {mf['fin_train_loss']:.4f}
- Pérdida (Val Loss): {mf['fin_val_loss']:.4f}
========================================"""

   

# ■■■■■ 6. EXPORTAR INFORME Y MODELO ■■■■■
    st.markdown("---")
    st.markdown("### 📥 Exportación de Resultados")
    
    col_exp1, col_exp2 = st.columns(2)
    
    # 1. Descarga del Informe (el código que ya teníamos)
    with col_exp1:
        st.download_button(
            label="📄 Descargar Informe (.txt)",
            data=informe,
            file_name="informe_entrenamiento.txt",
            mime="text/plain",
            type="primary"
        )
    
    # 2. Guardado y Descarga del Modelo (.keras)
    with col_exp2:
        # Guardamos el modelo temporalmente en el sistema de archivos del servidor
        model_path = "mi_modelo_mnist.keras"
        st.session_state.modelo_entrenado.save(model_path)
        
        # Leemos el archivo binario para ofrecérselo al usuario
        with open(model_path, "rb") as f:
            model_bytes = f.read()
            
        st.download_button(
            label="💾 Descargar Modelo (.keras)",
            data=model_bytes,
            file_name="modelo_mnist.keras",
            mime="application/x-keras",
            type="primary"
        )




# NotImplementedError: Object MyModel was created by passing
# Este error ocurre porque Keras necesita saber cómo "reconstruir" tu modelo cuando lo guardas. Como en tu __init__ le pasas parámetros (filtros y neuronas), Keras no sabe guardarlos automáticamente y te exige que implementes el método get_config().
# Para seguir con nuestra filosofía KISS, no necesitas complicarte la vida. Solo tienes que añadir este pequeño bloque dentro de tu clase MyModel:

# class MyModel(Model):
# def __init__(self, filtros, neuronas):
#     super().__init__()
#     self.filtros = filtros     # Guardamos los parámetros como atributos
#     self.neuronas = neuronas
#     self.conv1 = Conv2D(filtros, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(neuronas, activation='relu')
#     self.d2 = Dense(10)

# def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     return self.d2(x)

# # ■■■■■ LA SOLUCIÓN AL ERROR ■■■■■
# def get_config(self):
#     config = super().get_config()
#     config.update({
#         "filtros": self.filtros,
#         "neuronas": self.neuronas,
#     })
#     return config

# • ¿Por qué esto soluciona el problema?
#   Cuando llamas a .save(), Keras recorre el objeto para "serializarlo" (convertirlo en un archivo). 
# Al no encontrar la configuración, se bloquea. Con get_config, le estás dando un "mapa" de instrucciones 
# que dice: "Para volver a crear este modelo, usa estos dos valores: filtros y neuronas".

