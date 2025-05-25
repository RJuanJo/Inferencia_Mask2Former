import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Descripción de la Tarea", layout="wide")

st.title("Mask2Former para Segmentación de Instancias")

st.markdown("""
### Introducción

Mask2Former es una arquitectura de segmentación universal basada en transformers que unifica tres tareas clave: segmentación **panóptica**, **de instancias** y **semántica**.  
A diferencia de modelos especializados que requieren diseños distintos para cada tarea, Mask2Former utiliza un enfoque único basado en **clasificación de máscaras**, logrando un rendimiento superior en todas las tareas con la misma arquitectura.
""")

# Mostrar imagen de arquitectura
st.image(os.path.join("sources", "arquitectura.jpg"), caption="Arquitectura general de Mask2Former", use_column_width=True)

st.markdown("### ¿Cómo resuelve la segmentación de instancias?")
st.markdown("""
La segmentación de instancias busca **detectar y diferenciar cada objeto individual** en una imagen, asignándole una máscara distinta a cada uno (incluso si son de la misma clase).

Mask2Former logra esto de forma precisa y sin cajas delimitadoras, mediante un proceso en tres partes:

- **Predicción de máscaras binarias**: A diferencia de Mask R-CNN que genera *bounding boxes*, Mask2Former predice directamente **máscaras binarias** para cada instancia, identificando la forma completa del objeto.

- **Queries aprendibles**: Usa *queries* entrenables que representan propuestas de objetos. Estas queries interactúan con las características extraídas de la imagen, permitiendo que el modelo aprenda a detectar regiones relevantes sin necesidad de cajas previas.

- **Atención enmascarada (Masked Attention)**: En lugar de aplicar atención a toda la imagen, la atención se **restringe a regiones específicas** determinadas por las máscaras predichas. Esto mejora la precisión, especialmente en objetos pequeños o solapados, al concentrar los recursos del modelo en los objetos de interés.

Este enfoque permite que el modelo reconozca **cuántas instancias hay**, **a qué clase pertenece cada una**, y **dónde están ubicadas con precisión pixel a pixel**.
""")

st.markdown("### Ventajas clave:")
st.markdown("""
- **Alto rendimiento**: Supera a modelos especializados en COCO (50.1 AP en instancias) y otros benchmarks.

- **Entrenamiento eficiente**: Reduce el consumo de memoria y tiempo de entrenamiento (*3× menos memoria* que modelos anteriores).

- **Universalidad**: Un solo modelo para múltiples tareas, simplificando despliegue y mantenimiento.
""")
