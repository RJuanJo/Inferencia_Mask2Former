import streamlit as st

st.set_page_config(page_title="Descripción de la Tarea", layout="wide")

st.title("Mask2Former para Segmentación de Instancias")

st.markdown("""
### Introducción

Mask2Former es una arquitectura de segmentación universal basada en transformers que unifica tres tareas clave: segmentación **panóptica**, **de instancias** y **semántica**.  
A diferencia de modelos especializados que requieren diseños distintos para cada tarea, Mask2Former utiliza un enfoque único basado en **clasificación de máscaras**, logrando un rendimiento superior en todas las tareas con la misma arquitectura.
""")

st.markdown("### ¿Cómo resuelve la segmentación de instancias?")
st.markdown("""
- **Predicción de máscaras binarias**: En lugar de depender de *bounding boxes* (como en Mask R-CNN), genera un conjunto de máscaras binarias, cada una asociada a una categoría.

- **Queries aprendibles**: Emplea "consultas" (vectores de características) que interactúan con características de la imagen mediante un decodificador transformer mejorado.

- **Atención enmascarada**: Restringe la atención del modelo a regiones específicas predichas, enfocándose en objetos individuales y mejorando la precisión.
""")

st.markdown("### Ventajas clave:")
st.markdown("""
- **Alto rendimiento**: Supera a modelos especializados en COCO (50.1 AP en instancias) y otros benchmarks.

- **Entrenamiento eficiente**: Reduce el consumo de memoria y tiempo de entrenamiento (*3× menos memoria* que modelos anteriores).

- **Universalidad**: Un solo modelo para múltiples tareas, simplificando despliegue y mantenimiento.
""")
