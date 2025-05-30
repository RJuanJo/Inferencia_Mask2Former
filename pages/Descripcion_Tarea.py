import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Descripción de la Tarea", layout="wide")

st.title("Mask2Former para Segmentación Semántica")

st.markdown("""
### Introducción

Mask2Former es una arquitectura de segmentación universal basada en transformers que unifica tres tareas clave: segmentación **panóptica**, **de instancias** y **semántica**.  
A diferencia de modelos especializados que requieren diseños distintos para cada tarea, Mask2Former utiliza un enfoque único basado en **clasificación de máscaras**, logrando un rendimiento superior en todas las tareas con la misma arquitectura.
""")

# Mostrar imagen de arquitectura
st.image(os.path.join("sources", "arquitectura.jpg"), caption="Arquitectura general de Mask2Former", use_column_width=True)

st.markdown("### ¿Cómo resuelve la segmentación semántica?")
st.markdown("""
La segmentación semántica busca **clasificar cada píxel** de una imagen según su categoría (ej: "cielo", "carretera", "árbol"), **sin distinguir entre instancias individuales** (todos los píxeles de "árbol" pertenecen a la misma categoría, sin importar cuántos árboles haya).

Mask2Former aborda esta tarea mediante:
""")

st.markdown("""
#### 1. Predicción por categorías
- Genera **una máscara por clase** (en lugar de una por objeto).
- Ejemplo: 
  - Todos los píxeles de "árbol" se agrupan en una sola máscara, aunque haya 10 árboles en la imagen.
  - Difiere de la segmentación de instancias, donde cada árbol tendría su propia máscara.

#### 2. Queries aprendibles por clase
- Cada *query* se especializa en predecir una categoría específica (ej: Query 1 = "cielo", Query 2 = "carretera").
- Estas queries son entrenadas para activarse en regiones donde aparece su categoría asignada.

#### 3. Atención enmascarada global
- Aunque usa el mismo mecanismo de atención enmascarada que para instancias, aquí se enfoca en **todas las regiones de una categoría**.
- Ejemplo: Para la clase "árbol", la atención cubrirá todos los píxeles de árboles en la imagen.
""")

st.markdown("### Ventajas para segmentación semántica")
st.markdown("""
- **Alto rendimiento**: Logra **57.7 mIoU** en ADE20K.
- **Eficiencia**: Usa el mismo modelo para todas las tareas, reduciendo complejidad.
- **Calidad en bordes**: La atención enmascarada preserva detalles finos entre categorías (ej: bordes entre "calle" y "acera").
""")
