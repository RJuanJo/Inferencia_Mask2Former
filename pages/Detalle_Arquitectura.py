import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Detalle de la Arquitectura", layout="wide")

st.title("Funcionamiento de Mask2Former para Segmentación Semántica")  # Cambiado

st.markdown("### Introducción General")
st.markdown("""
Mask2Former es una arquitectura basada en transformers diseñada para segmentación **semántica**, **panóptica** y **de instancias**.  
Su mecanismo de atención enmascarada permite enfocarse en regiones relevantes de cada categoría, usando consultas aprendibles (queries) para predecir máscaras por clase (no por objeto individual).

En esta sección exploramos cómo se adapta para segmentación semántica.
""")

# COMPONENTES CLAVE
st.markdown("### 1. Componentes Clave")

with st.expander("Atención Enmascarada (Masked Attention)"):
    st.markdown("""
    - Reemplaza la atención global de los transformers tradicionales.  
    - **Enfoca la atención solo en píxeles de interés para cada clase** (ej: si una query representa "árboles", ignora edificios o calles).  
    - Ejemplo: Query "carretera" → atención limitada a píxeles de vialidad.
    """)

with st.expander("Características Multi-Escala"):
    st.markdown("""
    - Combina características de **alta resolución** (1/8) y **baja resolución** (1/32).  
    - Ventaja: Detecta categorías pequeñas (ej: señales de tráfico) con precisión.
    """)

with st.expander("Optimizaciones"):
    st.markdown("""
    - **Queries aprendibles**: Cada query se especializa en una clase (ej: Query 1 = "cielo", Query 2 = "vegetación").  
    - **Sin dropout**: Mejora rendimiento en tareas semánticas donde la coherencia espacial es crítica.
    """)

# ENTRADAS Y SALIDAS
st.markdown("### 2. Entradas y Salidas del Modelo")
st.markdown("""
**Entradas**  
- Imagen: Tensor `[H, W, 3]`.  
- Queries: `[N, C]`, vectores aprendibles asociados a categorías.

**Salidas**  
- Mapa de clases: `[H/4, W/4]` (cada píxel tiene un ID de clase).  
- Scores por categoría: `[N, K]` (confianza por clase).
""")
st.image("sources/entradas_salidas.jpg", caption="Entradas: imagen + queries; Salidas: mapa de clases y scores", use_column_width=True)  # Leyenda ajustada

# Q, K, V
st.markdown("### 3. Generación de Q, K y V")
st.markdown("""
**Queries (Q)**  
- Aprendibles, cada una representa una categoría semántica (ej: "edificios", "peatones").

**Keys y Values (K, V)**  
- Derivados del Pixel Decoder con características multi-escala.  
```python
K = Linear(features + pos_emb + scale_emb)
V = Linear(features + pos_emb + scale_emb)
```
""")
st.image("sources/vectores_qkv.jpg", caption="Generación de Q (clases), K, V (características)", use_column_width=True)

# CREACIÓN DE MÁSCARAS
st.markdown("### 4. Creación Iterativa del Mapa Semántico")
st.markdown("""
```python
for l in range(L):  
    Q = Q_prev + masked_attention(Q_prev, K, V)  # Atención por clase
    M_l = Linear(Q) + upsample(M_l-1)  # Refina máscaras de categorías
```
""")
st.image("sources/creacion_mascaras.jpg", caption="Refinamiento progresivo del mapa de clases", use_column_width=True)

# ATENCIÓN ENMASCARADA
st.markdown("### 5. Atención Enmascarada por Clase")
st.markdown("""
Cada query restringe la atención a píxeles de su categoría:
```math
{Attention}(Q, K, V) = \\text{softmax}(\\bm{\\mathcal{M}}_{l-1} + QK^T/\\sqrt{d})V
```
* 𝓜(x,y) = 0 si el píxel pertenece a la clase actual.  
* 𝓜(x,y) = -∞ si no es relevante para la query.
""")
st.image("sources/atencion_enmascarada.jpg", caption="Atención enfocada en píxeles de la clase objetivo", use_column_width=True)

# DIFERENCIAS
st.markdown("### 6. Diferencias con Modelos Anteriores")
st.image("sources/tabla_diferencias.jpg", use_column_width=True)
st.markdown("""
**Nota**:  
- Las queries agrupan píxeles por categoría.  
- Las máscaras son mapas por clase (ej: todos los "árboles" en una sola región).
""")

# POR QUÉ IDEAL
st.markdown("### 7. ¿Por qué es ideal para segmentación semántica?")
st.markdown("""
* **Precisión en bordes**: Define límites entre clases (ej: acera vs. calle).  
* **Consistencia espacial**: Mantiene coherencia en áreas grandes (ej: cielo).  
* **Eficiencia**: Menos queries necesarias vs. segmentación de instancias.
""")

# PROCESO PASO A PASO
st.markdown("### 8. Proceso Paso a Paso")
st.markdown("""
#### **Paso 1: Inicialización**  
- **Queries**: Cada una representa una clase (ej: Query 1 = "vehículos", Query 2 = "peatones").  

#### **Paso 2: Atención Enmascarada**  
- Cada query analiza solo píxeles de su categoría (ignora fondos irrelevantes).  

#### **Paso 3: Salidas**  
1. **Mapa de clases**: `[H/4, W/4]` (asignación por píxel).  
2. **Confianza por clase**: Probabilidad global de cada categoría.  

#### **Paso 4: Refinamiento (9 Iteraciones)**  
| **Capas** | **Resolución** | **Enfoque**                                  |  
|-----------|----------------|---------------------------------------------|  
| 1-3       | 1/32           | Contexto global (ej: "área urbana").        |  
| 4-6       | 1/16           | Estructuras (ej: "formas de edificios").    |  
| 7-9       | 1/8            | Detalles (ej: "ventanas", "señalización").  |  
""")

# RESULTADOS
st.markdown("### 9. Resultados en Semántica")
st.markdown("""
**Benchmarks clave**:  
- **ADE20K**: 57.7 mIoU (state-of-the-art)  
- **Cityscapes**: 84.3 mIoU  
- **COCO-Stuff**: 45.2 mIoU  

**Ventajas**:  
- Preserva bordes nítidos entre categorías.  
- Eficiente para escenas con muchas clases.

**Referencias Oficiales**

- [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) – Paper oficial.  
- [Repositorio GitHub oficial](https://github.com/facebookresearch/Mask2Former.git)  
- [Modelo en Hugging Face](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)

**Créditos**  
Todas las explicaciones, diagramas e ilustraciones fueron construidas con base en el contenido del paper oficial y sus recursos asociados.  
**Todo el crédito por la arquitectura y avances técnicos es de los autores originales. Este proyecto es una interfaz educativa/demostrativa sin ninguna autoría sobre el modelo ni sus fundamentos.**
""")