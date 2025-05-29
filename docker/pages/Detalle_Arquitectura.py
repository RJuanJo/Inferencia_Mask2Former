import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Detalle de la Arquitectura", layout="wide")

st.title("Funcionamiento de Mask2Former para Segmentación de Instancias")

st.markdown("### Introducción General")
st.markdown("""
Mask2Former es una arquitectura basada en transformers diseñada específicamente para segmentación **panóptica**, **de instancias** y **semántica**.  
Se basa en un mecanismo de atención enmascarada que restringe el foco del modelo a regiones relevantes, usando consultas aprendibles (queries) para predecir directamente máscaras binarias asociadas a cada clase.

En esta sección profundizamos en su arquitectura, entradas y salidas, y el funcionamiento interno del mecanismo de atención.
""")

# COMPONENTES CLAVE
st.markdown("### 1. Componentes Clave")

with st.expander("Atención Enmascarada (Masked Attention)"):
    st.markdown("""
    - Reemplaza la atención global de los transformers tradicionales.  
    - **Enfoca la atención solo en regiones de interés** (máscaras predichas), evitando distracciones en fondos irrelevantes.  
    - Ejemplo: Si una *query* predice un "perro", la atención se limita a la región de ese perro.
    """)

with st.expander("Características Multi-Escala"):
    st.markdown("""
    - Combina características de **alta resolución** (1/8 de la imagen original) y **baja resolución** (1/32).  
    - Ventaja: Detecta objetos pequeños con precisión gracias a detalles finos de alta resolución.
    """)

with st.expander("Optimizaciones"):
    st.markdown("""
    - **Orden de capas**: Cambia el orden de las capas de auto-atención y atención cruzada.  
    - **Queries aprendibles**: Inicializa las queries como parámetros entrenables (no ceros).  
    - **Sin dropout**: Mejora el rendimiento al remover regularización innecesaria.
    """)

with st.expander("Pérdida Eficiente"):
    st.markdown("""
    - Calcula la pérdida en puntos muestreados aleatoriamente (ej: 112×112 puntos), reduciendo consumo de memoria.
    """)

# ENTRADAS Y SALIDAS
st.markdown("### 2. Entradas y Salidas del Modelo")

st.markdown("""
**Entradas**  
- Imagen: Tensor `[H, W, 3]`.  
- Queries: `[N, C]`, vectores aprendibles inicializados aleatoriamente que actúan como propuestas de instancia.

**Salidas**  
- Máscaras binarias: `[N, H/4, W/4]`.  
- Scores de clases: `[N, K]`.
""")
st.image("sources/entradas_salidas.jpg", caption="Entradas: imagen + queries; Salidas: máscaras y scores", use_container_width=True)

# Q, K, V
st.markdown("### 3. Generación de Q, K y V")

st.markdown("""
**Queries (Q)**  
- Aprendibles, refinadas por el decodificador transformer en múltiples capas.

**Keys y Values (K, V)**  
- Derivados del Pixel Decoder que genera una pirámide de características multi-escala (resoluciones 1/32, 1/16, 1/8).  

```python
K = Linear(features + pos_emb + scale_emb)
V = Linear(features + pos_emb + scale_emb)
````

""")
st.image("sources/vectores_qkv.jpg", caption="Generación de vectores Q (consultas), K (claves), V (valores)", use_container_width=True)

# CREACIÓN DE MÁSCARAS

st.markdown("### 4. Creación Iterativa de Máscaras")

st.markdown("""

```python
for l in range(L):  
    Q = Q_prev + masked_attention(Q_prev, K, V)  
    Q = Q + self_attention(Q)  
    Q = Q + FFN(Q)  
    M_l = Linear(Q) + upsample(M_l-1)
```

Cada capa refina las máscaras predichas. Se construyen de forma progresiva con atención enmascarada, auto-atención y FFN.
""")
st.image("sources/creacion_mascaras.jpg", caption="Proceso de refinamiento iterativo de máscaras", use_container_width=True)

# ATENCIÓN ENMASCARADA

st.markdown("### 5. Mecanismo de Atención Enmascarada")

st.markdown("""
Cada capa del decodificador toma como entrada una máscara binaria de la capa anterior (`Mₗ₋₁`) y enfoca la atención solo dentro de esa región:

```math
{Attention}(Q, K, V) = \\text{softmax}(\\bm{\\mathcal{M}}_{l-1} + QK^T/\\sqrt{d})V
```

* 𝓜(x,y) = 0 si Mₗ₋₁(x,y) = 1 (dentro de la ROI).
* 𝓜(x,y) = -∞ si Mₗ₋₁(x,y) = 0 (excluye fondos).
  """)
st.image("sources/atencion_enmascarada.jpg", caption="Atención concentrada solo en regiones relevantes", use_container_width=True)

# DIFERENCIAS

st.markdown("### 6. Diferencias con Modelos Anteriores")
st.image("sources/tabla_diferencias.jpg", use_container_width=True)

# POR QUÉ IDEAL

st.markdown("### 7. ¿Por qué es ideal para segmentación de instancias?")
st.markdown("""

* **Precisión en bordes**: mejora la calidad del contorno.
* **Separación de objetos solapados**: atención localizada por instancia.
* **Reconocimiento de objetos pequeños**: gracias a las características de alta resolución.
  """)

# PROCESO PASO A PASO
st.markdown("### 8. Proceso Paso a Paso de Segmentación")
st.markdown("""
A continuación se detalla cómo Mask2Former procesa una imagen para generar máscaras de instancias:
""")

st.markdown("""
#### **Paso 1: Inicialización**  
- **Entrada**: El modelo recibe una imagen (ej: foto con perros y árboles).  
- **100 queries**: Vectores aprendibles que actúan como "notas adhesivas vacías" para registrar información de objetos.  
  - Cada query se especializa en un objeto distinto (Query 1 → Perro 1, Query 2 → Perro 2, etc.).  
- **Detalle técnico**: Estas queries son parámetros entrenables que el modelo ajusta durante el aprendizaje.  
""")

st.markdown("""
#### **Paso 2: Atención Enmascarada**  
- Cada query analiza solo la región de la imagen donde predijo un objeto en el paso anterior.  
  - **Ejemplo**: Si una query identificó un perro, ignora árboles, cielo y otros elementos.  
- **Innovación clave**:  
  - Modelos antiguos (como DETR) analizan toda la imagen para cada query.  
  - Mask2Former usa máscaras binarias para restringir el área de atención.  
""")

st.markdown("""
#### **Paso 3: Generación de Máscaras y Clases**  
Cada query produce:  
1. **Máscara binaria**:  
   - Mapa de píxeles donde `1` = objeto y `0` = fondo.  
   - Resolución: 1/4 del tamaño original (balance precisión-eficiencia).  
2. **Clase y confianza**:  
   - Predicción categórica (ej: "perro" con 95% de confianza).  
   - Si no detecta un objeto, devuelve "no objeto".  
""")

st.markdown("""
#### **Paso 4: Refinamiento (9 Iteraciones)**  
El proceso se repite 9 veces (3 grupos de 3 capas) para mejorar las máscaras:  

| **Capas** | **Resolución** | **Enfoque**                                  |  
|-----------|----------------|---------------------------------------------|  
| 1-3       | 1/32 (baja)    | Contexto general (ej: "hay un perro").      |  
| 4-6       | 1/16 (media)   | Formas básicas (ej: "4 patas y cola").      |  
| 7-9       | 1/8 (alta)     | Detalles finos (ej: "orejas puntiagudas").  |  

**¿Por qué 9 iteraciones?**  
- **Jerarquía de características**: Combina contexto global (capas iniciales) con detalles precisos (capas finales).  
- **Eficiencia**: 9 capas demostraron ser el equilibrio óptimo entre precisión y coste computacional.  
""")

# RESULTADOS Y REFERENCIAS

st.markdown("### 9. Resultados y Referencias")
st.markdown("""
**Resultados Clave**  
- **COCO Instance Segmentation**: 50.1 AP (supera HTC++).  
- **Eficiencia**: Solo 50 épocas para alcanzar rendimiento de alto nivel.

**Referencias Oficiales**

- [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) – Paper oficial.  
- [Repositorio GitHub oficial](https://github.com/facebookresearch/Mask2Former.git)  
- [Modelo en Hugging Face](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)

**Créditos**  
Todas las explicaciones, diagramas e ilustraciones fueron construidas con base en el contenido del paper oficial y sus recursos asociados.  
**Todo el crédito por la arquitectura y avances técnicos es de los autores originales. Este proyecto es una interfaz educativa/demostrativa sin ninguna autoría sobre el modelo ni sus fundamentos.**
""")
