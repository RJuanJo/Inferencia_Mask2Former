import streamlit as st

st.set_page_config(page_title="Detalle de la Arquitectura", layout="wide")

st.title("Funcionamiento de Mask2Former para Segmentación de Instancias")

st.markdown("### Componentes Clave")

st.markdown("#### Atención Enmascarada (Masked Attention)")
st.markdown("""
- Reemplaza la atención global de los transformers tradicionales.  
- **Enfoca la atención solo en regiones de interés** (máscaras predichas), evitando distracciones en fondos irrelevantes.  
- Ejemplo: Si una *query* predice un "perro", la atención se limita a la región de ese perro, no a toda la imagen.
""")

st.markdown("#### Características Multi-Escala")
st.markdown("""
- Combina características de **alta resolución** (1/8 de la imagen original) y **baja resolución** (1/32) en un esquema eficiente.  
- Ventaja: Detecta objetos pequeños con precisión gracias a detalles finos de alta resolución.
""")

st.markdown("#### Optimizaciones")
st.markdown("""
- **Orden de capas**: Cambia el orden de las capas de auto-atención y atención cruzada para acelerar la convergencia.  
- **Queries aprendibles**: Inicializa las queries con valores aprendidos (no ceros), actuando como "propuestas de región".  
- **Eliminación de dropout**: Mejora el rendimiento al remover regularización innecesaria.
""")

st.markdown("#### Pérdida Eficiente")
st.markdown("""
- Calcula la pérdida en puntos muestreados aleatoriamente (ej: 112×112 puntos), reduciendo memoria en entrenamiento.
""")

st.markdown("### Diferencias con Transformers Tradicionales (ej: DETR)")

st.table({
    "Aspecto": [
        "Atención",
        "Queries",
        "Manejo de Escalas",
        "Entrenamiento"
    ],
    "Transformers Tradicionales": [
        "Global (toda la imagen)",
        "Inicializadas en cero",
        "Características fijas",
        "Lento (500+ épocas para converger)"
    ],
    "Mask2Former": [
        "Enmascarada (regiones específicas)",
        "Aprendibles y supervisadas",
        "Multi-escala dinámica",
        "Rápido (50 épocas para alto rendimiento)"
    ]
})

st.markdown("### ¿Por qué es ideal para segmentación de instancias?")
st.markdown("""
- **Precisión en bordes**: Mejora la calidad de los límites de objetos (AP de borde = 36.2 en COCO).  
- **Manejo de superposición**: Separa instancias cercanas gracias a la atención localizada.  
- **Detalle en objetos pequeños**: Usa características de alta resolución para capturar detalles finos.
""")
