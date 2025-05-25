import streamlit as st
import os

# Configuración de la página
st.set_page_config(page_title="Mask2Former App", layout="wide")
st.title("Bienvenido a Segmentación con Mask2Former")

st.markdown("""
## Navegación

Utiliza la barra lateral para explorar las siguientes secciones:

1. **Descripción de la Tarea**: Introducción general a la tarea de segmentación semántica.
2. **Detalle de la Arquitectura**: Explicación de Mask2Former y la arquitectura utilizada.
3. **Inferencia**: Prueba interactiva del modelo con imágenes, cámara o video en vivo.

---
""")

# Mostrar botones para ver contenido adicional
if st.button("Ver imagen de segmentación"):
    st.image("sources/demo.jpg", caption="Ejemplo de segmentación con Mask2Former", use_container_width=True)

if st.button("Ver video demostración"):
    st.video("sources/video_demostracion.mp4")