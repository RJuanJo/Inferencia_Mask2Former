import streamlit as st
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor
from PIL import Image
import torch
import numpy as np
import cv2
import time
from datetime import datetime
import os

# Configuración de página en Streamlit para diseño ancho
st.set_page_config(layout="wide")
st.title("Segmentación con Mask2Former (Imagen, Cámara, Video en Vivo)")

# Paleta de colores para visualización de máscaras segmentadas
COLOR_PALETTE = np.array([
    [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
    [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
    [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
    [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
    [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]
], dtype=np.uint8)

torch.set_num_threads(4)  # Limitar threads para evitar sobrecarga - https://pytorch.org/docs/stable/generated/torch.set_num_threads.html

@st.cache_resource
def load_model():
    # Carga el modelo preentrenado y el procesador para segmentación semántica
    model = AutoModelForUniversalSegmentation.from_pretrained("./model", local_files_only=True)
    processor = AutoImageProcessor.from_pretrained("./model", local_files_only=True)
    return model, processor

model, processor = load_model()

def segment_image_with_labels(image, alpha=0.5):
    # Segmenta una imagen PIL y superpone la máscara segmentada con colores y etiquetas
    rgb = np.array(image.convert("RGB"))
    resized = image.resize((512, 512))  # Ajuste de tamaño para el modelo
    inputs = processor(images=resized, return_tensors="pt")  # Preprocesamiento con Huggingface
    with torch.no_grad():
        outputs = model(**inputs)  # Inferencia sin gradientes
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb.shape[:2]])[0]
    seg_mask = result.numpy()

    colored_mask = np.zeros_like(rgb)
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        colored_mask[mask] = COLOR_PALETTE[label_id % 20]

    blended = cv2.addWeighted(rgb, 1 - alpha, colored_mask, alpha, 0)  # Mezcla de imagen original y máscara (OpenCV) - https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html

    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        if mask.sum() > 300:  # Umbral para mostrar etiquetas grandes
            y, x = np.where(mask)
            label = model.config.id2label.get(label_id, str(label_id))  # Obtiene nombre de clase
            cv2.putText(blended, label, (int(x.mean()), int(y.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # Añade texto (OpenCV) - https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    return blended, seg_mask

def segment_frame(frame, alpha=0.5, show_labels=True):
    # Segmenta un frame (imagen BGR) capturado de video y superpone máscara y etiquetas
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para PIL y modelo
    pil_img = Image.fromarray(rgb_frame).resize((256, 256))
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb_frame.shape[:2]])[0]
    seg_mask = result.numpy()

    colored_mask = np.zeros_like(rgb_frame)
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        colored_mask[mask] = COLOR_PALETTE[label_id % 20]

    blended = cv2.addWeighted(rgb_frame, 1 - alpha, colored_mask, alpha, 0)

    if show_labels:
        for label_id in np.unique(seg_mask):
            mask = seg_mask == label_id
            if mask.sum() > 500:
                y, x = np.where(mask)
                label = model.config.id2label.get(label_id, str(label_id))
                cv2.putText(blended, label, (int(x.mean()), int(y.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return blended, seg_mask

def show_legend(seg_mask):
    # Muestra la leyenda con colores y etiquetas para la segmentación
    unique_labels = np.unique(seg_mask)
    cols = st.columns(min(len(unique_labels), 5))
    for i, label_id in enumerate(unique_labels):
        label = model.config.id2label.get(label_id, str(label_id))
        color = COLOR_PALETTE[label_id % 20]
        with cols[i % 5]:
            st.markdown(
                f"<div style='display:flex; align-items:center;'>"
                f"<div style='width:20px; height:20px; background-color: rgb{tuple(color)}; "
                f"border-radius:3px; margin-right:8px;'></div>"
                f"<span>{label}</span></div>", unsafe_allow_html=True)

# Tabs de la app (https://docs.streamlit.io/library/api-reference/layout/st.tabs)
tab1, tab2, tab3 = st.tabs(["Subir Imagen", "Cámara - Tomar Foto", "Video en Vivo"])

# Tab 1: carga y segmentación de imagen estática
with tab1:
    st.header("Segmentación de Imagen Subida")
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg"])  # Carga archivo
    if uploaded_file:
        image = Image.open(uploaded_file)
        alpha = st.slider("Transparencia", 0.1, 1.0, 0.5, key="alpha_img")  # Control de transparencia
        result_img, seg_mask = segment_image_with_labels(image, alpha)
        st.image(result_img, channels="RGB", caption="Resultado", use_column_width=False)
        show_legend(seg_mask)

# Tab 2: captura y segmentación desde cámara con botón para activar
with tab2:
    st.header("Captura con Cámara")
    cam_active = st.checkbox("Activar cámara")
    if cam_active:
        col_preview, col_empty = st.columns([1, 3])
        with col_preview:
            img_file_buffer = st.camera_input("Toma una foto")  # Streamlit camera input widget - https://docs.streamlit.io/library/api-reference/widgets/st.camera_input
        if img_file_buffer:
            image = Image.open(img_file_buffer)
            alpha = st.slider("Transparencia", 0.1, 1.0, 0.5, key="alpha_cam")
            result_img, seg_mask = segment_image_with_labels(image, alpha)
            st.image(result_img, channels="RGB", caption="Segmentación", use_column_width=False)
            show_legend(seg_mask)

# Tab 3: segmentación en vivo desde video webcam con control de FPS, etiquetas y grabación
with tab3:
    st.header("Segmentación en Tiempo Real (Video en Vivo)")
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Transparencia", 0.1, 1.0, 0.6, key="alpha_vid")
        show_labels = st.checkbox("Mostrar etiquetas", True, key="labels_vid")
    with col2:
        target_fps = st.slider("FPS objetivo", 1, 30, 10)

    save_video = st.checkbox("Guardar video segmentado", False, key="save_vid")

    FRAME_WINDOW = st.image([], use_column_width=True)

    # Estado para controlar la grabación y renderizado en Streamlit
    if "recording" not in st.session_state:
        st.session_state.recording = False

    # Botones para iniciar y detener grabación
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if not st.session_state.recording:
            if st.button("Iniciar grabación"):
                st.session_state.recording = True
                st.rerun()  # Fuerza rerun para actualizar estado y UI
    with col_btn2:
        if st.session_state.recording:
            if st.button("Parar grabación"):
                st.session_state.recording = False
                st.rerun()

    video_path = None

    # Bucle principal de captura y segmentación de frames cuando está grabando
    if st.session_state.recording:
        cap = cv2.VideoCapture(0)  # Acceso a cámara (OpenCV) - https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
        frame_count, fps, last_log = 0, 0, time.time()
        prev_time = time.time()
        out = None

        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara.")
        else:

            if save_video:
                os.makedirs("sources/outputs", exist_ok=True)  # Crea la carpeta si no existe
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = f"sources/outputs/video_segmentado_{now}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para grabación video MP4 - https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html
                out = cv2.VideoWriter(video_path, fourcc, target_fps, (frame.shape[1], frame.shape[0]))

            try:
                while st.session_state.recording:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    curr_time = time.time()
                    if curr_time - prev_time >= 1. / target_fps:  # Control FPS objetivo
                        segmented_frame, _ = segment_frame(frame, alpha, show_labels)

                        if save_video and out is not None:
                            # Guardar frame segmentado en video
                            out.write(cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR))

                        frame_count += 1
                        if curr_time - last_log >= 1.0:
                            fps = frame_count / (curr_time - last_log)
                            frame_count = 0
                            last_log = curr_time

                        cv2.putText(segmented_frame, f"FPS: {fps:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        FRAME_WINDOW.image(segmented_frame, channels="RGB")  # Mostrar frame segmentado en Streamlit

                        prev_time = curr_time
            finally:
                cap.release()  # Liberar cámara cuando termina
                if out is not None:
                    out.release()  # Liberar archivo video