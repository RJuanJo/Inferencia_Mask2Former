import streamlit as st
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor
from PIL import Image
import torch
import numpy as np
import cv2
import time
from datetime import datetime

# Configuración para Docker (https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config)
st.set_page_config(layout="wide")
st.title("Segmentación con Mask2Former (Imagen, Cámara, Video en Vivo)")

# Paleta de colores para visualización de máscaras (definida arbitrariamente) 
COLOR_PALETTE = np.array([
    [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
    [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
    [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
    [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
    [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]
], dtype=np.uint8)

torch.set_num_threads(4)  # Limita hilos

@st.cache_resource  # Cachear recursos
def load_model():
    # Carga modelo y procesador
    model = AutoModelForUniversalSegmentation.from_pretrained("./model", local_files_only=True)
    processor = AutoImageProcessor.from_pretrained("./model", local_files_only=True)
    return model, processor

model, processor = load_model()

def segment_image_with_labels(image, alpha=0.5):
    rgb = np.array(image.convert("RGB"))  # PIL a numpy RGB
    resized = image.resize((512, 512))  # Ajustar tamaño para el modelo
    inputs = processor(images=resized, return_tensors="pt")  # Preprocesar imagen 
    with torch.no_grad():  # Desactiva cálculo de gradientes para eficiencia en inferencia
        outputs = model(**inputs)
    # Post-procesamiento de la segmentación (https://huggingface.co/docs/transformers/main_classes/feature_extraction#transformers.SegmentationProcessor.post_process_semantic_segmentation)
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb.shape[:2]])[0]
    seg_mask = result.numpy()

    colored_mask = np.zeros_like(rgb)  # Crear máscara coloreada para visualización
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        colored_mask[mask] = COLOR_PALETTE[label_id % 20]

    # Combinar imagen original y máscara coloreada con transparencia alpha (https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html)
    blended = cv2.addWeighted(rgb, 1 - alpha, colored_mask, alpha, 0)

    # Añadir texto con etiquetas para regiones segmentadas grandes (https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html)
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        if mask.sum() > 300:  # Umbral para evitar etiquetas en pequeñas regiones
            y, x = np.where(mask)
            label = model.config.id2label.get(label_id, str(label_id))  # Diccionario etiquetas del modelo (https://huggingface.co/docs/transformers/model_doc/universal_segmentation#config)
            cv2.putText(blended, label, (int(x.mean()), int(y.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return blended, seg_mask

def segment_frame(frame, alpha=0.5, show_labels=True):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir BGR(OpenCV) a RGB (https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)
    pil_img = Image.fromarray(rgb_frame).resize((256, 256))  # Resize para el modelo (https://pillow.readthedocs.io)
    inputs = processor(images=pil_img, return_tensors="pt")  # Preprocesar imagen (https://huggingface.co/docs/transformers/main_classes/feature_extraction)
    with torch.no_grad():  # Inferencia sin gradientes 
        outputs = model(**inputs)
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb_frame.shape[:2]])[0]  # Post-proceso (https://huggingface.co/docs/transformers/main_classes/feature_extraction)
    seg_mask = result.numpy()

    colored_mask = np.zeros_like(rgb_frame)  # Máscara coloreada
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        colored_mask[mask] = COLOR_PALETTE[label_id % 20]

    blended = cv2.addWeighted(rgb_frame, 1 - alpha, colored_mask, alpha, 0)  # Mezclar (https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html)

    if show_labels:
        for label_id in np.unique(seg_mask):
            mask = seg_mask == label_id
            if mask.sum() > 500:  # Etiquetas solo para regiones grandes
                y, x = np.where(mask)
                label = model.config.id2label.get(label_id, str(label_id))  # Etiqueta (https://huggingface.co/docs/transformers/model_doc/universal_segmentation#config)
                cv2.putText(blended, label, (int(x.mean()), int(y.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return blended, seg_mask

def show_legend(seg_mask):
    unique_labels = np.unique(seg_mask)
    cols = st.columns(min(len(unique_labels), 5))  # Streamlit columnas para leyenda
    for i, label_id in enumerate(unique_labels):
        label = model.config.id2label.get(label_id, str(label_id))  # Obtener etiqueta
        color = COLOR_PALETTE[label_id % 20]
        with cols[i % 5]:
            st.markdown(
                f"<div style='display:flex; align-items:center;'>"
                f"<div style='width:20px; height:20px; background-color: rgb{tuple(color)}; "
                f"border-radius:3px; margin-right:8px;'></div>"
                f"<span>{label}</span></div>", unsafe_allow_html=True)  # HTML seguro para mostrar leyenda de colores

# Tabs de la app (https://docs.streamlit.io/library/api-reference/layout/st.tabs)
tab1, tab2, tab3 = st.tabs(["Subir Imagen", "Cámara - Tomar Foto", "Video en Vivo"])

# Tab 1: Segmentación de imagen subida
with tab1:
    st.header("Segmentación de Imagen Subida")
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])  # Subir archivo
    if uploaded_file:
        image = Image.open(uploaded_file)  # Abrir imagen con PIL (https://pillow.readthedocs.io)
        alpha = st.slider("Transparencia", 0.1, 1.0, 0.5, key="alpha_img")  # Slider para transparencia
        result_img, seg_mask = segment_image_with_labels(image, alpha)
        st.image(result_img, channels="RGB", caption="Resultado", use_container_width=True)  # Mostrar imagen 
        show_legend(seg_mask)  # Mostrar leyenda de colores

# Tab 2: Captura con cámara
with tab2:
    st.header("Captura con Cámara")
    cam_active = st.checkbox("Activar cámara")  # Checkbox para activar cámara 
    if cam_active:
        col_preview, col_empty = st.columns([1, 3])  # Layout con columnas
        with col_preview:
            img_file_buffer = st.camera_input("Toma una foto")  # Entrada cámara 
        if img_file_buffer:
            image = Image.open(img_file_buffer)
            alpha = st.slider("Transparencia", 0.1, 1.0, 0.5, key="alpha_cam")
            result_img, seg_mask = segment_image_with_labels(image, alpha)
            st.image(result_img, channels="RGB", caption="Segmentación", use_container_width=True)
            show_legend(seg_mask)

# Tab 3: Segmentación en vivo (streaming video)
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode  # https://github.com/whitphx/streamlit-webrtc
import av  # Para manejo de frames de video (https://pyav.org/docs/develop/)

with tab3:
    st.header("Segmentación en Tiempo Real (Video en Vivo, vía navegador)")

    alpha = st.slider("Transparencia", 0.1, 1.0, 0.6, key="alpha_webcam")
    show_labels = st.checkbox("Mostrar etiquetas", True, key="labels_webcam")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")  # Convertir frame a ndarray (https://pyav.org/docs/develop/api/video.html#av.video.frame.VideoFrame.to_ndarray)
            segmented, _ = segment_frame(img, alpha=alpha, show_labels=show_labels)
            return av.VideoFrame.from_ndarray(segmented, format="bgr24")  # Crear frame para enviar

    webrtc_streamer(
        key="streaming",
        mode=WebRtcMode.SENDRECV,  # Modo envío y recepción de video (https://github.com/whitphx/streamlit-webrtc#usage)
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )