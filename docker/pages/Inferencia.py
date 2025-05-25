import streamlit as st
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor
from PIL import Image
import torch
import numpy as np
import cv2
import time
from datetime import datetime

# Configuración para Docker
st.set_page_config(layout="wide")
st.title("Segmentación con Mask2Former (Imagen, Cámara, Video en Vivo)")

COLOR_PALETTE = np.array([
    [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
    [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
    [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
    [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
    [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]
], dtype=np.uint8)

torch.set_num_threads(4)

@st.cache_resource
def load_model():
    model = AutoModelForUniversalSegmentation.from_pretrained("./model", local_files_only=True)
    processor = AutoImageProcessor.from_pretrained("./model", local_files_only=True)
    return model, processor

model, processor = load_model()

def segment_image_with_labels(image, alpha=0.5):
    rgb = np.array(image.convert("RGB"))
    resized = image.resize((512, 512))
    inputs = processor(images=resized, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb.shape[:2]])[0]
    seg_mask = result.numpy()
    colored_mask = np.zeros_like(rgb)
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        colored_mask[mask] = COLOR_PALETTE[label_id % 20]
    blended = cv2.addWeighted(rgb, 1 - alpha, colored_mask, alpha, 0)
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        if mask.sum() > 300:
            y, x = np.where(mask)
            label = model.config.id2label.get(label_id, str(label_id))
            cv2.putText(blended, label, (int(x.mean()), int(y.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return blended, seg_mask

def segment_frame(frame, alpha=0.5, show_labels=True):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

# Tabs
tab1, tab2, tab3 = st.tabs(["Subir Imagen", "Cámara - Tomar Foto", "Video en Vivo"])

# Cargar imagen
with tab1:
    st.header("Segmentación de Imagen Subida")
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        alpha = st.slider("Transparencia", 0.1, 1.0, 0.5, key="alpha_img")
        result_img, seg_mask = segment_image_with_labels(image, alpha)
        st.image(result_img, channels="RGB", caption="Resultado", use_container_width=True)
        show_legend(seg_mask)

# Cámara - Captura
with tab2:
    st.header("Captura con Cámara")
    cam_active = st.checkbox("Activar cámara")
    if cam_active:
        col_preview, col_empty = st.columns([1, 3])
        with col_preview:
            img_file_buffer = st.camera_input("Toma una foto")
        if img_file_buffer:
            image = Image.open(img_file_buffer)
            alpha = st.slider("Transparencia", 0.1, 1.0, 0.5, key="alpha_cam")
            result_img, seg_mask = segment_image_with_labels(image, alpha)
            st.image(result_img, channels="RGB", caption="Segmentación", use_container_width=True)
            show_legend(seg_mask)

# Video en Vivo
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

with tab3:
    st.header("Segmentación en Tiempo Real (Video en Vivo, vía navegador)")

    alpha = st.slider("Transparencia", 0.1, 1.0, 0.6, key="alpha_webcam")
    show_labels = st.checkbox("Mostrar etiquetas", True, key="labels_webcam")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            segmented, _ = segment_frame(img, alpha=alpha, show_labels=show_labels)
            return av.VideoFrame.from_ndarray(segmented, format="bgr24")

    webrtc_streamer(
        key="streaming",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )