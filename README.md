# Segmentación con Mask2Former

Esta aplicación realiza inferencia utilizando el modelo **Mask2Former** sobre imágenes, cámara en vivo o video en tiempo real. Fue desarrollada con fines demostrativos y educativos, y **no representa una implementación propia del modelo**, sino una interfaz para facilitar su uso.

## ¿Qué es Mask2Former?

Mask2Former es un modelo de segmentación universal propuesto por Facebook AI Research. Puede usarse para segmentación semántica, por instancia y panóptica.

- Paper oficial: [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
- Repositorio en GitHub: [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)
- Modelo usado en esta app: [`facebook/mask2former-swin-large-coco-panoptic`](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)

## Funcionalidades

- Carga de imágenes para segmentación.
- Segmentación usando cámara en vivo.
- Segmentación de video en tiempo real, con opción para guardar el video resultante.
- Interfaz multipágina con descripción y detalles del modelo.

> **Nota técnica:**  
> Esta aplicación ha sido **configurada para funcionar completamente en CPU**, sin necesidad de aceleración por GPU. A pesar de ello, la inferencia es razonablemente rápida y permite una experiencia fluida si se utilizan imagenes. La segmentacion a tiempo real en video si puede ser menos eficiente al no tener uso de GPU sin embargo está la opcion de guardar en un video toda la grabacion en vivo el cual se puede ver en la raiz del repositorio y ver con mas fluides la segmentacion de este mismo.

## Estructura del Proyecto

├── model/ ← Archivos del modelo (no incluidos por tamaño)
├── pages/ ← Páginas de la app Streamlit
│ ├── Descripcion_Tarea.py
│ ├── Detalle_Arquitectura.py
│ └── Inferencia.py
├── sources/ ← Imagen y video de demostración
│ ├── demo.jpg
│ └── video_demostracion.mp4
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt

### Crea un Entorno Virtual (Recomendado)**

python \-m venv venv  
\# En Windows  
.\\venv\\Scripts\\activate  
\# En macOS/Linux  
source venv/bin/activate

### Instala las Dependencias**

pip install \-r requirements.txt

### **Ejecuta la Aplicación**

streamlit run Pagina\_principal.py

**Nota:** Asegúrate de tener una cámara funcional si deseas probar la segmentación en vivo.

## **Uso con Docker**

También puedes ejecutar la aplicación utilizando Docker para un entorno aislado:

### **Construye la Imagen**

docker build \-t mask2former-app .

### **Ejecuta el Contenedor**

docker run \-p 8501:8080 mask2former-app

Accede a la aplicación desde tu navegador en: [http://localhost:8501](http://localhost:8501)

## **Demostración**

La página principal de la aplicación incluye una demostración visual del modelo en video e imagen.

Solo haz clic en los botones correspondientes para visualizarla.

## **Créditos**

* **Modelo original:** [facebook/mask2former-swin-large-coco-panoptic](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)  
* **Investigación:** [Masked-attention Mask Transformer](https://arxiv.org/abs/2112.01527) (Paper CVPR 2022\)  
* **Código base:** Adaptado desde la documentación oficial de Hugging Face y Streamlit.

## **Licencia**

Este proyecto es solo para uso académico y demostrativo.

Todos los derechos del modelo pertenecen a sus creadores originales (Meta AI).
