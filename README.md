# Segmentación con Mask2Former

Esta aplicación realiza inferencia utilizando el modelo **Mask2Former** sobre imágenes, cámara en vivo o video en tiempo real. Fue desarrollada con fines demostrativos y educativos, y **no representa una implementación propia del modelo**, sino una interfaz para facilitar su uso.

## ¿Qué es Mask2Former?

Mask2Former es un modelo de segmentación universal propuesto por Facebook AI Research. Puede usarse para segmentación semántica, por instancia y panóptica.

- **Paper oficial:** [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)  
- **Repositorio en GitHub:** [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)  
- **Modelo usado en esta app:** [`facebook/mask2former-swin-large-coco-panoptic`](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)

## Funcionalidades

- Carga de imágenes para segmentación.
- Segmentación usando cámara en vivo.
- Segmentación de video en tiempo real, con opción para guardar el video resultante.
- Interfaz multipágina con descripción y detalles del modelo.

> **Nota técnica:**  
> Esta aplicación ha sido **configurada para funcionar completamente en CPU**, sin necesidad de aceleración por GPU. A pesar de ello, la inferencia es razonablemente rápida y permite una experiencia fluida si se utilizan imágenes.  
> La segmentación a tiempo real en video puede ser menos eficiente sin GPU, sin embargo, existe la opción de guardar la grabación en un video (como el `video_demostracion.mp4`) que puede reproducirse posteriormente para visualizar la segmentación de manera fluida.

## Estructura del Proyecto

```
├── docker/                  ← Funcionamiento de la aplicacion en un contenedor de docker
├── model/                  ← Archivos del modelo (no incluidos por tamaño)
├── pages/                  ← Páginas de la app Streamlit
│   ├── Descripcion\_Tarea.py
│   ├── Detalle\_Arquitectura.py
│   └── Inferencia.py
├── sources/                ← Imagen y video de demostración
│   ├── demo.jpg
│   └── video\_demostracion.mp4
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── Pagina\_principal.py     ← Página principal de la app

````

---

## Ejecución Local

### Crea un entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
# En Windows
.\venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
````
---

## Preparación de la Carpeta `model/`

La carpeta `model/` no está incluida en este repositorio debido a su tamaño. Para que la aplicación funcione correctamente, debes crearla manualmente y colocar dentro los archivos del modelo preentrenado.

### Pasos:

1. **Crea la carpeta en la raíz del proyecto** si aún no existe:

```bash
mkdir model
```

2. **Descarga los siguientes archivos desde la página oficial del modelo en Hugging Face**:
   [`facebook/mask2former-swin-large-coco-panoptic`](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic) en la parte de **"Files and Versions"**

   Archivos requeridos:

   * `config.json`
   * `preprocessor_config.json`
   * `pytorch_model.bin`

3. **Coloca esos tres archivos dentro de la carpeta `model/`**, así:

### En Linux/macOS

```bash
cp /ruta/a/los/archivos/config.json model/
cp /ruta/a/los/archivos/preprocessor_config.json model/
cp /ruta/a/los/archivos/pytorch_model.bin model/
```

### En Windows (PowerShell)

```powershell
Copy-Item "C:\ruta\a\los\archivos\config.json" model\
Copy-Item "C:\ruta\a\los\archivos\preprocessor_config.json" model\
Copy-Item "C:\ruta\a\los\archivos\pytorch_model.bin" model\
```

**Reemplaza `/ruta/a/los/archivos/` o `C:\ruta\a\los\archivos\` con la ruta donde tienes los archivos descargados.**

> Nota: Asegúrate de que los archivos están correctamente nombrados y completos. Sin estos, la aplicación **no podrá ejecutar la inferencia**.

---

### Instala las dependencias

```bash
pip install -r requirements.txt
```

### Ejecuta la aplicación

```bash
streamlit run Pagina_Principal.py
```

> Asegúrate de tener una cámara funcional si deseas probar la segmentación en vivo.

---

## Uso con Docker

### Estructura para Docker

La carpeta `docker/` debe contener todo el proyecto para la ejecución dentro del contenedor:

```
docker/
├── model/                  ← Archivos del modelo (config.json, pytorch_model.bin, preprocessor_config.json)
├── pages/                  ← Páginas Streamlit (Descripcion_Tarea.py, Detalle_Arquitectura.py, Inferencia.py)
├── sources/                ← Archivos de demo (demo.jpg, video_demostracion.mp4)
├── Pagina_Principal.py     ← Página principal
├── requirements.txt        ← Dependencias Python
└── Dockerfile              ← Archivo para crear la imagen Docker
```

> **Importante:** La carpeta `model/` no está en el repositorio y debe copiarse manualmente dentro de `docker/` antes de construir la imagen.

### Construir la imagen Docker

Desde la raíz del proyecto, ingresa a la carpeta `docker`:

```bash
cd docker
```

Construye la imagen con:

```bash
docker build -t mask2former-app .
```
### Ejecutar el contenedor

Ejecuta el contenedor mapeando el puerto:

```bash
docker run -p 8501:8501 mask2former-app
```
### Accede a la aplicación

Abre en tu navegador:

```
http://localhost:8501
```
---

## Demostración

La página principal de la aplicación incluye una demostración visual del modelo en video e imagen.
Solo haz clic en los botones correspondientes para visualizarla.

---

## Créditos

* **Modelo original:** [facebook/mask2former-swin-large-coco-panoptic](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)
* **Investigación:** [Masked-attention Mask Transformer](https://arxiv.org/abs/2112.01527) (CVPR 2022)
* **Código base:** Adaptado desde la documentación oficial de Hugging Face y Streamlit.

---

## Licencia

Este proyecto es solo para **uso académico y demostrativo**.
Todos los derechos del modelo pertenecen a sus creadores originales (Meta AI).
