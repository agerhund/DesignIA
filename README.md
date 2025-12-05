# DesignIA - Recomendador de Muebles Inteligente

DesignIA es una aplicación inteligente que te ayuda a diseñar tu salón ideal. A partir de una imagen panorámica (360°) de tu habitación vacía, la aplicación detecta automáticamente la geometría del espacio (paredes, puertas, ventanas) y te sugiere una distribución óptima de muebles, recomendándote productos reales de IKEA que se ajustan a tu estilo y presupuesto.

## Características

*   **Escaneo de Habitación 3D**: Utiliza **HorizonNet** para detectar la estructura de la habitación a partir de una sola imagen panorámica.
*   **Diseño Automático**: Algoritmos de optimización espacial para colocar muebles respetando zonas de paso y distancias de visualización (TV-Sofá).
*   **Recomendación Estilística**: Un modelo basado en **BERT** analiza el estilo de los muebles para sugerir combinaciones coherentes.
*   **Visualización Interactiva**: Visualiza tu futuro salón en 3D directamente en el navegador.
*   **Presupuesto Ajustable**: Define cuánto quieres gastar y la IA buscará la mejor combinación calidad/precio.

## Requisitos Previos

Debido al uso de modelos de Inteligencia Artificial avanzados, este proyecto requiere descargar archivos de gran tamaño.

1.  **Python 3.8+**
2.  **Git LFS (Large File Storage)**: Imprescindible para descargar los modelos de IA.
    *   Instalación: [https://git-lfs.com/](https://git-lfs.com/)
    *   O ejecuta: `git lfs install`

## Instalación

1.  **Clonar el repositorio**
    Asegúrate de tener Git LFS instalado antes de clonar.
    ```bash
    git lfs install
    git clone https://github.com/agerhund/DesignIA.git
    cd DesignIA
    ```
    *Nota: La descarga puede tardar unos minutos debido a los modelos (~1.7 GB).*

2.  **Crear un entorno virtual (Recomendado)**
    ```bash
    python -m venv venv
    # En Windows:
    venv\Scripts\activate
    # En Mac/Linux:
    source venv/bin/activate
    ```

3.  **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución

Para iniciar la aplicación web localmente:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador (usualmente en `http://localhost:8501`).

## Notas sobre el Rendimiento

*   **Memoria RAM**: Se recomienda disponer de al menos **8 GB de RAM**, ya que los modelos de visión y lenguaje se cargan en memoria.
*   **GPU**: Si dispones de una GPU NVIDIA (CUDA) o un Mac con chip M-series (MPS), la aplicación intentará usarla para acelerar la detección. De lo contrario, funcionará en CPU (más lento).
*   **Streamlit Cloud**: Es posible que esta aplicación **no funcione** en la capa gratuita de Streamlit Cloud debido a las limitaciones de memoria y almacenamiento (los modelos exceden el límite habitual). Se recomienda ejecutar en local o en un servidor con mayores recursos.

## Estructura del Proyecto

*   `app.py`: Punto de entrada de la aplicación Streamlit.
*   `logic.py`: Lógica principal (IA, geometría, recomendación).
*   `models/`: Contiene los pesos de los modelos (HorizonNet y BERT Encoder).
*   `data/`: Base de datos de muebles (CSV).
*   `horizonnet/`: Código fuente del modelo de visión computacional.

## Autor

**Andrés Gerlotti Slusnys**
Máster de Data Science, Business Analytics y Big Data
Universidad Complutense de Madrid
© 2025
