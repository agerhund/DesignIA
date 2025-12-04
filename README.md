# DesignIA - Deployment

Esta carpeta contiene todo lo necesario para desplegar la aplicación en Streamlit Cloud.

## 1. Preparación de Git y LFS (Archivos Grandes)
Debido a que los modelos pesan más de 100MB, necesitas usar Git LFS.

1.  **Inicializar Git:**
    ```bash
    cd deploy_app
    git init
    ```

2.  **Instalar Git LFS (si no lo tienes):**
    *   Mac: `brew install git-lfs`
    *   Windows: Descargar instalador de git-lfs.github.com
    *   Ejecutar: `git lfs install`

3.  **Configurar LFS para los modelos:**
    ```bash
    git lfs track "models/*.pth"
    git lfs track "models/*.tar"
    git add .gitattributes
    ```

4.  **Subir a GitHub:**
    ```bash
    git add .
    git commit -m "Initial commit for deployment"
    # Crea un repo vacío en GitHub y luego:
    git remote add origin <TU_URL_DEL_REPO>
    git push -u origin main
    ```

## 2. Despliegue en Streamlit Cloud
1.  Ve a [share.streamlit.io](https://share.streamlit.io/).
2.  Conecta tu cuenta de GitHub.
3.  Selecciona "New App".
4.  Elige el repositorio que acabas de crear.
5.  **Main file path:** `app.py`
6.  Dale a "Deploy".

Streamlit Cloud detectará automáticamente `requirements.txt` y `packages.txt` para instalar todo lo necesario.
