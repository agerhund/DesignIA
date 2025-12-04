import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

# Importar la lógica principal del proyecto
import logic as logic

# --- CONFIGURACIÓN INICIAL DE LA PÁGINA ---
st.set_page_config(page_title="DesignIA - Recomendador de Muebles inteligente", layout="wide")

# --- CARGAR ESTILOS CSS ---
def cargar_estilo():
    """Define y aplica estilos CSS para la UI de Streamlit."""
    st.markdown("""
        <style>
        /* Ocultar elementos de sistema de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Estilo de Botones (Azul IKEA) */
        div.stButton > button:first-child {
            background-color: #0051ba; 
            color: white; 
            border-radius: 8px; 
            font-weight: bold; 
            border: none; 
            padding: 0.5rem 1rem;
        }
        div.stButton > button:first-child:hover { 
            background-color: #003e8f; 
            border: none; 
        }
        
        /* Estilo de las Tarjetas de Producto */
        .product-card {
            background-color: white; 
            padding: 15px; 
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05); 
            margin-bottom: 15px; 
            border: 1px solid #eee;
            color: black !important; /* Forzar texto negro dentro de la tarjeta blanca */
        }
        </style>
    """, unsafe_allow_html=True)

cargar_estilo()

# --- CONSTANTES Y RUTAS DE RECURSOS ---
# Rutas relativas para despliegue
csv_path = "data/furniture_data.csv"
model_path = "models/bert_style_encoder.pth"
horizon_path = "models/horizonnet_model.pth"
cache_path = "vectores_cache.pkl" # Se generará en el directorio de trabajo

# --- SIDEBAR: INFORMACIÓN DEL PROYECTO ---
with st.sidebar:
    st.title("DesignIA - Asistente de Diseño")
    st.markdown("---")
    st.markdown("**Trabajo de fin de Máster**")
    st.caption("Máster de Data Science, Business Analytics y Big Data")
    st.caption("Universidad Complutense de Madrid")
    st.markdown("---")
    st.markdown("Desarrollado por **Andrés Gerlotti Slusnys**")
    st.markdown("© 2025")
    
    # Indicador de estado para debugging
    with st.expander("Estado del Sistema", expanded=False):
        st.success("Motor Gráfico: Activo")
        st.success("Modelo NLP: Cargado")
        if os.path.exists(horizon_path):
            st.success("HorizonNet: Conectado")
        else:
            st.error("HorizonNet: No encontrado")

# --- INICIALIZACIÓN DEL ESTADO DE SESIÓN ---
if 'stage' not in st.session_state: st.session_state.stage = 0
if 'room_data' not in st.session_state: st.session_state.room_data = None
if 'muebles_df' not in st.session_state: st.session_state.muebles_df = None
if 'data_manager' not in st.session_state: st.session_state.data_manager = None

# --- 1. CARGA DE DATOS Y MODELOS (Cacheado) ---
@st.cache_resource
def init_backend(csv, cache, model):
    """Inicializa DataManager y carga/genera el DataFrame de muebles."""
    dm = logic.DataManager(csv, cache, model)
    df = dm.cargar_datos()
    return dm, df

try:
    if st.session_state.data_manager is None:
        with st.spinner("Cargando base de datos de muebles y modelos IA..."):
            dm, df = init_backend(csv_path, cache_path, model_path)
            st.session_state.data_manager = dm
            st.session_state.muebles_df = df
    st.sidebar.success(f"Base de datos cargada: {len(st.session_state.muebles_df)} items")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# --- INTERFAZ PRINCIPAL ---
st.title("Recomendador de Muebles Inteligente")
st.markdown("Sube una panorámica, detecta el espacio y obtén el diseño ideal según tu presupuesto.")

# --- PASO 1: CARGA DE IMAGEN Y DETECCIÓN ---
st.header("1. Escaneo de Habitación")
uploaded_file = st.file_uploader("Sube tu imagen panorámica (360)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_container_width=True)
    
    if st.button("Analizar la habitación"):
        with st.spinner("Detectando la geometría..."):
            # Guardar temporalmente la imagen para que la librería pueda leerla
            with open("temp_pano.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Instanciar y ejecutar el detector de layout (HorizonNet)
            try:
                detector = logic.RoomLayoutDetector(horizon_path)
                room_data = detector.detect_layout("temp_pano.jpg")
                
                # Validación de datos y manejo de fallos
                if room_data is None or not isinstance(room_data, dict) or 'width' not in room_data:
                    st.error("**Detección fallida.** El modelo de Computer Vision no pudo extraer las dimensiones ni los obstáculos. Asegúrate de que el modelo HorizonNet está configurado y funcionando correctamente.")
                    st.session_state.room_data = None
                    st.session_state.stage = 0
                else:
                    st.session_state.room_data = room_data
                    st.session_state.stage = 1
                    st.success("Análisis completado")

                    # Mostrar resultado de la detección de HorizonNet
                    st.header("Resultado del análisis visual")
                    annotated_image = logic.dibujar_layout_sobre_imagen("temp_pano.jpg", room_data)
                    st.image(annotated_image, caption='Análisis de HorizonNet (Vértices, Puertas y Ventanas)', use_container_width=True)
                
            except Exception as e:
                st.error(f"Error en detección: {e}. Revisa la configuración del modelo HorizonNet.")
                st.session_state.stage = 0

# --- PASO 2: VERIFICACIÓN Y EDICIÓN DE GEOMETRÍA/OBSTÁCULOS ---
if st.session_state.stage >= 1 and st.session_state.room_data:
    st.header("2. Verificación de Geometría")
    
    # Mostrar dimensiones detectadas
    w_m = st.session_state.room_data.get('width', 0.0)
    l_m = st.session_state.room_data.get('length', 0.0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ancho (m)", f"{w_m:.2f}")
    with col2:
        st.metric("Largo (m)", f"{l_m:.2f}")
    
    # Mostrar el diagrama de planta
    st.subheader("Planta de la habitación")
    floor_plan_fig = logic.generar_diagrama_planta(st.session_state.room_data)
    st.pyplot(floor_plan_fig)
    
    # Formulario para añadir puertas y ventanas manualmente
    st.subheader("Añadir puertas y ventanas manualmente")
    
    polygon_points = st.session_state.room_data.get('polygon_points', [])
    num_walls = len(polygon_points) if polygon_points is not None else 0
    
    if num_walls > 0:
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            wall_options = [f"Pared {i+1}" for i in range(num_walls)]
            selected_wall = st.selectbox("Seleccionar Pared", wall_options, key="wall_select")
            wall_idx = int(selected_wall.split()[1]) - 1
        
        with col_b:
            obstacle_type = st.radio("Tipo", ["Puerta", "Ventana"], key="obs_type")
        
        with col_c:
            # Posición normalizada [0.0, 1.0]
            position_pct = st.number_input("Posición (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0, key="obs_pos")
        
        with col_d:
            width_m = st.number_input("Ancho (m)", min_value=0.1, max_value=5.0, value=0.9, step=0.1, key="obs_width")
        
        if st.button("Añadir elemento"):
            # Los datos de centro están normalizados
            new_obstacle = {
                'center': [position_pct / 100.0, wall_idx / max(1, num_walls)],
                'width': width_m
            }
            
            if obstacle_type == "Puerta":
                st.session_state.room_data['doors'].append(new_obstacle)
            else:
                st.session_state.room_data['windows'].append(new_obstacle)
            
            st.success(f"{obstacle_type} añadida a {selected_wall}")
            st.rerun()
    else:
        st.warning("No se detectaron paredes en el polígono.")
    
    # Editor de datos para modificar obstáculos detectados/añadidos
    st.subheader("Editar elementos (puertas y ventanas)")
    st.info("Ajusta las coordenadas de los obstáculos. Los valores X/Y están normalizados [0.0, 1.0].")

    # Preparar datos para st.data_editor
    doors_data = []
    for i, d in enumerate(st.session_state.room_data.get('doors', [])):
        center_y = d['center'][1] if len(d['center']) > 1 else 0 
        doors_data.append({"ID": f"P{i}", "Tipo": "Puerta", "Centro X (Norm.)": d['center'][0], "Centro Y (Norm.)": center_y, "Ancho (m)": d['width']})
    
    windows_data = []
    for i, w in enumerate(st.session_state.room_data.get('windows', [])):
        center_y = w['center'][1] if len(w['center']) > 1 else 0
        windows_data.append({"ID": f"V{i}", "Tipo": "Ventana", "Centro X (Norm.)": w['center'][0], "Centro Y (Norm.)": center_y, "Ancho (m)": w['width']})

    all_obstacles = doors_data + windows_data
    df_obs = pd.DataFrame(all_obstacles)
    
    col_config = {
        "Centro X (Norm.)": st.column_config.NumberColumn("Centro X (Norm.)", help="Posición horizontal normalizada [0.0, 1.0]", format="%.2f"),
        "Centro Y (Norm.)": st.column_config.NumberColumn("Centro Y (Norm.)", help="Posición vertical normalizada [0.0, 1.0]", format="%.2f"),
        "Ancho (m)": st.column_config.NumberColumn("Ancho (m)", help="Ancho del obstáculo en metros", format="%.2f"),
    }
    
    edited_df = st.data_editor(df_obs, num_rows="dynamic", use_container_width=True, column_config=col_config)

    if st.button("Confirmar geometría"):
        # Reconstruir el diccionario room_data a partir del DataFrame editado
        new_doors = []
        new_windows = []
        for index, row in edited_df.iterrows():
            obj = {'center': [row['Centro X (Norm.)'], row['Centro Y (Norm.)']], 'width': row['Ancho (m)']}
            if row['Tipo'] == 'Puerta': new_doors.append(obj)
            else: new_windows.append(obj)
            
        st.session_state.room_data['doors'] = new_doors
        st.session_state.room_data['windows'] = new_windows
        st.session_state.stage = 2
        st.rerun()

# --- PASO 3: PRESUPUESTO Y GENERACIÓN DE LAYOUT/RECOMENDACIÓN ---
if st.session_state.stage >= 2:
    st.header("3. Presupuesto y generación")
    
    presupuesto = st.number_input("Presupuesto Máximo (€)", min_value=100.0, value=1000.0, step=100.0)
    
    if st.button("Generar diseño"):
        # Convertir dimensiones de m a cm para el LayoutEngine
        w_cm = st.session_state.room_data.get('width', 0.0) * 100
        l_cm = st.session_state.room_data.get('length', 0.0) * 100
        
        if w_cm < 200 or l_cm < 200:
            st.error("Las dimensiones de la habitación son demasiado pequeñas (mínimo 2x2m) o no fueron capturadas correctamente.")
        else:
            with st.spinner("Calculando distribución óptima y seleccionando muebles..."):
                # 1. Inicializar motores
                layout_engine = logic.LayoutEngine(st.session_state.data_manager.dimensiones_promedio)
                recommender = logic.Recommender(st.session_state.muebles_df)
                
                # 2. Sugerir el pack de muebles base
                pack_sugerido = layout_engine.sugerir_pack(w_cm, l_cm)
                
                # 3. Convertir obstáculos a polígonos para el motor
                obs_layout = layout_engine.convertir_obstaculos(
                    st.session_state.room_data, 
                    w_cm, l_cm, 
                    polygon_points=st.session_state.room_data.get('polygon_points')
                )
                
                # 4. Generar el Layout
                layout_plan, constraints, log_msgs = layout_engine.generar_layout(
                    w_cm, l_cm, 
                    pack_sugerido, 
                    obs_layout,
                    polygon_points=st.session_state.room_data.get('polygon_points')
                )

                # Mostrar Log de Generación
                with st.expander("📝 Detalles de la Generación del Layout", expanded=False):
                    for msg in log_msgs:
                        if "✅" in msg: st.success(msg)
                        elif "❌" in msg: st.error(msg)
                        elif "⚠️" in msg: st.warning(msg)
                        else: st.text(msg)
                
                if not layout_plan:
                    st.error("No se pudo generar una distribución válida para este espacio (demasiado pequeño o muchos obstáculos).")
                else:
                    # 5. Recomendar productos (Knapsack para optimización de precio/estilo)
                    # Las 'constraints' se definen por los muebles que el layout PUDO colocar
                    best_combo = recommender.buscar_combinacion(constraints, presupuesto, top_n=1)
                    
                    if not best_combo:
                        st.error("No se encontraron muebles que se ajusten al presupuesto y restricciones.")
                    else:
                        st.session_state.result_layout = layout_plan
                        st.session_state.result_items = best_combo[0]['items']
                        st.session_state.result_total = best_combo[0]['precio_total']
                        st.session_state.result_score = best_combo[0]['score']
                        st.session_state.stage = 3

# --- PASO 4: RESULTADOS Y VISUALIZACIÓN FINAL ---
if st.session_state.stage == 3:
    st.divider()
    st.header("Tu salón ideal")
    
    # --- VISUALIZACIÓN 3D Interactiva (Plotly) ---
    st.subheader("Visualización 3D Interactiva")
    
    # Generar la figura 3D
    fig_plotly = logic.generar_figura_3d_plotly(
        st.session_state.result_layout, 
        st.session_state.room_data,
        st.session_state.result_items
    )
    
    # Renderizar la figura de Plotly
    st.plotly_chart(fig_plotly, use_container_width=True, theme="streamlit")
    
    st.info("💡 Usa el ratón: Clic izquierdo para rotar, rueda para zoom.")
        
    st.divider()

    # --- LISTA DE COMPRA ---
    st.subheader("Lista de Compra")
    
    # Totales y Score de Diseño
    c_tot1, c_tot2 = st.columns([2, 1])
    with c_tot1:
        st.markdown("### Total Estimado")
        st.caption(f"Score de Diseño (Estilo + Puntuación Base): {st.session_state.result_score:.2f}/1.0")
    with c_tot2:
        st.markdown(f"### {st.session_state.result_total:.2f}€")
    
    st.markdown("---")
    
    # Listado de productos
    for item in st.session_state.result_items:
        with st.container():
            c_img, c_info, c_price, c_link = st.columns([1, 2, 1, 1])
            
            url = f"https://www.ikea.com/es/es/p/{item.get('Enlace_producto', '')}-{item.get('ID', '')}"
            img_src = item.get('Imagen_principal', '')
            nombre = item['Nombre']
            tipo = item['Tipo_mueble']
            precio = float(item['Precio'])
            
            with c_img:
                if img_src:
                    st.image(img_src, width=150)
                else:
                    st.text("Sin imagen")
            
            with c_info:
                st.subheader(nombre)
                st.caption(tipo)
                st.text(item.get('Descripcion', '')[:100] + '...')
            
            with c_price:
                st.markdown(f"### {precio:.2f} €")
            
            with c_link:
                st.link_button("Ver en IKEA", url)
            
            st.divider()
    
    if st.button("Reiniciar"):
        for key in ['room_data', 'result_layout', 'result_items', 'result_total', 'result_score']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.stage = 0
        st.rerun()