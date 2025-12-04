import pandas as pd
import numpy as np
import os
import itertools
import pickle
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from tqdm import tqdm
import sys
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import pywavefront

# --- CONFIGURACIÓN DE RUTAS EXTERNAS ---
# En despliegue, HorizonNet está en un subdirectorio local
HORIZON_NET_PATH = os.path.join(os.path.dirname(__file__), 'horizonnet')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'modelos_3D')

if HORIZON_NET_PATH not in sys.path:
    sys.path.append(HORIZON_NET_PATH)

try:
    # Importar desde el paquete local horizonnet
    from horizonnet.model import HorizonNet
    from horizonnet.inference import inference
    from horizonnet.misc import utils
except ImportError as e:
    print(f"Error al importar HorizonNet desde {HORIZON_NET_PATH}")
    print(f"Detalle: {e}")
    # Definición de clase MOCK para evitar un crash de la aplicación
    class RoomLayoutDetector:
        def __init__(self, model_path): 
            print("RoomLayoutDetector en modo MOCK (Error de importación)")
        def detect_layout(self, img_path): return None

# ==========================================
# CLASE DE DETECCIÓN DE LAYOUT (HorizonNet)
# ==========================================
class RoomLayoutDetector:
    """
    Clase para la detección de geometría de habitación (floor/ceiling boundaries)
    y corners a partir de una imagen panorámica 360 usando HorizonNet.
    """
    def __init__(self, model_path):
        # Utiliza MPS (Metal Performance Shaders) si está disponible en M4
        self.device = torch.device('cpu') 
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        
        print(f"Cargando modelo desde: {model_path} en {self.device}")
        try:
            self.net = utils.load_trained_model(HorizonNet, model_path).to(self.device)
            self.net.eval()
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.net = None

    def detect_layout(self, img_path):
        """Ejecuta la inferencia de HorizonNet y escala los resultados a metros."""
        if self.net is None: return None
        
        # 1. Preprocesar imagen
        try:
            img_pil = Image.open(img_path)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])
            
            # 2. Inferencia
            with torch.no_grad():
                cor_id, z0, z1, vis_out = inference(
                    net=self.net, x=x, device=self.device,
                    flip=False, rotate=[], visualize=False,
                    force_cuboid=False, force_raw=False,
                    min_v=None, r=0.05
                )
            
            # 3. Procesar resultados: Obtener polígono del suelo
            uv = [[float(u), float(v)] for u, v in cor_id]
            floor_points = self._uv_to_floor_polygon(uv, z0, z1)
            
            # 4. Calcular dimensiones y escalar a metros
            min_x, min_y = floor_points.min(axis=0)
            max_x, max_y = floor_points.max(axis=0)
            ancho_bbox = max_x - min_x
            largo_bbox = max_y - min_y
            altura_unidades = abs(z1 - z0)
            
            # Factor de escala: Se asume altura de cámara de 1.6m (z0)
            ALTURA_CAMARA = 1.6
            factor_escala = ALTURA_CAMARA / z0
            
            ancho_m = ancho_bbox * factor_escala
            largo_m = largo_bbox * factor_escala
            altura_m = altura_unidades * factor_escala
            
            # Escalar puntos para visualización 3D y normalizar para 2D (planta)
            floor_points_scaled = floor_points * factor_escala
            # Para la planta 2D, normalizamos para que empiece en (0,0) y esté en metros
            floor_points_norm = (floor_points_scaled - [floor_points_scaled[:,0].min(), floor_points_scaled[:,1].min()])
            
            # Obtener datos raw para la visualización de la detección
            x_tensor = x.to(self.device)
            y_bon_, y_cor_ = self.net(x_tensor)
            y_bon_ = y_bon_.cpu().detach().numpy()[0]
            y_cor_ = torch.sigmoid(y_cor_).cpu().detach().numpy()[0, 0]
            
            return {
                'width': ancho_m,
                'length': largo_m,
                'height': altura_m,
                'doors': [], 'windows': [], # Estos deberían ser inferidos en tu versión completa
                'y_bon': y_bon_, 'y_cor': y_cor_,
                'polygon_points': floor_points_norm.tolist(), 
                'polygon_points_raw': floor_points_scaled.tolist(), 
                'factor_escala': factor_escala
            }
            
        except Exception as e:
            print(f"Error en inferencia: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _uv_to_floor_polygon(self, uv, z0, z1):
        """Convierte los puntos uv (coordenadas de la imagen) en coordenadas (x,y) del plano del suelo."""
        floor_points = []
        for i in range(0, len(uv), 2):
            u = uv[i][0]
            v_floor = uv[i+1][1]
            lon = (u - 0.5) * 2 * np.pi
            lat = (0.5 - v_floor) * np.pi
            # Fórmula de proyección esférica
            r = abs(z0 / np.tan(lat)) if np.tan(lat) != 0 else 0
            x = r * np.cos(lon)
            y = r * np.sin(lon)
            floor_points.append([x, y])
        return np.array(floor_points)

# ==========================================
# 1. MODELO DE ESTILO (Encoder basado en BERT)
# ==========================================
class StyleEncoder(nn.Module):
    """
    Encoder de estilo basado en BERT para generar un vector de embedding
    a partir de la descripción de un mueble.
    """
    def __init__(self, n_dims=128):
        super(StyleEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.fc = nn.Linear(self.bert.config.hidden_size, n_dims)
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = bert_output[1] # Vector [CLS]
        vector_estilo = self.fc(pooler_output)
        return vector_estilo

# ==========================================
# 2. GESTOR DE DATOS Y CARGA
# ==========================================
class DataManager:
    """Gestiona la carga de la base de datos, la vectorización y el cacheo."""
    def __init__(self, csv_path, vectors_cache_path, bert_model_path):
        self.csv_path = csv_path
        self.cache_path = vectors_cache_path
        self.bert_model_path = bert_model_path
        self.df_muebles = None
        self.dimensiones_promedio = {}
        
        # Tipos de muebles relevantes para el layout del salón
        self.muebles_a_usar = [
            'Sofás', 'Sillones', 'Muebles de salón',
            'Mesas bajas de salón, de centro y auxiliares',
            'Estanterías y librerías'
        ]

    def cargar_datos(self):
        """Carga los vectores desde caché o los genera usando el StyleEncoder."""
        if os.path.exists(self.cache_path):
            print(f"Cargando caché de: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.df_muebles = pickle.load(f)
        else:
            print("Generando vectores (esto puede tardar)...")
            self._generar_vectores()
        
        # Filtrar sofás excesivamente grandes (> 300cm) que rompen el layout en habitaciones normales
        if self.df_muebles is not None:
            self.df_muebles = self.df_muebles[~((self.df_muebles['Tipo_mueble'] == 'Sofás') & (self.df_muebles['Ancho'] > 300))]
        
        # Pre-procesar columnas de dimensiones para asegurar que son numéricas y válidas
        if self.df_muebles is not None:
            cols_dim = ['Ancho', 'Largo', 'Altura']
            for col in cols_dim:
                self.df_muebles[col] = pd.to_numeric(self.df_muebles[col], errors='coerce')
                self.df_muebles.loc[self.df_muebles[col] <= 0, col] = np.nan

        # Calcular dimensiones promedio por categoría
        if self.df_muebles is not None:
            avg_df = self.df_muebles.groupby('Tipo_mueble')[['Ancho', 'Largo']].mean()
            self.dimensiones_promedio = avg_df.to_dict('index')
            
            # Normalizar claves a minúsculas para evitar errores de key y aplicar lógica de negocio
            processed_dims = {}
            for k, v in self.dimensiones_promedio.items():
                ancho = round(v.get('Ancho', 100.0), 1)
                largo = round(v.get('Largo', 50.0), 1)

                # Correcciones de lógica de negocio o datos faltantes (en cm)
                if k == 'Sofás':
                    if largo < 50: largo = 90.0
                if k == 'Sillones':
                    if largo < 40: largo = ancho if ancho >= 40 else 70.0
                
                processed_dims[k] = {'ancho': ancho, 'largo': largo}
            self.dimensiones_promedio = processed_dims

        # Imputar valores faltantes o inválidos en el DataFrame con los promedios calculados
        if self.df_muebles is not None:
            for tipo, dims in self.dimensiones_promedio.items():
                mask_tipo = self.df_muebles['Tipo_mueble'] == tipo
                
                # Imputar Ancho
                mask_invalid_w = mask_tipo & (self.df_muebles['Ancho'].isna())
                if mask_invalid_w.any():
                    self.df_muebles.loc[mask_invalid_w, 'Ancho'] = dims['ancho']
                    
                # Imputar Largo
                mask_invalid_l = mask_tipo & (self.df_muebles['Largo'].isna() | (self.df_muebles['Largo'] <= 0))
                if mask_invalid_l.any():
                    self.df_muebles.loc[mask_invalid_l, 'Largo'] = dims['largo']
                    
                # Imputar Altura (opcional, pero bueno para consistencia)
                # mask_invalid_h = mask_tipo & (self.df_muebles['Altura'].isna() | (self.df_muebles['Altura'] <= 0))
                # if mask_invalid_h.any():
                #    self.df_muebles.loc[mask_invalid_h, 'Altura'] = 60 # Valor por defecto seguro

        return self.df_muebles

    def _generar_vectores(self, n_dims=128):
        """Procesa el CSV y genera el vector de estilo para cada mueble."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No se encuentra el CSV en {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        df_filtrado = df[df['Tipo_mueble'].isin(self.muebles_a_usar)].copy()
        
        # Carga modelo BERT
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StyleEncoder(n_dims=n_dims).to(device)
        
        if os.path.exists(self.bert_model_path):
            model.load_state_dict(torch.load(self.bert_model_path, map_location=device))
        else:
            print("ADVERTENCIA: No se encontraron pesos del modelo BERT, usando aleatorios.")

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model.eval()
        
        df_filtrado['text_data'] = df_filtrado['Nombre'].fillna('') + ' ' + df_filtrado['Descripcion'].fillna('')
        generated_vectors = []
        
        with torch.no_grad():
            for text in tqdm(df_filtrado['text_data'], desc="Vectorizando"):
                tokens = tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                vector = model(input_ids, attention_mask)
                generated_vectors.append(vector.cpu().detach().numpy().flatten())
        
        df_filtrado['vector_estilo'] = generated_vectors
        df_filtrado['vector_estilo'] = df_filtrado['vector_estilo'].apply(lambda x: np.array(x))
        
        self.df_muebles = df_filtrado
        # Guardar caché
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.df_muebles, f)
        except Exception as e:
            print(f"Advertencia: No se pudo guardar la caché de vectores: {e}")


    def _calcular_dimensiones_promedio(self):
        """Calcula el ancho y largo promedio por Tipo_mueble."""
        if self.df_muebles is None: return
        
        cols = ['Ancho', 'Largo', 'Altura']
        for col in cols:
            self.df_muebles[col] = pd.to_numeric(self.df_muebles[col], errors='coerce')
            # Reemplazar valores <= 0 con NaN para no afectar el promedio
            self.df_muebles.loc[self.df_muebles[col] <= 0, col] = np.nan
            
        avg_df = self.df_muebles.groupby('Tipo_mueble')[cols].mean()
        
        self.dimensiones_promedio = {}
        for tipo, row in avg_df.iterrows():
            self.dimensiones_promedio[tipo] = {
                'nombre': tipo,
                'ancho': round(row['Ancho'], 1) if not pd.isna(row['Ancho']) else 100.0,
                'largo': round(row['Largo'], 1) if not pd.isna(row['Largo']) else 50.0
            }
        
        # Correcciones de lógica de negocio o datos faltantes (en cm)
        if 'Sofás' in self.dimensiones_promedio:
            if self.dimensiones_promedio['Sofás']['largo'] < 50: self.dimensiones_promedio['Sofás']['largo'] = 90.0
        if 'Sillones' in self.dimensiones_promedio:
             if self.dimensiones_promedio['Sillones']['largo'] < 40:
                 self.dimensiones_promedio['Sillones']['largo'] = self.dimensiones_promedio['Sillones']['ancho'] if self.dimensiones_promedio['Sillones']['ancho'] >= 40 else 70.0

# ==========================================
# 3. MOTOR DE LAYOUT (Lógica Espacial y de Colocación)
# ==========================================
class LayoutEngine:
    """Implementa la lógica de colocación de muebles, restricciones espaciales y colisiones."""
    def __init__(self, dimensiones_promedio):
        self.dim_promedio = dimensiones_promedio
        self.config = {
            'pasillo_minimo': 90, # cm
            'margen_pared': 10, # cm
            'margen_obstaculo': 20, # cm
            'distancia_tv_sofa': 280 # cm (distancia ideal)
        }

    def sugerir_pack(self, ancho_cm, largo_cm):
        """Sugiere un conjunto de muebles base según el área de la habitación."""
        area = (ancho_cm * largo_cm) / 10000.0
        # Definición de tipos de mueble base
        pack = [{'tipo': 'Muebles de salón'}, {'tipo': 'Mesas bajas de salón, de centro y auxiliares'}, {'tipo': 'Sofás'}]
        # Añadir complejidad según el tamaño de la sala
        if area > 16.0: pack.insert(0, {'tipo': 'Sillones'})
        if area > 22.0: pack.insert(0, {'tipo': 'Estanterías y librerías'})
        return pack

    def convertir_obstaculos(self, obstaculos_dict, ancho_hab, largo_hab, polygon_points):
        """Convierte los obstáculos detectados (puertas/ventanas) en polígonos Shapely con margen de seguridad."""
        obs_polys = []
        num_walls = len(polygon_points)
        
        todos_obs = []
        for d in obstaculos_dict.get('doors', []): d['type'] = 'door'; todos_obs.append(d)
        for w in obstaculos_dict.get('windows', []): w['type'] = 'window'; todos_obs.append(w)

        for obs in todos_obs:
            # Lógica para mapear la posición normalizada (0-1) a un polígono en CM
            w_idx = int(round(obs['center'][1] * num_walls)) % num_walls
            p1 = np.array(polygon_points[w_idx]) * 100
            p2 = np.array(polygon_points[(w_idx + 1) % num_walls]) * 100
            
            vec_pared = p2 - p1
            len_pared = np.linalg.norm(vec_pared)
            unit_pared = vec_pared / len_pared
            
            # Centro del obstáculo a lo largo de la pared
            center_pt = p1 + vec_pared * obs['center'][0]
            width = obs['width'] * 100
            depth = 100 if obs['type'] == 'door' else 30 # Profundidad de la zona de exclusión
            
            # Vector normal a la pared
            normal = np.array([-unit_pared[1], unit_pared[0]])
            
            # Definir las esquinas del polígono de exclusión (con margen)
            c1 = center_pt - unit_pared * (width/2) - normal * (depth/2)
            c2 = center_pt + unit_pared * (width/2) - normal * (depth/2)
            c3 = center_pt + unit_pared * (width/2) + normal * (depth/2)
            c4 = center_pt - unit_pared * (width/2) + normal * (depth/2)
            
            poly = Polygon([c1, c2, c3, c4])
            obs_polys.append({'poly': poly, 'tipo': obs['type']})
            
        return obs_polys

    def _get_poly_from_rect(self, x, y, w, l, angle_rad):
        """Genera un Polygon de Shapely rotado a partir del centro (x, y), dimensiones (w, l) y ángulo."""
        cx, cy = x, y
        dx = w / 2
        dy = l / 2
        
        corners = [
            (dx, dy), (-dx, dy), (-dx, -dy), (dx, -dy)
        ]
        
        new_corners = []
        c_cos = np.cos(angle_rad)
        c_sin = np.sin(angle_rad)
        
        for px, py in corners:
            nx = px * c_cos - py * c_sin + cx
            ny = px * c_sin + py * c_cos + cy
            new_corners.append((nx, ny))
            
        return Polygon(new_corners)

    def _check_collision(self, candidate_poly, room_poly, placed_items, obstacles):
        """Verifica si el polígono candidato colisiona con la pared, ítems colocados u obstáculos."""
        # 1. Dentro de la habitación (con margen de pared)
        buffered_room = room_poly.buffer(-self.config['margen_pared'])
        if not buffered_room.contains(candidate_poly):
            # print(f"DEBUG: Colisión con PARED. Poly fuera del buffer.")
            # print(f"  Room Buffer Bounds: {buffered_room.bounds}")
            # print(f"  Candidate Bounds: {candidate_poly.bounds}")
            return False
        
        # 2. Colisión con items ya colocados
        for item in placed_items:
            # Usar un buffer de 10cm para asegurar un pequeño espacio entre muebles
            if candidate_poly.buffer(10).intersects(item['poly']): 
                # print(f"DEBUG: Colisión con ITEM {item['tipo']}.")
                return False
                
        # 3. Colisión con obstáculos (con margen)
        for obs in obstacles:
            if candidate_poly.intersects(obs['poly']):
                # print(f"DEBUG: Colisión con OBSTÁCULO {obs['tipo']}.")
                return False
                
        return True

    def _scan_wall(self, p1, p2, item_dim, room_poly, placed, obstacles, align_dist=None, align_target=None):
        """Barre una pared específica buscando la primera ubicación válida para un mueble."""
        vec = p2 - p1
        wall_len = np.linalg.norm(vec)
        unit_vec = vec / wall_len
        
        normal = np.array([-unit_vec[1], unit_vec[0]])
        
        # Verificar que la normal apunta al interior de la habitación
        centroid = np.array(room_poly.centroid.coords[0])
        mid_wall = (p1 + p2) / 2
        if np.dot(centroid - mid_wall, normal) < 0:
            normal = -normal 

        w_item = item_dim['ancho']
        l_item = item_dim['largo']
        angle = np.arctan2(unit_vec[1], unit_vec[0])
        
        # Distancia del centro del mueble a la pared (para pegar a la pared)
        dist_from_wall = self.config['margen_pared'] + l_item/2 
        if align_dist: dist_from_wall = align_dist
        
        step = 20 # cm
        margin_side = self.config['margen_pared'] + w_item/2
        
        range_start = margin_side
        range_end = wall_len - margin_side
        
        if range_end < range_start: return []
        
        candidates = list(np.arange(range_start, range_end, step))
        # Asegurar que el centro de la pared está en los candidatos
        mid_dist = wall_len / 2
        if range_start <= mid_dist <= range_end:
            candidates.append(mid_dist)
        # Ordenar para probar primero el centro? No necesariamente, pero ayuda.
        candidates = sorted(list(set(candidates)))
        
        if align_target is not None:
            # Lógica para alinear con un objeto existente
            v_target = np.array([align_target['x'], align_target['y']]) - p1
            proj_dist = np.dot(v_target, unit_vec)
            candidates = [proj_dist] 
            
        valid_candidates = []
        for dist_along in candidates:
            center = p1 + unit_vec * dist_along + normal * dist_from_wall
            cand_poly = self._get_poly_from_rect(center[0], center[1], w_item, l_item, angle)
            
            if self._check_collision(cand_poly, room_poly, placed, obstacles):
                valid_candidates.append({
                    'x': center[0], 'y': center[1], 
                    'ancho': w_item, 'largo': l_item,
                    'angle': angle, 
                    'tipo': item_dim['nombre'],
                    'poly': cand_poly
                })
        return valid_candidates

    def generar_layout(self, ancho_hab, largo_hab, pack_sugerido, obs_layout, polygon_points=None):
        """Genera el layout buscando una configuración TV-Sofá válida y añadiendo la mesa de centro."""
        # setup básico
        layout = []
        constraints = []
        log = []
        final_layout_objs = []
        
        if not polygon_points:
            # Usar un rectángulo simple si no hay polígono
            polygon_points = [[0,0], [ancho_hab/100, 0], [ancho_hab/100, largo_hab/100], [0, largo_hab/100]]

        poly_pts_cm = np.array(polygon_points) * 100
        room_poly = Polygon(poly_pts_cm)
        num_walls = len(poly_pts_cm)
        
        # Dimensiones promedio de los dos muebles clave
        d_tv = self.dim_promedio.get('Muebles de salón', {'ancho': 120, 'largo': 40})
        d_tv['nombre'] = 'Muebles de salón'
        d_sofa = self.dim_promedio.get('Sofás', {'ancho': 200, 'largo': 90})
        d_sofa['nombre'] = 'Sofás'
        
        # Estrategia de reintento con tamaños reducidos
        attempt_configs = [
            {'scale': 1.0, 'desc': 'Standard'},
            {'scale': 0.8, 'desc': 'Compact'}
        ]
        
        best_overall_score = -1
        best_overall_layout = []
        
        for config in attempt_configs:
            scale = config['scale']
            # Aplicar escala a las dimensiones temporales para este intento
            current_d_tv = d_tv.copy()
            current_d_tv['ancho'] *= scale
            current_d_tv['largo'] *= scale
            
            current_d_sofa = d_sofa.copy()
            current_d_sofa['ancho'] *= scale
            current_d_sofa['largo'] *= scale
            
            log.append(f"🔄 Intentando generación con modo {config['desc']} (Escala {scale})")
            
            best_score = -1
            best_layout = []
            
            # Iterar sobre todas las paredes para encontrar la mejor ubicación para el binomio TV-Sofá
            for i in range(num_walls):
                p1 = poly_pts_cm[i]
                p2 = poly_pts_cm[(i+1)%num_walls]
                
                # 1. Intentar poner Mueble de Salón (TV) en la pared
                # Ahora devuelve una lista de candidatos
                tv_candidates = self._scan_wall(p1, p2, current_d_tv, room_poly, [], obs_layout)
                
                for tv_pos in tv_candidates:
                    current_layout = [tv_pos]
                    
                    # Calcular la normal del TV
                    vec_pared = p2 - p1
                    unit_vec_pared = vec_pared / np.linalg.norm(vec_pared)
                    normal_base = np.array([-unit_vec_pared[1], unit_vec_pared[0]])
                    
                    centroid = np.array(room_poly.centroid.coords[0])
                    mid_wall = (p1 + p2) / 2
                    
                    # Asegurar que la normal apunta al centro (adentro)
                    if np.linalg.norm((mid_wall + normal_base) - centroid) > np.linalg.norm((mid_wall - normal_base) - centroid):
                        tv_normal = -normal_base
                    else:
                        tv_normal = normal_base
    
                    # 2. Calcular dónde estaría el sofá y proyectar un rayo
                    tv_center_pt = Point(tv_pos['x'], tv_pos['y'])
                    ray_end_np = np.array([tv_pos['x'], tv_pos['y']]) + tv_normal * max(ancho_hab, largo_hab) * 100 
                    ray = LineString([tv_center_pt, (ray_end_np[0], ray_end_np[1])])
                    
                    intersection = ray.intersection(room_poly.boundary)
                    
                    distancia_pared_opuesta = 9999
                    
                    if not intersection.is_empty:
                        if intersection.geom_type == 'Point':
                            d = tv_center_pt.distance(intersection)
                            if d > 50: distancia_pared_opuesta = d
                        elif intersection.geom_type == 'MultiPoint':
                            for pt in intersection.geoms:
                                d = tv_center_pt.distance(pt)
                                if d > 50 and d < distancia_pared_opuesta:
                                    distancia_pared_opuesta = d
    
                    dist_ideal = self.config['distancia_tv_sofa']
                    fondo_sofa = current_d_sofa['largo']
                    
                    # Calcular el espacio libre entre la parte trasera del sofá y la pared
                    espacio_detras = distancia_pared_opuesta - (tv_pos['largo']/2 + dist_ideal + fondo_sofa/2)
                    
                    dist_sofa_desde_tv = dist_ideal
                    
                    if distancia_pared_opuesta < (tv_pos['largo']/2 + dist_ideal + fondo_sofa + self.config['margen_pared']):
                        # Si la habitación es demasiado pequeña, pega el sofá a la pared trasera
                        # Corrección: No restar tv_pos['largo']/2 porque la distancia es desde el centro de la TV
                        dist_sofa_desde_tv = distancia_pared_opuesta - fondo_sofa/2 - self.config['margen_pared']
                        log_msg = f"✅ Sofá ajustado a pared (Espacio insuficiente para ideal)"
                    elif espacio_detras < 100: # Aumentado a 100cm. Si sobra menos de 1m, pégalo atrás.
                        # Corrección: Añadir 5cm extra de margen para asegurar que el polígono esté DENTRO del buffer de la habitación
                        dist_sofa_desde_tv = distancia_pared_opuesta - fondo_sofa/2 - self.config['margen_pared'] - 5.0
                        log_msg = f"✅ Sofá ajustado a pared (Evitar espacio muerto de {espacio_detras:.0f}cm)"
                    else:
                        # Posicionamiento ideal frente a TV
                        dist_sofa_desde_tv = dist_ideal + tv_pos['largo']/2 + fondo_sofa/2
                        log_msg = "✅ Sofá en isla"
    
    
                    # Intentar colocar el sofá con "nudge" (empujoncitos) SOLO frontal para mantener alineación
                    sofa_placed = False
                    # Nudge frontal: 0 a 30cm
                    for nudge in [0, 5, 10, 15, 20, 25, 30]:
                        current_dist = dist_sofa_desde_tv - nudge
                        
                        sofa_center_np = np.array([tv_pos['x'], tv_pos['y']]) + tv_normal * current_dist
                        sofa_poly = self._get_poly_from_rect(sofa_center_np[0], sofa_center_np[1], current_d_sofa['ancho'], current_d_sofa['largo'], tv_pos['angle'])
                        
                        sofa_cand = {
                            'x': sofa_center_np[0], 'y': sofa_center_np[1],
                            'ancho': current_d_sofa['ancho'], 'largo': current_d_sofa['largo'],
                            'angle': tv_pos['angle'], 'tipo': 'Sofás', 'poly': sofa_poly
                        }
                        
                        if self._check_collision(sofa_poly, room_poly, [tv_pos], obs_layout):
                            current_layout.append(sofa_cand)
                            score = 100 - nudge 
                            if score > best_score:
                                best_score = score
                                best_layout = current_layout
                                log.append(f"{log_msg} (Nudge={nudge}cm)")
                                sofa_placed = True
                            break 
                    
                    if sofa_placed:
                        break # Romper el loop de TV candidates si ya encontramos un layout válido
                    else:
                        log.append(f"❌ Sofá colisiona tras intentos. DistBase={dist_sofa_desde_tv:.1f}")
                        # print(f"DEBUG: Fallo Sofá. Dist={dist_sofa_desde_tv:.1f}. Wall={i}")
            
            if best_layout:
                best_overall_layout = best_layout
                best_overall_score = best_score
                break # Si encontramos un layout válido con esta escala, nos quedamos con él
        
        best_layout = best_overall_layout # Restaurar para el resto del código

        if best_layout:
            # 2. Convertir layout a formato final y generar restricciones
            for item in best_layout:
                final_layout_objs.append({
                    'x': item['x'], 'y': item['y'], 
                    'ancho': item['ancho'], 'largo': item['largo'],
                    'angle': item['angle'], 'tipo': item['tipo']
                })
                # Definir la restricción dimensional
                constraints.append({'tipo': item['tipo'], 'max_ancho': item['ancho']*1.2, 'max_largo': item['largo']*1.2})
            
            # 3. Colocación condicional de Mesa de Centro
            tv = best_layout[0]
            sofa = best_layout[1]

            # Calcular la distancia libre entre TV y Sofá
            dist_centros = np.linalg.norm(np.array([tv['x'], tv['y']]) - np.array([sofa['x'], sofa['y']]))
            depth_tv = tv['largo']
            depth_sofa = sofa['largo']
            
            espacio_libre = dist_centros - (depth_tv / 2) - (depth_sofa / 2)

            profundidad_mesa = self.dim_promedio.get('Mesas bajas de salón, de centro y auxiliares', {'largo': 60})['largo']
            pasillo_minimo = 60 # cm para circular alrededor

            if espacio_libre >= (profundidad_mesa + pasillo_minimo):
                mid_x = (tv['x'] + sofa['x']) / 2
                mid_y = (tv['y'] + sofa['y']) / 2
                
                final_layout_objs.append({
                    'x': mid_x, 'y': mid_y, 
                    'ancho': 100, 'largo': profundidad_mesa, # Usamos ancho y largo genérico
                    'angle': tv['angle'],
                    'tipo': 'Mesas bajas de salón, de centro y auxiliares'
                })
                constraints.append({'tipo': 'Mesas bajas de salón, de centro y auxiliares'})
            else:
                log.append(f"⚠️ Mesa de centro omitida: Espacio libre ({espacio_libre:.0f}cm) insuficiente para mesa + paso.")

        else:
            log.append("No se encontró distribución válida TV-Sofá. Distribución fallida.")
            
        return final_layout_objs, constraints, log

# ==========================================
# 4. MOTOR DE RECOMENDACIÓN (Knapsack/Estilo)
# ==========================================
class Recommender:
    """Implementa el algoritmo de selección de productos para maximizar el Score/Coherencia dentro del presupuesto."""
    def __init__(self, df_data):
        self.df = df_data

    def _coherencia(self, vectores):
        """Calcula la coherencia de estilo promedio (Similitud Coseno) entre los vectores de estilo de los muebles."""
        if len(vectores) < 2: return 1.0
        mat = cosine_similarity(np.array(vectores))
        # Promedio de la similitud entre todos los pares (triángulo superior)
        indices = np.triu_indices_from(mat, k=1)
        return float(np.mean(mat[indices])) if indices[0].size > 0 else 1.0

    def buscar_combinacion(self, constraints, presupuesto, top_n=1):
        """
        Algoritmo Knapsack de fuerza bruta optimizada. 
        Busca la mejor combinación de productos que cumpla restricciones dimensionales y presupuestarias, 
        maximizando el score total (Score Base + Coherencia de Estilo).
        """
        listas_candidatos = []
        
        for const in constraints:
            tipo = const['tipo']
            max_w = const.get('max_ancho', 9999)
            max_l = const.get('max_largo', 9999)
            
            # Filtro dimensional 
            pool = self.df[self.df['Tipo_mueble'] == tipo]
            # Permite rotación (Ancho <= Max_W y Largo <= Max_L) o (Ancho <= Max_L y Largo <= Max_W)
            fits = pool[((pool['Ancho'] <= max_w) & (pool['Largo'] <= max_l)) | 
                        ((pool['Ancho'] <= max_l) & (pool['Largo'] <= max_w))]
            
            if fits.empty: fits = pool # Si no hay que cumplen, toma la lista completa
            
            # ESTRATEGIA HÍBRIDA:
            # Seleccionar los top 5 por Score (Calidad) Y los top 5 más baratos (Presupuesto)
            # para asegurar que tenemos opciones viables si el presupuesto es bajo.
            
            top_score = fits.sort_values('Score', ascending=False).head(5)
            top_cheap = fits.sort_values('Precio', ascending=True).head(5)
            top_expensive = fits.sort_values('Precio', ascending=False).head(5)
            
            # Combinar y eliminar duplicados (usando ID para evitar error con numpy arrays)
            candidates_df = pd.concat([top_score, top_cheap, top_expensive]).drop_duplicates(subset='ID')
            
            candidatos = candidates_df.to_dict('records')
            listas_candidatos.append(candidatos)

        if not listas_candidatos: return []

        validas = []
        for combo in itertools.product(*listas_candidatos):
            precio = sum(x['Precio'] for x in combo)
            if precio <= presupuesto:
                score_base = np.mean([x['Score'] for x in combo])
                vectores = [x['vector_estilo'] for x in combo]
                coherencia = self._coherencia(vectores)
                
                # Score Final: 
                # 40% Score Base (Calidad/Popularidad) 
                # 40% Coherencia (Estilo)
                # 20% Aprovechamiento de Presupuesto (Reward por usar el presupuesto disponible)
                budget_utilization = precio / presupuesto
                
                final_score = 0.4 * score_base + 0.4 * coherencia + 0.2 * budget_utilization
                
                validas.append({
                    'items': combo,
                    'precio_total': precio,
                    'score': final_score
                })
        
        validas.sort(key=lambda x: x['score'], reverse=True)
        return validas[:top_n]

# ==========================================
# 5. VISUALIZADORES (PLANTAS 2D y 3D con OBJs)
# ==========================================

def get_segment_properties(p1, p2):
    """Calcula longitud, punto central y ángulo de un segmento 2D."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx**2 + dy**2)
    midpoint_x = (p1[0] + p2[0]) / 2
    midpoint_y = (p1[1] + p2[1]) / 2
    angle = np.arctan2(dy, dx) 
    return length, midpoint_x, midpoint_y, angle


def read_kenney_obj(obj_path):
    """
    Lee archivos OBJ simples (como los de Kenney) extrayendo solo vértices (v) y caras (f).
    Retorna (vertices_list, faces_list).
    """
    vertices = []
    faces = []
    
    # --- DEBUG: Comprobar lectura de archivo ---
    print(f"\n--- DEBUG: Leyendo manualmente: {obj_path} ---")
    v_count = 0
    f_count = 0
    
    try:
        # Nota: El error podría ser la codificación. Usamos 'utf-8' o 'latin-1'
        with open(obj_path, 'r', encoding='latin-1') as f: 
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                prefix = parts[0]
                
                if prefix == 'v':
                    # Vértices: 'v x y z'
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        v_count += 1
                    except ValueError:
                        print(f"DEBUG: Vértice inválido en {obj_name}: {line.strip()}")
                
                elif prefix == 'f':
                    # Caras: 'f v/vt/vn v/vt/vn v/vt/vn ...'
                    try:
                        face_indices = []
                        for part in parts[1:]:
                            v_index = int(part.split('/')[0])
                            face_indices.append(v_index - 1) 
                        faces.append(face_indices)
                        f_count += 1
                    except ValueError:
                        print(f"DEBUG: Cara inválida en {obj_name}: {line.strip()}")
                        
    except FileNotFoundError:
        print(f"DEBUG: Archivo no encontrado en la ruta de {obj_path}")
        return [], []
    except Exception as e:
        print(f"DEBUG: Error inesperado de E/S: {e}")
        return [], []

    # --- DEBUG: Reporte final de la lectura ---
    print(f"DEBUG RESULTADO: Vértices leídos (v): {v_count}")
    print(f"DEBUG RESULTADO: Caras leídas (f): {f_count}")
    print(f"---------------------------------------------")
        
    return vertices, faces


def load_and_transform_mesh(obj_name, w, l, h, cx, cy, angle, base_z=0, rotation_offset=0):
    """
    Carga un modelo .obj usando el parser manual, lo escala de forma NO UNIFORME 
    para encajar en (w, l, h), y lo rota/traslada a la posición.
    """
    obj_path = os.path.join(MODEL_DIR, obj_name)
    
    if not os.path.exists(obj_path):
        print(f"!!! ERROR MODELO 3D: '{obj_name}' no encontrado en {MODEL_DIR}")
        return None, None, None, None, None, None
        
    # --- 1. CARGA USANDO PARSER MANUAL ---
    vertices, faces_indices_list = read_kenney_obj(obj_path)
    
    if not vertices:
        print(f"!!! ERROR MODELO 3D: Modelo '{obj_name}' sin datos 3D después del parseo.")
        return None, None, None, None, None, None
        
    vertices_np = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    
    # --- 2. PREPARAR CARAS PARA PLOTLY ---
    i_faces, j_faces, k_faces = [], [], []
    
    for face in faces_indices_list:
        if len(face) == 3:
            i_faces.append(face[0])
            j_faces.append(face[1])
            k_faces.append(face[2])
        elif len(face) == 4:
            i_faces.extend([face[0], face[0]])
            j_faces.extend([face[1], face[2]])
            k_faces.extend([face[2], face[3]])
            
    if not i_faces:
        print(f"!!! ERROR MODELO 3D: Modelo '{obj_name}' sin caras válidas para Plotly.")
        return None, None, None, None, None, None

    # --- 3. TRANSFORMAR VÉRTICES (ESCALADO NO UNIFORME) ---
    
    # 3.0. CORRECCIÓN CRÍTICA DE ORIENTACIÓN (Rotación 90° sobre X)
    # Rota el modelo de Kenney (que suele tener Y=Arriba, Z=Profundidad) a
    # la convención de tu sistema (Z=Arriba, Y=Profundidad).
    
    # Matriz de rotación 90° sobre eje X (Rotar Y a Z, Z a -Y)
    R_X = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    vertices_np = vertices_np @ R_X.T # Aplicamos la rotación BASE

    # 3.1. Encontrar el Bounding Box (para escalado) - ¡Usando los vértices rotados!
    min_x, max_x = vertices_np[:, 0].min(), vertices_np[:, 0].max()
    min_y, max_y = vertices_np[:, 1].min(), vertices_np[:, 1].max()
    min_z, max_z = vertices_np[:, 2].min(), vertices_np[:, 2].max()
    
    bbox_w = max_x - min_x
    bbox_l = max_y - min_y
    bbox_h = max_z - min_z
    
    # 3.2. Calcular Factor de Escala NO UNIFORME
    # Evitar división por cero
    scale_x = w / bbox_w if bbox_w > 0 else 1.0
    scale_y = l / bbox_l if bbox_l > 0 else 1.0
    scale_z = h / bbox_h if bbox_h > 0 else 1.0

    # 3.3. Trasladar al origen (centrar en XY y base en Z=0)
    center_x_base = (min_x + max_x) / 2
    center_y_base = (min_y + max_y) / 2
    
    # Trasladar el centro y mover la base al plano Z=0
    transformed_v = vertices_np - np.array([center_x_base, center_y_base, min_z]) 

    # 3.4. Aplicar Escala No Uniforme
    transformed_v[:, 0] *= scale_x
    transformed_v[:, 1] *= scale_y
    transformed_v[:, 2] *= scale_z
    
    # --- 4. APLICAR ROTACIÓN Y TRASLACIÓN FINAL ---
    
    # Aplicar Rotación del Layout (en el eje Z) + Offset
    final_angle = angle + rotation_offset
    c_cos = np.cos(final_angle)
    c_sin = np.sin(final_angle)
    rot_matrix = np.array([[c_cos, -c_sin], [c_sin, c_cos]])
    transformed_v[:, :2] = transformed_v[:, :2] @ rot_matrix.T

    # Aplicar Traslación Final
    transformed_v[:, 0] += cx
    transformed_v[:, 1] += cy
    transformed_v[:, 2] += base_z

    x_coords, y_coords, z_coords = transformed_v[:, 0], transformed_v[:, 1], transformed_v[:, 2]
    
    return x_coords, y_coords, z_coords, i_faces, j_faces, k_faces

def dibujar_layout_sobre_imagen(img_path, room_data):
    """
    Dibuja las predicciones de HorizonNet (líneas de floor/ceiling y corners)
    sobre la imagen panorámica para visualización de la detección.
    """
    try:
        # Cargar imagen y redimensionar a 1024x512
        img = Image.open(img_path).convert("RGB")
        img = img.resize((1024, 512), Image.LANCZOS)
        img_array = np.array(img)
        
        # Verificar que tenemos los datos raw del modelo
        if 'y_bon' not in room_data or 'y_cor' not in room_data:
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Sin datos de visualización raw", fill=(255, 0, 0))
            return img
        
        y_bon = room_data['y_bon']  # [2, 1024] en radianes
        y_cor = room_data['y_cor']  # [1024] probabilidades [0,1]
        
        # Convertir boundary de radianes a píxeles 
        y_bon_pix = ((y_bon / np.pi + 0.5) * 512).round().astype(int)
        y_bon_pix[0] = np.clip(y_bon_pix[0], 0, 511)  # ceiling
        y_bon_pix[1] = np.clip(y_bon_pix[1], 0, 511)  # floor
        
        # Crear visualización
        img_vis = (img_array * 0.5).astype(np.uint8)  # Oscurecer imagen un poco
        
        # Dibujar las líneas de ceiling y floor (verde)
        for x in range(1024):
            img_vis[y_bon_pix[0][x], x] = [0, 255, 0]  # Verde para ceiling
            img_vis[y_bon_pix[1][x], x] = [0, 255, 0]  # Verde para floor
            
        # Dibujar probabilidades de corner como barra en la parte superior
        cor_height = 30  
        gt_cor = np.zeros((cor_height, 1024, 3), np.uint8)
        gt_cor[:] = (y_cor[None, :, None] * 255).astype(np.uint8)  # Escala de grises
        
        separator = np.ones((3, 1024, 3), np.uint8) * 255
        
        # Concatenar: corner heatmap + separador + imagen con boundaries
        final_vis = np.concatenate([gt_cor, separator, img_vis], axis=0)
        
        return Image.fromarray(final_vis)
        
    except Exception as e:
        print(f"Error al dibujar layout: {e}")
        import traceback
        traceback.print_exc()
        # En caso de error, devuelve la imagen original
        return Image.open(img_path).convert("RGB")


def generar_figura_3d_plotly(layout_plan, room_data, items_recomendados, altura_pared=250):
    """
    Genera visualización 3D interactiva, usando modelos .OBJ para estructura y muebles.
    Si un OBJ no carga, el elemento es OMITIDO.
    """
    if not layout_plan:
        return go.Figure()

    # --- Mapeo y Constantes ---
    WALL_COLOR = '#bdbdbd' 
    CORNER_COLOR = '#6d6d6d' 
    DOOR_COLOR = '#8d6e63' 
    WINDOW_COLOR = '#d4e6f1' 
    WALL_H = altura_pared 
    WALL_DEPTH_CM = 10 # Profundidad de la pared a renderizar en 3D

    MODEL_MAP = {
        'Sofás': 'loungeSofa.obj',
        'Sillones': 'loungeChair.obj', 
        'Mesas bajas de salón, de centro y auxiliares': 'tableCoffee.obj',
        'Muebles de salón': 'cabinetTelevision.obj',
        'Estanterías y librerías': 'bookcaseOpen.obj' 
    }
    
    OBSTACLE_MAP = {
        # w y l representan las dimensiones del OBJ
        'door': {'obj': 'wallDoorway.obj', 'color': DOOR_COLOR, 'height': 200, 'v_offset': 0, 'w': WALL_DEPTH_CM, 'l': 0}, 
        'window': {'obj': 'wallWindow.obj', 'color': WINDOW_COLOR, 'height': 120, 'v_offset': 100, 'w': WALL_DEPTH_CM, 'l': 0}
    }

    pool_items = {}
    if items_recomendados:
        for it in items_recomendados: 
            pool_items.setdefault(it['Tipo_mueble'], []).append(it)

    colores = {
        'Sofás': '#7f8c8d', 'Muebles de salón': '#95a5a6', 
        'Mesas bajas de salón, de centro y auxiliares': '#d6bfa9', 
        'Sillones': '#5d6d7e', 'Estanterías y librerías': '#ecf0f1' 
    }
    
    polygon_points = room_data.get('polygon_points', [])
    poly_pts_cm = np.array(polygon_points) * 100 
    data = []

    # --- FASE 1: DIBUJAR ESTRUCTURA DE LA HABITACIÓN (PAREDES Y OBSTÁCULOS) ---
    if len(poly_pts_cm) > 1:
        num_walls = len(poly_pts_cm)
        all_obstacles = room_data.get('doors', []) + room_data.get('windows', [])

        for i in range(num_walls):
            p1 = poly_pts_cm[i]
            p2 = poly_pts_cm[(i+1) % num_walls]
            
            length, cx_seg, cy_seg, angle = get_segment_properties(p1, p2)
            
            # Identificar obstáculos en esta pared (índice i) y ordenarlos
            wall_obstacles = []
            for obs in all_obstacles:
                # Nota: obs['center'][1] es el índice normalizado de pared (0 a 1)
                wall_idx_obs = int(round(obs['center'][1] * num_walls)) % num_walls 
                if wall_idx_obs == i:
                    obs['type'] = 'door' if 'doors' in room_data and obs in room_data['doors'] else 'window'
                    wall_obstacles.append(obs)
            wall_obstacles.sort(key=lambda x: x['center'][0]) # Ordenar por posición a lo largo de la pared

            # Definir segmentos de pared a dibujar
            segments_to_draw = []
            current_start_pct = 0.0 
            
            for obs in wall_obstacles:
                center_pct = obs['center'][0]
                width_m = obs['width']
                width_pct = (width_m * 100) / length 
                
                obs_start_pct = max(0.0, center_pct - width_pct / 2)
                obs_end_pct = min(1.0, center_pct + width_pct / 2)
                
                # Segmento de pared antes del obstáculo (pared vacía)
                if obs_start_pct > current_start_pct:
                    segments_to_draw.append({'type': 'wall', 'start': current_start_pct, 'end': obs_start_pct})
                
                # Segmento de obstáculo
                segments_to_draw.append({'type': obs['type'], 'start': obs_start_pct, 'end': obs_end_pct, 'w': width_m * 100})
                
                current_start_pct = obs_end_pct
                
            # Segmento de pared final (pared vacía)
            if current_start_pct < 1.0:
                segments_to_draw.append({'type': 'wall', 'start': current_start_pct, 'end': 1.0})
            
            # --- DIBUJAR LOS SEGMENTOS CON OBJS ---
            
            for seg in segments_to_draw:
                seg_start_cm = seg['start'] * length
                seg_end_cm = seg['end'] * length
                seg_len = seg_end_cm - seg_start_cm
                
                if seg_len < 1: continue 

                # Recalcular centro y ángulo para el subsegmento
                seg_mid_x = p1[0] + (seg['start'] + seg['end']) / 2 * (p2[0] - p1[0])
                seg_mid_y = p1[1] + (seg['start'] + seg['end']) / 2 * (p2[1] - p1[1])

                # Configuración del modelo
                if seg['type'] == 'wall':
                    obj_file = 'wall.obj'
                    color = WALL_COLOR
                    h_val = WALL_H
                    z_base = 0
                    w_seg = seg_len # El largo del segmento es el ANCHO (X) del OBJ
                    l_seg = WALL_DEPTH_CM # La profundidad de la pared es el LARGO (Y) del OBJ
                else: 
                    obs_data = OBSTACLE_MAP[seg['type']]
                    obj_file = obs_data['obj']
                    color = obs_data['color']
                    h_val = obs_data['height']
                    z_base = obs_data['v_offset']
                    w_seg = seg_len
                    l_seg = WALL_DEPTH_CM 
                
                # Cargar y transformar el OBJ
                
                # Lista de elementos a dibujar en este segmento (puede ser múltiple para ventanas/puertas)
                sub_elements = []
                
                if seg['type'] == 'wall':
                    sub_elements.append({
                        'obj': 'wall.obj', 'color': WALL_COLOR, 
                        'h': WALL_H, 'z': 0, 
                        'w': seg_len, 'l': WALL_DEPTH_CM,
                        'name': 'Pared'
                    })
                else: 
                    obs_data = OBSTACLE_MAP[seg['type']]
                    
                    # 1. El Obstáculo en sí
                    sub_elements.append({
                        'obj': obs_data['obj'], 'color': obs_data['color'],
                        'h': obs_data['height'], 'z': obs_data['v_offset'],
                        'w': seg_len, 'l': WALL_DEPTH_CM,
                        'name': seg['type'].title()
                    })
                    
                    # 2. Relleno SUPERIOR (Dintel) - Si hay espacio hasta el techo
                    top_gap = WALL_H - (obs_data['v_offset'] + obs_data['height'])
                    if top_gap > 1:
                        sub_elements.append({
                            'obj': 'wall.obj', 'color': WALL_COLOR,
                            'h': top_gap, 'z': obs_data['v_offset'] + obs_data['height'],
                            'w': seg_len, 'l': WALL_DEPTH_CM,
                            'name': 'Muro Superior'
                        })
                        
                    # 3. Relleno INFERIOR (Antepecho) - Si el obstáculo no empieza en el suelo
                    bottom_gap = obs_data['v_offset']
                    if bottom_gap > 1:
                        sub_elements.append({
                            'obj': 'wall.obj', 'color': WALL_COLOR,
                            'h': bottom_gap, 'z': 0,
                            'w': seg_len, 'l': WALL_DEPTH_CM,
                            'name': 'Muro Inferior'
                        })

                # Renderizar todos los sub-elementos del segmento
                for el in sub_elements:
                    x_seg, y_seg, z_seg, i_seg, j_seg, k_seg = load_and_transform_mesh(
                        el['obj'], w=el['w'], l=el['l'], h=el['h'], 
                        cx=seg_mid_x, cy=seg_mid_y, angle=angle, base_z=el['z']
                    )
                    
                    if x_seg is not None:
                        data.append(go.Mesh3d(
                            x=x_seg, y=y_seg, z=z_seg, i=i_seg, j=j_seg, k=k_seg,
                            color=el['color'], opacity=1.0, flatshading=True,
                            name=f'{el["name"]} P{i}', showlegend=(seg['type']!='wall' and el['name'] == seg['type'].title())
                        ))
                    else:
                        print(f"!!! FALLO DE CARGA: Omisión de {el['name']} en pared {i}.")
            
            # 5. ESQUINA (wallCorner.obj) - Se dibuja en el vértice P1
            if i == 0 or True: # Dibujar esquina en cada vértice para cerrar bien
                # Nota: Dibujamos esquina en P1 (inicio del segmento)
                x_c, y_c, z_c, i_c, j_c, k_c = load_and_transform_mesh(
                    'wallCorner.obj', w=WALL_DEPTH_CM, l=WALL_DEPTH_CM, h=WALL_H, 
                    cx=p1[0], cy=p1[1], angle=angle
                )
                
                if x_c is not None:
                    data.append(go.Mesh3d(
                        x=x_c, y=y_c, z=z_c, i=i_c, j=i_c, k=i_c, 
                        color=CORNER_COLOR, opacity=1.0, flatshading=True,
                        name=f'Esquina {i}', showlegend=False
                    ))


    # --- FASE 2: DIBUJAR SUELO ---
    # Usar un fill poly simple. Convertir a (X, Y, Z) para Plotly
    x_floor = poly_pts_cm[:, 0]
    y_floor = poly_pts_cm[:, 1]
    z_floor = np.zeros_like(x_floor)

    # Crear caras del suelo (convexhull)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(np.array([x_floor, y_floor]).T)
        i_f, j_f, k_f = hull.simplices.T
    except: # Fails if points are collinear or too few
        i_f, j_f, k_f = [], [], []

    suelo_trace = go.Mesh3d(
        x=x_floor, y=y_floor, z=z_floor,
        i=i_f, j=j_f, k=k_f,
        color='#fafafa', opacity=1, name='Suelo', hoverinfo='skip'
    )
    data.append(suelo_trace)

    # --- FASE 3: DIBUJAR MUEBLES (OBJ o OMISIÓN) ---
    for mueble in layout_plan:
        tipo = mueble['tipo']
        obj_file = MODEL_MAP.get(tipo)
        # Buscar la info real del mueble seleccionado
        info_real = pool_items.get(tipo, [{}])[0] if tipo in pool_items else {}
            
        nombre_display = info_real.get('Nombre', tipo)
        precio = info_real.get('Precio', '?')
        desc = info_real.get('Descripcion', '')[:60]
        
        hover_text = (
            f"<b>TIPO:</b> {tipo}<br>"
            f"<b>MODELO:</b> {nombre_display}<br>"
            f"<b>PRECIO:</b> {precio}€<br>"
            f"<i>{desc}...</i>"
        )
        
        try:
            h_val = float(info_real.get('Altura', 60))
            if np.isnan(h_val) or h_val <= 0: h_val = 60
        except: h_val = 60

        w_m = mueble['ancho']
        l_m = mueble['largo']
        
        # Rotación extra para sofás (suelen venir mirando hacia atrás)
        rot_offset = np.pi if tipo == 'Sofás' else 0
        
        # Lógica específica para detectar Sofás en L (Chaise Longue / Rinconera)
        if tipo == 'Sofás':
            keywords_l_shape = ['chaise', 'esquina', 'rincon', 'l-shaped', 'modular', 'l shape']
            text_to_search = (nombre_display + " " + desc).lower()
            if any(k in text_to_search for k in keywords_l_shape):
                obj_file = 'loungeDesignSofaCorner.obj'
                # Ajuste de rotación específico para este modelo si es necesario (a veces los modelos de esquina tienen otra orientación)
                # Por ahora mantenemos la rotación de sofá estándar (pi) o ajustamos si el usuario reporta algo raro.
                # rot_offset = np.pi

        if obj_file:
            x_m, y_m, z_m, i_m, j_m, k_m = load_and_transform_mesh(
                obj_file, w=w_m, l=l_m, h=h_val,
                cx=mueble['x'], cy=mueble['y'], angle=mueble['angle'],
                rotation_offset=rot_offset
            )

            if x_m is not None:
                traces = [go.Mesh3d(
                    x=x_m, y=y_m, z=z_m, i=i_m, j=j_m, k=k_m,
                    color=colores.get(tipo, '#95a5a6'), opacity=1.0, flatshading=True,
                    name=nombre_display, hoverinfo='text', text=hover_text,
                    lighting=dict(ambient=0.6, diffuse=0.8), showlegend=True
                )]
                data.extend(traces)
            else:
                # Fallo de carga: OMITIR
                print(f"!!! FALLO DE CARGA/RENDERIZADO: {tipo} ({nombre_display}). Modelo OBJ no usado.")
                continue 
        else:
            # Omisión si no hay OBJ mapeado
            print(f"!!! OMISIÓN: No hay OBJ mapeado para el tipo: {tipo}. Saltando renderizado.")
            continue

    # --- CONFIGURACIÓN FINAL ---
    max_dim = np.max(poly_pts_cm, axis=0) if poly_pts_cm.size > 0 else [500, 500]

    layout = go.Layout(
        title="Diseño 3D (Interactúa con el ratón)",
        showlegend=True,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data', # Usar 'data' para que los ejes sean proporcionales a los valores reales
            aspectratio=None, 
            bgcolor='white',
            camera=dict(eye=dict(x=2.0, y=2.0, z=2.0)) # Zoom out inicial
        ),
        margin=dict(r=0, l=0, b=0, t=30),
        height=600 
    )
    
    return go.Figure(data=data, layout=layout)


def generar_diagrama_planta(room_data):
    """Genera un diagrama de planta 2D de la habitación con paredes, puertas y ventanas."""
    try:
        # --- 1. CONFIGURACIÓN DE ESTILO ---
        COLORS = {
            'bg': '#1C4E80', # Azul oscuro de fondo
            'line': '#ffffff', # Blanco para líneas
            'hole': '#1C4E80', # Mismo color que el fondo para "borrar" la pared
            'grid': '#ffffff',
            'text': '#ffffff'
        }

        WALL_WIDTH = 6
        HOLE_WIDTH = 8
        ELEM_WIDTH = 1.5

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(COLORS['bg'])
        ax.set_facecolor(COLORS['bg'])
        
        polygon_points = room_data.get('polygon_points', None)
        if polygon_points is None or len(polygon_points) < 3:
            ax.text(0.5, 0.5, "ERROR: Polígono de habitación inválido", color=COLORS['text'], ha='center')
            return fig
        
        polygon_m = np.array(polygon_points)
        centroid = np.mean(polygon_m, axis=0) 
        
        # A. Forzar proporción real (1 metro visual = 1 metro dato)
        ax.set_aspect('equal', adjustable='box') 
        
        # B. Calcular límites enteros para asegurar que el grid cae en el metro exacto
        min_x, min_y = np.min(polygon_m, axis=0)
        max_x, max_y = np.max(polygon_m, axis=0)
        
        # Márgenes de 1 metro extra alrededor
        start_x = np.floor(min_x - 1)
        end_x = np.ceil(max_x + 1)
        start_y = np.floor(min_y - 1)
        end_y = np.ceil(max_y + 1)
        
        ax.set_xlim(start_x, end_x)
        ax.set_ylim(start_y, end_y)
        
        # C. Definir ticks explícitamente cada 1.0 unidades (1 metro)
        xticks = np.arange(start_x, end_x + 1, 1.0)
        yticks = np.arange(start_y, end_y + 1, 1.0)
        
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
        # Grid muy sutil
        ax.grid(True, color=COLORS['grid'], linestyle=':', linewidth=0.5, alpha=0.2)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(length=0)
        
        # --- Helper: Datos de Muro (Necesario aquí para Doors/Windows) ---
        num_walls = len(polygon_m)
        def get_wall_data(idx, pct):
            idx = idx % num_walls
            p1, p2 = polygon_m[idx], polygon_m[(idx + 1) % num_walls]
            vec = p2 - p1
            L_wall = np.linalg.norm(vec)
            if L_wall == 0: return None
            unit = vec / L_wall
            center_on_wall = p1 + vec * (pct / 100.0)
            
            # Vector normal hacia adentro
            n1 = np.array([-unit[1], unit[0]])
            # Comprobar si n1 apunta hacia el centroide
            if np.linalg.norm((center_on_wall + n1) - centroid) > np.linalg.norm((center_on_wall - n1) - centroid):
                normal_in = -n1
            else:
                normal_in = n1
            return center_on_wall, unit, normal_in

        # --- 3. DIBUJAR PAREDES ---
        polygon_closed = np.vstack([polygon_m, polygon_m[0]])
        
        # Relleno muy sutil del suelo
        ax.fill(polygon_closed[:, 0], polygon_closed[:, 1], color=COLORS['line'], alpha=0.05, zorder=1)
        
        # EL MURO GRUESO
        ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], 
               color=COLORS['line'], linewidth=WALL_WIDTH, zorder=2, solid_capstyle='round')


        # --- 4. PUERTAS (Hueco + Hoja + Arco) ---
        for d in room_data.get('doors', []):
            # Obtener índice de pared y posición porcentual a lo largo de esa pared
            wall_idx = int(round(d['center'][1] * num_walls)) % num_walls
            pos_pct = d['center'][0] * 100 # a cm
            
            info = get_wall_data(wall_idx, pos_pct)
            if not info: continue
            center, unit, normal_in = info
            w = d['width'] # ancho de la puerta en metros
            
            # A. HUECO (Borrar muro)
            h_s = center - unit * (w/2)
            h_e = center + unit * (w/2)
            ax.plot([h_s[0], h_e[0]], [h_s[1], h_e[1]], 
                   color=COLORS['hole'], linewidth=HOLE_WIDTH, zorder=5) 
            
            # B. HOJA
            hinge = h_s
            tip = hinge + normal_in * w
            ax.plot([hinge[0], tip[0]], [hinge[1], tip[1]], 
                   color=COLORS['line'], linewidth=ELEM_WIDTH, zorder=6)
            
            # C. ARCO (Interpolado)
            arc_pts = []
            start_angle = np.arctan2(unit[1], unit[0])
            
            # Crear la rotación desde el vector 'unit' al vector 'normal_in'
            for t in np.linspace(0, 1, 15):
                angle_interp = t * (np.pi/2)
                # Aplicar rotación al vector 'unit'
                v_rot = unit * np.cos(angle_interp) + normal_in * np.sin(angle_interp)
                pt = hinge + v_rot * w
                arc_pts.append(pt)
            arc_pts = np.array(arc_pts)
            ax.plot(arc_pts[:, 0], arc_pts[:, 1], 
                   color=COLORS['line'], linestyle=':', linewidth=1, zorder=6)

        # --- 5. VENTANAS (Hueco + Rectángulo vacío) ---
        for w_obj in room_data.get('windows', []):
            wall_idx = int(round(w_obj['center'][1] * num_walls)) % num_walls
            pos_pct = w_obj['center'][0] * 100
            
            info = get_wall_data(wall_idx, pos_pct)
            if not info: continue
            center, unit, normal_in = info
            w = w_obj['width'] # ancho de la ventana en metros
            
            # A. HUECO (Borrar muro grueso)
            h_s = center - unit * (w/2); h_e = center + unit * (w/2)
            ax.plot([h_s[0], h_e[0]], [h_s[1], h_e[1]], 
                   color=COLORS['hole'], linewidth=HOLE_WIDTH, zorder=5)
            
            # B. MARCO RECTANGULAR (Sin relleno)
            frame_depth = 0.1 # 10 cm de profundidad de marco (en metros)
        
            # 4 Esquinas del rectángulo
            c1 = h_s - normal_in * (frame_depth/2)
            c2 = h_e - normal_in * (frame_depth/2)
            c3 = h_e + normal_in * (frame_depth/2)
            c4 = h_s + normal_in * (frame_depth/2)
            
            # Dibujar perímetro (c1->c2->c3->c4->c1)
            rect_x = [c1[0], c2[0], c3[0], c4[0], c1[0]]
            rect_y = [c1[1], c2[1], c3[1], c4[1], c1[1]]
            
            ax.plot(rect_x, rect_y, color=COLORS['line'], linewidth=ELEM_WIDTH, zorder=6)

        # --- 6. ETIQUETAS Y LEYENDA ---
        legend_items = []
        for i in range(num_walls):
            p1, p2 = polygon_m[i], polygon_m[(i + 1) % num_walls]
            mid = (p1 + p2) / 2
            
            # Vector hacia afuera para etiqueta
            vec_out = mid - centroid
            vec_out = vec_out / np.linalg.norm(vec_out)
            text_pos = mid + vec_out * 0.5 # 50cm afuera para no tocar el muro grueso
            
            L = np.linalg.norm(p2 - p1)
            legend_items.append(f"P{i+1}: {L:.2f} m")
            
            ax.text(text_pos[0], text_pos[1], f"P{i+1}", color=COLORS['bg'], fontsize=8, fontweight='bold',
                   ha='center', va='center', zorder=10,
                   bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='none'))
            
        plt.subplots_adjust(right=0.70)
        
        info_text = "HABITACIÓN\n(Grid 1x1m)\n\n" + "\n".join(legend_items)
        
        fig.text(0.72, 0.5, info_text, fontsize=10, color=COLORS['text'], 
                fontfamily='monospace', va='center',
                bbox=dict(boxstyle='square,pad=1', fc=COLORS['hole'], ec=COLORS['line']))

        return fig

    except Exception as e:
        print(f"Error planta: {e}")
        import traceback
        traceback.print_exc()
        return plt.figure()