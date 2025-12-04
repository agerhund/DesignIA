import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import scipy.signal
import matplotlib.pyplot as plt
from torchvision import transforms

# Importamos la arquitectura del modelo desde el archivo del repositorio
# Asegúrate de que este script esté en la misma carpeta que model.py
try:
    from model import HorizonNet
except ImportError:
    print("Error: No se encuentra 'model.py'. Asegúrate de guardar este script en la raíz de la carpeta HorizonNet-master.")
    sys.exit(1)

def cargar_modelo(ruta_checkpoint, device):
    """
    Carga el modelo entrenado y sus pesos.
    """
    print(f"Cargando modelo desde: {ruta_checkpoint}")
    
    # Inicializar la arquitectura (Backbone ResNet50 es el default habitual)
    # Si usaste otro backbone (ej. resnet18), cámbialo aquí.
    net = HorizonNet(backbone='resnet50', use_rnn=True).to(device)

    # Cargar los pesos
    # map_location es importante para evitar errores si se entrenó en GPU y se usa CPU/MPS
    checkpoint = torch.load(ruta_checkpoint, map_location=device)
    
    # A veces el checkpoint guarda todo el estado ('state_dict') o solo los pesos
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Fix para cuando el modelo se entrenó con DataParallel (nombres empiezan por 'module.')
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    net.load_state_dict(new_state_dict)
    net.eval() # Poner en modo evaluación (importante para BatchNorm/Dropout)
    return net

def procesar_imagen(ruta_imagen):
    """
    Preprocesa la imagen panorámica al formato que espera HorizonNet (512x1024).
    """
    img_pil = Image.open(ruta_imagen).convert('RGB')
    
    # Redimensionar a 512 (alto) x 1024 (ancho)
    img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
    
    # Convertir a tensor y normalizar (Mean/Std estándar de ImageNet)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = to_tensor(img_pil)
    return img_tensor.unsqueeze(0), img_pil # Añadir dimensión de batch

def post_procesado_coordenadas(y_bon, y_cor, altura_camara=1.6):
    """
    Convierte la salida de la red neuronal en coordenadas 2D (planta).
    """
    W = 1024
    H = 512
    
    # 1. Detectar esquinas (Picos de probabilidad)
    # distance=20 evita detectar picos demasiado juntos
    # height=0.5 es el umbral de confianza (ajustable)
    cor_ids, _ = scipy.signal.find_peaks(y_cor, height=0.5, distance=20)
    
    # Fallback: Si no detecta esquinas, asumimos una forma básica o usamos todo el perímetro
    if len(cor_ids) < 2:
        print("Advertencia: No se detectaron esquinas claras. Usando muestreo uniforme.")
        cor_ids = np.linspace(0, W-1, 50, dtype=int)

    # Aseguramos que el último punto conecte con el primero para cerrar el polígono visualmente
    # (Aunque para cálculos de área mejor no duplicarlo)
    
    coordenadas_x = []
    coordenadas_y = []

    for c in cor_ids:
        # A. Ángulo horizontal (Theta)
        # Mapeamos 0..1024 a -Pi..Pi
        theta = (c / W) * 2 * np.pi - np.pi
        
        # B. Ángulo vertical (Phi) - Profundidad
        # y_bon[1] es el suelo, y_bon[0] es el techo
        v_suelo = y_bon[1, c]
        
        # Convertir pixel Y a ángulo de elevación respecto al horizonte
        # En HorizonNet: v=0 es techo, v=H es suelo. Centro es H/2.
        phi = (v_suelo - (H / 2)) / (H / 2) * (np.pi / 2)
        
        # Calcular distancia al suelo
        # Evitamos división por cero o ángulos imposibles
        if phi <= 0.05: phi = 0.05 
        distancia = altura_camara / np.tan(phi)
        
        # C. Coordenadas Polares a Cartesianas
        x = distancia * np.sin(theta)
        y = distancia * np.cos(theta)
        
        coordenadas_x.append(x)
        coordenadas_y.append(y)
        
    return list(zip(coordenadas_x, coordenadas_y))

def calcular_area(puntos):
    """Calcula el área del polígono usando la fórmula de Shoelace (Gauß)"""
    x = [p[0] for p in puntos]
    y = [p[1] for p in puntos]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def visualizar_resultado(img_pil, puntos, output_path="resultado_layout.png"):
    """
    Genera una imagen con la vista panorámica y el plano 2D al lado.
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Imagen Original
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.title("Vista Panorámica (Input)")
    plt.axis('off')
    
    # 2. Plano 2D Generado
    plt.subplot(1, 2, 2)
    
    # Extraer X e Y
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    
    # Cerrar el polígono para el dibujo
    xs.append(xs[0])
    ys.append(ys[0])
    
    plt.plot(xs, ys, 'b-', linewidth=2, marker='o')
    plt.fill(xs, ys, alpha=0.3, color='blue')
    
    # Dibujar la cámara
    plt.plot(0, 0, 'rx', label="Cámara")
    
    # Estética
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # Importante para no distorsionar proporciones
    plt.title("Plano de Planta Reconstruido (Metros)")
    plt.xlabel("X (metros)")
    plt.ylabel("Y (metros)")
    plt.legend()
    
    # Guardar y mostrar
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Gráfico guardado en: {output_path}")
    plt.show()

def main():
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description='Inferencia TFM - Layout Habitación')
    parser.add_argument('--img', type=str, required=True, help='Ruta a la imagen panorámica 360')
    parser.add_argument('--altura', type=float, default=1.6, help='Altura de la cámara en metros (Default: 1.6m)')
    args = parser.parse_args()

    # Configuración del dispositivo (Apple Silicon M4 support)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Usando dispositivo: Apple MPS (GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Usando dispositivo: CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print("Usando dispositivo: CPU")

    # Ruta hardcodeada de tu modelo (puedes cambiarla o pasarla por arg)
    MODEL_PATH = "/Users/ag/Documents/Master_Data_Science/97_TFM/App/HorizonNet-master/ckpt/zind_local_resnet50/best_model_5.pth.tar"

    # 1. Cargar Modelo
    try:
        model = cargar_modelo(MODEL_PATH, device)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo del modelo en: {MODEL_PATH}")
        return

    # 2. Procesar Imagen
    try:
        img_tensor, img_pil = procesar_imagen(args.img)
        img_tensor = img_tensor.to(device)
    except FileNotFoundError:
        print(f"Error: No se encuentra la imagen en: {args.img}")
        return

    # 3. Inferencia
    print("Ejecutando inferencia...")
    with torch.no_grad():
        # HorizonNet devuelve 3 valores: y_bon (techo/suelo), y_cor (esquinas)
        y_bon, y_cor = model(img_tensor)
        
        # Pasar a CPU y numpy para post-procesado
        y_bon = y_bon.cpu().numpy().squeeze() # Shape: (2, 1024)
        y_cor = y_cor.cpu().numpy().squeeze() # Shape: (1024,)

    # 4. Post-procesado (De Red Neuronal a Plano 2D)
    coordenadas_plano = post_procesado_coordenadas(y_bon, y_cor, altura_camara=args.altura)
    
    area = calcular_area(coordenadas_plano)
    
    # 5. Resultados
    print("\n--- RESULTADOS ---")
    print(f"Número de esquinas detectadas: {len(coordenadas_plano)}")
    print(f"Área estimada de la habitación: {area:.2f} m²")
    print("\nCoordenadas (X, Y) respecto al centro de la habitación:")
    for i, (x, y) in enumerate(coordenadas_plano):
        print(f"Esquina {i+1}: ({x:.2f}, {y:.2f})")
        
    # 6. Visualizar
    visualizar_resultado(img_pil, coordenadas_plano)

if __name__ == "__main__":
    main()