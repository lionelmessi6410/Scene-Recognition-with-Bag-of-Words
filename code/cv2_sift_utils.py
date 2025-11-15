"""
Utilidades para reemplazar cyvlfeat con OpenCV
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

def dsift(img, step=[5, 5], fast=True):
    """
    Extrae descriptores SIFT densos usando OpenCV como reemplazo de cyvlfeat.dsift
    
    Args:
        img: imagen en formato numpy array (float32)
        step: paso para el muestreo denso [step_y, step_x]
        fast: si es True, usa menos octavas para mayor velocidad
        
    Returns:
        frames: coordenadas de los keypoints (N, 2) [y, x]
        descriptors: descriptores SIFT (N, 128)
    """
    # Convertir a uint8 si es necesario
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convertir a escala de grises si es necesario
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Crear detector SIFT
    if fast:
        sift = cv2.SIFT_create(nOctaveLayers=2, contrastThreshold=0.02, edgeThreshold=5)
    else:
        sift = cv2.SIFT_create()
    
    # Generar grid de keypoints
    step_y, step_x = step
    keypoints = []
    
    height, width = img_gray.shape
    for y in range(0, height - 16, step_y):  # -16 para evitar bordes
        for x in range(0, width - 16, step_x):
            # Crear keypoint con tamaño fijo
            kp = cv2.KeyPoint(x=float(x), y=float(y), size=16.0)
            keypoints.append(kp)
    
    # Calcular descriptores para los keypoints
    if len(keypoints) == 0:
        return np.zeros((0, 2)), np.zeros((0, 128))
    
    keypoints, descriptors = sift.compute(img_gray, keypoints)
    
    # Extraer coordenadas de frames
    if descriptors is None or len(keypoints) == 0:
        return np.zeros((0, 2)), np.zeros((0, 128))
    
    frames = np.array([[kp.pt[1], kp.pt[0]] for kp in keypoints])  # [y, x]
    
    # Asegurar que descriptors sea float32
    descriptors = descriptors.astype(np.float32)
    
    return frames, descriptors


def kmeans(X, K, initialization="PLUSPLUS"):
    """
    Realiza clustering K-means usando scikit-learn como reemplazo de cyvlfeat.kmeans
    
    Args:
        X: matriz de características (M, d)
        K: número de clusters
        initialization: método de inicialización
        
    Returns:
        centers: centroides de los clusters (K, d)
    """
    # Usar KMeans de scikit-learn
    init_method = 'k-means++' if initialization == "PLUSPLUS" else 'random'
    
    kmeans_model = KMeans(
        n_clusters=K,
        init=init_method,
        n_init=10,
        max_iter=300,
        random_state=42,
        verbose=0
    )
    
    kmeans_model.fit(X)
    
    # Retornar centroides (K, d)
    return kmeans_model.cluster_centers_
