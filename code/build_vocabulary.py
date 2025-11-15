from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from time import time

def build_vocabulary(image_paths, vocab_size):
    bag_of_features = []
    print("Extracting SIFT features...")

    sift = cv2.SIFT_create()  # cria o extrator SIFT

    for path in image_paths:
        img = np.array(Image.open(path).convert('L'))  # converte para grayscale
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            bag_of_features.append(descriptors)

    # junta todos os descritores de todas as imagens
    bag_of_features = np.vstack(bag_of_features).astype('float32')

    print("Clustering with KMeans...")
    start_time = time()

    # substitui o vl_kmeans por scikit-learn
    kmeans = KMeans(
        n_clusters=vocab_size,
        init="k-means++",
        n_init=10,
        random_state=42,
        verbose=1
    )
    kmeans.fit(bag_of_features)

    end_time = time()
    print("Vocabulary built in %.2f seconds." % (end_time - start_time))

    vocab = kmeans.cluster_centers_.astype('float32')
    return vocab
