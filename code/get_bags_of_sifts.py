from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import cv2
from time import time

def get_bags_of_sifts(image_paths):
    '''
    Input : 
        image_paths : list(N) of image paths
    Output : 
        image_feats : (N, d) feature, each row is a normalized histogram for one image
    '''
    # Carrega o vocabulário (cluster centers do KMeans)
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)
    vocab = np.array(vocab, dtype=np.float32)

    image_feats = []

    sift = cv2.SIFT_create()  # cria extrator SIFT
    start_time = time()
    print("Constructing bags of SIFTs...")

    for path in image_paths:
        img = np.array(Image.open(path).convert('L'), dtype='uint8')  # grayscale
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None or len(descriptors) == 0:
            # caso não encontre keypoints, adiciona histograma vazio
            hist_norm = np.zeros(len(vocab))
        else:
            # calcula distância entre descritores e centros do vocabulário
            dist = distance.cdist(descriptors, vocab, metric='euclidean')
            idx = np.argmin(dist, axis=1)

            # histogram of visual words
            hist, _ = np.histogram(idx, bins=np.arange(len(vocab)+1))
            hist_norm = hist.astype('float32') / np.sum(hist)

        image_feats.append(hist_norm)

    image_feats = np.vstack(image_feats)
    end_time = time()
    print(f"It took {end_time - start_time:.2f} seconds to construct bags of SIFTs.")

    return image_feats
