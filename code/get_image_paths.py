import os
from glob import glob

def get_image_paths(data_path, categories):
    image_paths = []
    labels = []

    for category in categories:
        category_path = os.path.join(data_path, category)
        images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.tif')]
        image_paths.extend(images)
        labels.extend([category] * len(images))

    return image_paths, labels

