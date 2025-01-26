import os
import shutil
import random
import sys

def split_dataset(dataset_path, output_path, train_ratio=0.8):
    """
    Split a dataset of images organized in class-specific folders into
    training and testing datasets.

    Args:
        dataset_path (str): Path to the dataset with class folders.
        output_path (str): Path to the output directory for train/test folders.
        train_ratio (float): Ratio of data to use for training (default 0.8).

    Structure:
    dataset_path/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...

    Output:
    output_path/
        train/
            class1/
                img1.jpg
                ...
            class2/
                ...
        test/
            class1/
                img1.jpg
                ...
            class2/
                ...
    """
    random.seed(42)  # For reproducibility

    # Paths for train and test folders
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')

    # Create train and test directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Iterate through class folders
    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)

        # Ensure it's a folder
        if not os.path.isdir(class_folder):
            continue

        # Get all image file paths in the class folder
        images = [img for img in os.listdir(class_folder) if img.endswith(('jpg', 'png', 'jpeg', 'tif'))]

        # Shuffle and split images
        random.shuffle(images)
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Create class subfolders in train and test directories
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

        # Move images to the respective folders
        for img in train_images:
            src_path = os.path.join(class_folder, img)
            dst_path = os.path.join(train_path, class_name, img)
            shutil.copy(src_path, dst_path)

        for img in test_images:
            src_path = os.path.join(class_folder, img)
            dst_path = os.path.join(test_path, class_name, img)
            shutil.copy(src_path, dst_path)

        print(f"Processed class '{class_name}': {len(train_images)} train, {len(test_images)} test")

    print("Dataset split completed!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_dataset.py <dataset_path> <output_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]  # First command-line argument
    output_path = sys.argv[2]  # Second command-line argument

    split_dataset(dataset_path, output_path, train_ratio=0.8)
