import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
data_dir = 'path_to_UECFOOD256/'  # Path to your UECFOOD256 dataset directory
output_dir = 'data/processed/'  # Directory to save processed images

# Image settings
img_size = (224, 224)  # Image size for resizing
test_size = 0.2  # Fraction of data for validation


# Function to load image paths and labels
def load_food_data(data_dir):
    """
    Loads image file paths and corresponding labels from the UECFOOD256 dataset.

    :param data_dir: Path to the UECFOOD256 dataset
    :return: Tuple (image_paths, labels)
    """
    image_paths = []
    labels = []
    label_map = {}

    # Loop through the directories corresponding to food classes
    for idx, class_dir in enumerate(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            label_map[class_dir] = idx  # Map the food class to an integer label
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                if img_file.endswith('.jpg'):
                    image_paths.append(img_path)
                    labels.append(idx)

    return image_paths, labels, label_map


# Function to save images to the output directory
def save_images(images, labels, output_dir):
    """
    Saves processed images to the output directory.

    :param images: List of preprocessed images
    :param labels: Corresponding labels for each image
    :param output_dir: Directory where processed images will be saved
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, img in enumerate(images):
        label = labels[idx]
        img_path = os.path.join(output_dir, f"{label}_{idx}.jpg")
        cv2.imwrite(img_path, img)
        print(f"Saved image {img_path}")


# Function to preprocess images and split data
def preprocess_images(image_paths, labels, output_dir, img_size, test_size):
    """
    Processes images and creates train/validation split.

    :param image_paths: List of image file paths
    :param labels: Corresponding labels for each image
    :param output_dir: Directory to save processed images
    :param img_size: Tuple for resizing images
    :param test_size: Fraction of data for validation
    """
    images = []
    for img_path in image_paths:
        # Read and resize the image (using OpenCV)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=42)

    # Save preprocessed images
    save_images(X_train, y_train, os.path.join(output_dir, 'train'))
    save_images(X_val, y_val, os.path.join(output_dir, 'val'))

    print(f"Preprocessing completed. Train set: {len(X_train)}, Validation set: {len(X_val)}")


if __name__ == '__main__':
    # Load image paths and labels from UECFOOD256
    image_paths, labels, label_map = load_food_data(data_dir)

    # Preprocess images and split into train and validation sets
    preprocess_images(image_paths, labels, output_dir, img_size, test_size)