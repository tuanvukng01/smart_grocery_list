import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.utils import load_images, save_images

# Configuration
data_dir = 'data/food475db/'
output_dir = 'data/processed/food475db/'
img_size = (224, 224)  # Image size for resizing
test_size = 0.2  # Fraction of data for validation


def preprocess_images(data_dir, output_dir, img_size, test_size):
    # Load and preprocess images
    images, labels = load_images(data_dir, img_size)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=42)

    # Save preprocessed images
    save_images(X_train, y_train, os.path.join(output_dir, 'train'))
    save_images(X_val, y_val, os.path.join(output_dir, 'val'))

    print(f"Preprocessing completed. Train set: {len(X_train)}, Validation set: {len(X_val)}")


if __name__ == '__main__':
    preprocess_images(data_dir, output_dir, img_size, test_size)