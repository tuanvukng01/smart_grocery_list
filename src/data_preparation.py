import os
import cv2
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
train_mat_file_path = 'path_to_training_mat_file.mat'  # Update with your training .mat file path
image_folder = 'data/food475/images/'  # Folder where images are stored
output_dir = 'data/processed/'  # Where the processed images will be saved

# Image settings
img_size = (224, 224)  # Image size for resizing
test_size = 0.2  # Fraction of data for validation


# Function to load food names and IDs
def load_food_data(mat_file_path):
    """
    Loads the food names and IDs from a .mat file.

    :param mat_file_path: Path to the .mat file containing food data
    :return: Tuple (food_ids, food_names)
    """
    # Load the .mat file
    mat = scipy.io.loadmat(mat_file_path)

    # Extract relevant data
    food_ids = mat['id']  # Extract food IDs
    food_names = mat['name']  # Extract food names

    # Convert the extracted data into usable formats
    food_ids = [food_id[0] for food_id in food_ids]
    food_names = [food_name[0] for food_name in food_names]

    return food_ids, food_names


# Function to construct image paths based on food IDs or names
def get_image_path(food_id):
    """
    Constructs the image path based on the food ID.

    :param food_id: ID of the food item
    :return: Full image path
    """
    # Construct image path based on the food ID (adjust this to match your file structure)
    image_filename = f"{food_id}.jpg"  # Assuming images are named by food ID
    image_path = os.path.join(image_folder, image_filename)
    return image_path


# Function to save images to the output directory
def save_images(images, labels, output_dir):
    """
    Saves processed images to the output directory.

    :param images: List of preprocessed images
    :param labels: Corresponding labels (IDs) for each image
    :param output_dir: Directory where the processed images will be saved
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, img in enumerate(images):
        label = labels[idx]
        img_path = os.path.join(output_dir, f"{label}_{idx}.jpg")
        cv2.imwrite(img_path, img)
        print(f"Saved image {img_path}")


# Function to preprocess images and split data
def preprocess_images(food_ids, output_dir, img_size, test_size):
    """
    Processes images and creates train/validation split.

    :param food_ids: List of food IDs
    :param output_dir: Directory to save processed images
    :param img_size: Tuple for resizing images
    :param test_size: Fraction of data for validation
    """
    images = []
    labels = []

    for food_id in food_ids:
        image_path = get_image_path(food_id)

        # Read and resize the image (using OpenCV)
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(food_id)

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
    # Load food data from the training .mat file
    food_ids, food_names = load_food_data(train_mat_file_path)

    # Preprocess images and split into train and validation sets
    preprocess_images(food_ids, output_dir, img_size, test_size)