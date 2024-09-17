import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.utils import save_images

# Paths
train_mat_file_path = 'path_to_training_mat_file.mat'  # Update with your training .mat file path
data_dir = 'data/food475/'  # Path to your dataset directory containing images
output_dir = 'data/processed/'  # Where the processed images will be saved

# Image settings
img_size = (224, 224)  # Image size for resizing
test_size = 0.2  # Fraction of data for validation


# Function to load food names and images from the .mat file
def load_food_data(mat_file_path):
    """
    Loads the food names and IDs, and retrieves corresponding image paths.

    :param mat_file_path: Path to the .mat file containing food data
    :return: Tuple (food_ids, food_names, food_images) where food_images are file paths to images
    """
    # Load the .mat file
    mat = scipy.io.loadmat(mat_file_path)

    # Extract relevant data (modify keys if necessary based on file structure)
    food_names = mat['name']  # Extract food names
    food_ids = mat['id']  # Extract food IDs
    food_images = mat['images']  # Extract image file paths (assuming these are available)

    # Convert the extracted data into usable formats
    food_ids = [food_id[0] for food_id in food_ids]
    food_names = [food_name[0] for food_name in food_names]
    food_images = [img[0] for img in food_images]

    return food_ids, food_names, food_images


# Function to preprocess images and split data
def preprocess_images(food_ids, food_images, food_names, output_dir, img_size, test_size):
    """
    Processes images and creates train/validation split.

    :param food_ids: List of food IDs
    :param food_images: List of image file paths
    :param food_names: List of food names
    :param output_dir: Directory to save processed images
    :param img_size: Tuple for resizing images
    :param test_size: Fraction of data for validation
    """
    # Create a list to store processed image paths and labels
    images = []
    labels = []

    # Loop through the images and process them
    for idx, image_path in enumerate(food_images):
        # Construct the full path to the image
        full_image_path = os.path.join(data_dir, image_path)

        # Read and resize the image (assuming OpenCV is used)
        img = cv2.imread(full_image_path)
        if img is not None:
            img = cv2.resize(img, img_size)

            # Append the image and its corresponding label (food_id)
            images.append(img)
            labels.append(food_ids[idx])

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=42)

    # Save preprocessed images (this is just an example, update to match your image saving method)
    save_images(X_train, y_train, os.path.join(output_dir, 'train'))
    save_images(X_val, y_val, os.path.join(output_dir, 'val'))

    print(f"Preprocessing completed. Train set: {len(X_train)}, Validation set: {len(X_val)}")


if __name__ == '__main__':
    # Load food data from the training .mat file
    food_ids, food_names, food_images = load_food_data(train_mat_file_path)

    # Preprocess images and split into train and validation sets
    preprocess_images(food_ids, food_images, food_names, output_dir, img_size, test_size)