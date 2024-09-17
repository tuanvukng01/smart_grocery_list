import os
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Path to the saved model
model_path = 'models/saved_model/food_model.h5'

# Image size (must match the size used during training)
img_size = (224, 224)

# Function to load the test data from the test .mat file
def load_test_data(mat_file_path):
    """
    Loads the test data (image paths and corresponding labels) from the .mat file.

    :param mat_file_path: Path to the .mat file containing test data
    :return: Tuple (test_ids, test_images) where test_images are file paths to images
    """
    mat = scipy.io.loadmat(mat_file_path)

    # Extract relevant data
    test_ids = mat['id']  # Extract food IDs
    test_images = mat['images']  # Extract image file paths

    # Convert the extracted data into usable formats
    test_ids = [test_id[0] for test_id in test_ids]
    test_images = [img[0] for img in test_images]

    return test_ids, test_images

# Function to load the trained model
def load_model(model_path):
    """
    Loads the saved Keras model for inference.

    :param model_path: Path to the saved model
    :return: Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)

# Function to preprocess a single image for inference
def preprocess_image(img_path, img_size):
    """
    Loads and preprocesses an image for model inference.

    :param img_path: Path to the image file
    :param img_size: Target size for the image
    :return: Preprocessed image ready for model input
    """
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

# Function to run inference on a single image
def predict(model, img_path, img_size):
    """
    Predicts the class of the given image using the trained model.

    :param model: Trained Keras model
    :param img_path: Path to the image to be classified
    :param img_size: Target image size for the model
    :return: Predicted class index
    """
    img = preprocess_image(img_path, img_size)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return predicted_class

# Function to evaluate the model on the test set
def evaluate_model(model, test_images, test_ids, img_size):
    """
    Evaluates the model on the test set and calculates accuracy.

    :param model: Trained Keras model
    :param test_images: List of test image paths
    :param test_ids: List of correct labels for the test images
    :param img_size: Target image size for preprocessing
    :return: Accuracy of the model on the test set
    """
    correct_predictions = 0
    total_images = len(test_images)

    for idx, img_path in enumerate(test_images):
        predicted_class = predict(model, img_path, img_size)
        if predicted_class == test_ids[idx]:
            correct_predictions += 1

    accuracy = correct_predictions / total_images
    print(f"Test set accuracy: {accuracy * 100:.2f}%")

    return accuracy

if __name__ == '__main__':
    # Load the trained model
    model = load_model(model_path)

    # Load the test data from the test .mat file
    test_mat_file_path = 'path_to_test_mat_file.mat'  # Update this with the correct path to your test .mat file
    test_ids, test_images = load_test_data(test_mat_file_path)

    # Evaluate the model on the test set
    evaluate_model(model, test_images, test_ids, img_size)