import os
import scipy.io
import tensorflow as tf
from infer import load_model, predict, preprocess_image

# Path to the saved model
model_path = 'models/saved_model/food_model.h5'

# Image size (must match the size used during training)
img_size = (224, 224)


# Function to load food names and IDs from the .mat file
def load_food_names(mat_file_path):
    """
    Loads the food names and IDs from a .mat file and returns a dictionary.

    :param mat_file_path: Path to the .mat file
    :return: Dictionary mapping food IDs to names
    """
    mat = scipy.io.loadmat(mat_file_path)
    food_names = mat['name']  # Extract food names
    food_ids = mat['id']  # Extract corresponding food ids
    food_dict = {food_ids[i][0]: food_names[i][0] for i in range(len(food_ids))}
    return food_dict


# Function to load the test data (for evaluation)
def load_test_data(mat_file_path):
    """
    Loads the test data (image paths and corresponding labels) from the .mat file.

    :param mat_file_path: Path to the .mat file containing test data
    :return: Tuple (test_ids, test_images) where test_images are file paths to images
    """
    mat = scipy.io.loadmat(mat_file_path)
    test_ids = mat['id']  # Extract food IDs
    test_images = mat['images']  # Extract image file paths
    test_ids = [test_id[0] for test_id in test_ids]
    test_images = [img[0] for img in test_images]
    return test_ids, test_images


# GroceryApp class to manage grocery list updates
class GroceryApp:
    def __init__(self, model_path, grocery_list, food_dict):
        """
        Initializes the GroceryApp.

        :param model_path: Path to the trained model
        :param grocery_list: Initial grocery list of items
        :param food_dict: Dictionary mapping class IDs to food names
        """
        self.model = load_model(model_path)
        self.grocery_list = grocery_list
        self.food_dict = food_dict

    def update_grocery_list(self, img_path):
        """
        Predicts the item in the image and updates the grocery list by removing the recognized item.

        :param img_path: Path to the image file to be processed
        """
        predicted_class = predict(self.model, img_path, img_size)
        item_name = self.class_to_item(predicted_class)
        if item_name in self.grocery_list:
            self.grocery_list.remove(item_name)
        print(f"Updated grocery list: {self.grocery_list}")

    def class_to_item(self, class_idx):
        """
        Maps a class index from the model's prediction to a food name.

        :param class_idx: Class index predicted by the model
        :return: The corresponding food name or "Unknown" if not found
        """
        return self.food_dict.get(class_idx, "Unknown")


# Function to evaluate the grocery app using test data
def evaluate_grocery_app(app, test_images, test_ids):
    """
    Evaluates the grocery app on test images and prints the updated grocery list.

    :param app: Instance of GroceryApp
    :param test_images: List of test image paths
    :param test_ids: List of correct labels for the test images
    """
    for idx, img_path in enumerate(test_images):
        print(f"Processing image: {img_path}")
        app.update_grocery_list(img_path)
        predicted_class = predict(app.model, img_path, img_size)
        actual_class = test_ids[idx]
        print(f"Predicted class: {predicted_class}, Actual class: {actual_class}")


if __name__ == '__main__':
    # Load food names and IDs from the training .mat file
    mat_file_path = 'path_to_training_mat_file.mat'  # Update with the correct path
    food_dict = load_food_names(mat_file_path)

    # Initial grocery list
    initial_grocery_list = ['apple', 'banana', 'orange']  # Modify this as necessary

    # Initialize the GroceryApp with the model and food dictionary
    app = GroceryApp(model_path=model_path, grocery_list=initial_grocery_list, food_dict=food_dict)

    # Load test data from the test .mat file
    test_mat_file_path = 'path_to_test_mat_file.mat'  # Update this with the correct path to your test .mat file
    test_ids, test_images = load_test_data(test_mat_file_path)

    # Evaluate the grocery app on the test images
    evaluate_grocery_app(app, test_images, test_ids)