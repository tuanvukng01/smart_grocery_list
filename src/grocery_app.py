import os
import scipy.io
from infer import load_model, predict


# Function to load the food names and IDs from the .mat file
def load_food_names(mat_file_path):
    """
    Loads the food names and IDs from a .mat file and returns a dictionary.

    :param mat_file_path: Path to the .mat file
    :return: Dictionary mapping food IDs to names
    """
    mat = scipy.io.loadmat(mat_file_path)

    # Assuming the structure contains 'name' and 'id' fields
    food_names = mat['name']  # Extract food names
    food_ids = mat['id']  # Extract corresponding food ids

    # Convert these arrays to a dictionary mapping IDs to names
    food_dict = {food_ids[i][0]: food_names[i][0] for i in range(len(food_ids))}

    return food_dict


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
        predicted_class = predict(self.model, img_path, img_size=(224, 224))
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


if __name__ == '__main__':
    # Load food names and IDs from the training .mat file
    mat_file_path = 'path_to_training_mat_file.mat'  # Update with the correct path to your .mat file
    food_dict = load_food_names(mat_file_path)

    # Initial grocery list
    initial_grocery_list = ['apple', 'banana', 'orange']

    # Initialize the app with the model path, grocery list, and food dictionary
    app = GroceryApp(model_path='models/saved_model/food_model.h5', grocery_list=initial_grocery_list,
                     food_dict=food_dict)

    # Test with an image path
    test_image_path = 'path_to_test_image.jpg'  # Update with the path to your test image
    app.update_grocery_list(test_image_path)