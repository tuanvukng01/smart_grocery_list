import os
from infer import load_model, predict


class GroceryApp:
    def __init__(self, model_path, grocery_list):
        self.model = load_model(model_path)
        self.grocery_list = grocery_list

    def update_grocery_list(self, img_path):
        predicted_class = predict(self.model, img_path, img_size=(224, 224))
        item_name = self.class_to_item(predicted_class)
        if item_name in self.grocery_list:
            self.grocery_list.remove(item_name)
        print(f"Updated grocery list: {self.grocery_list}")

    def class_to_item(self, class_idx):
        # Map class index to actual item names
        class_to_item_map = {0: 'apple', 1: 'banana', 2: 'orange', ...}  # Complete this with all mappings
        return class_to_item_map.get(class_idx, "Unknown")


if __name__ == '__main__':
    initial_grocery_list = ['apple', 'banana', 'orange']
    app = GroceryApp(model_path='models/saved_model/food_model.h5', grocery_list=initial_grocery_list)
    test_image_path = 'path_to_test_image.jpg'
    app.update_grocery_list(test_image_path)