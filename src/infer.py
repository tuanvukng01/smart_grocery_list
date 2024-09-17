import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model_path = 'models/saved_model/food_model.h5'
img_size = (224, 224)


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict(model, img_path, img_size):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class


if __name__ == '__main__':
    model = load_model(model_path)
    test_image_path = 'path_to_test_image.jpg'
    predicted_class = predict(model, test_image_path, img_size)
    print(f"Predicted class: {predicted_class}")