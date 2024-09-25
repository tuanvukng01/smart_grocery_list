import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Path to the saved PyTorch model
model_path = 'models/saved_model/uecfood_model.pth'

# Image size (must match the size used during training)
img_size = (224, 224)

# Define transformations for inference (same as training)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Function to load the trained model
def load_model(model_path, num_classes):
    """
    Loads the trained PyTorch model.

    :param model_path: Path to the saved model
    :param num_classes: Number of output classes
    :return: Loaded PyTorch model
    """
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model


# Function to preprocess an image for inference
def preprocess_image(img_path):
    """
    Preprocesses an image for PyTorch model input.

    :param img_path: Path to the image
    :return: Preprocessed image tensor
    """
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img


# Function to predict the class of an image
def predict(model, img_path):
    """
    Predicts the class of the given image using the trained PyTorch model.

    :param model: Trained PyTorch model
    :param img_path: Path to the image to be classified
    :return: Predicted class index
    """
    img = preprocess_image(img_path)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return predicted.item()


# Function to evaluate the model on the test set
def evaluate_model(model, test_images, test_labels):
    """
    Evaluates the model on the test set.

    :param model: Trained PyTorch model
    :param test_images: List of test image paths
    :param test_labels: List of correct labels for the test images
    :return: Accuracy of the model on the test set
    """
    correct_predictions = 0
    total_images = len(test_images)

    for idx, img_path in enumerate(test_images):
        predicted_class = predict(model, img_path)
        if predicted_class == test_labels[idx]:
            correct_predictions += 1

    accuracy = correct_predictions / total_images
    print(f"Test set accuracy: {accuracy * 100:.2f}%")

    return accuracy


if __name__ == '__main__':
    # Load the trained model
    num_classes = 256  # Adjust to the number of classes in your dataset
    model = load_model(model_path, num_classes)

    # Load the test data
    test_images = ['path_to_test_image_1.jpg', 'path_to_test_image_2.jpg']  # Update with actual image paths
    test_labels = [1, 5]  # Update with actual labels

    # Evaluate the model on the test set
    evaluate_model(model, test_images, test_labels)