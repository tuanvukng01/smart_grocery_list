import torch
from torchvision import transforms
from PIL import Image
import os

# Path to the saved model
model_path = 'models/saved_model/uecfood_model.pth'

# Image size
img_size = (224, 224)

# Define image transformation for inference
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Function to load and preprocess an image
def preprocess_image(img_path):
    """
    Preprocesses the image for PyTorch model input.

    :param img_path: Path to the image
    :return: Preprocessed image tensor
    """
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img


# Function to load the PyTorch model
def load_model(model_path, num_classes):
    """
    Loads the trained PyTorch model.

    :param model_path: Path to the saved model
    :param num_classes: Number of output classes
    :return: Loaded PyTorch model
    """
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Function to make predictions
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


if __name__ == '__main__':
    num_classes = 256  # Adjust to the number of classes in your dataset
    model = load_model(model_path, num_classes)

    img_path = 'path_to_test_image.jpg'  # Update with the path to your test image
    predicted_class = predict(model, img_path)
    print(f"Predicted class: {predicted_class}")