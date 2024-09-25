import os
from torchvision import transforms, datasets
from torch.utils.data import random_split
from PIL import Image

# Paths
data_dir = 'path_to_UECFOOD256/'  # Path to your UECFOOD256 dataset directory
output_dir = 'data/processed/'  # Directory to save processed images
img_size = (224, 224)  # Image size for resizing
test_size = 0.2  # Fraction of data for validation

# Define transformations for training and validation sets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ]),
}

# Function to save images to appropriate folders
def save_image(image_tensor, path):
    """Convert the image tensor back to a PIL image and save it."""
    image = transforms.ToPILImage()(image_tensor)  # Convert back to PIL image
    image.save(path)

def split_and_save_images(data_dir, output_dir, test_size):
    """
    Splits the dataset into training and validation sets, preprocesses images,
    and saves them into the appropriate directories.

    :param data_dir: Path to the UECFOOD256 dataset
    :param output_dir: Path to save processed images
    :param test_size: Fraction of data for validation
    """
    # Load the entire dataset using ImageFolder
    full_dataset = datasets.ImageFolder(data_dir)

    # Split dataset into training and validation sets
    train_size = int((1 - test_size) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create output directories
    train_output_dir = os.path.join(output_dir, 'train')
    val_output_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    # Save training images
    print(f"Saving {train_size} training images...")
    for idx, (image, label) in enumerate(train_dataset):
        class_name = full_dataset.classes[label]  # Get class name
        class_dir = os.path.join(train_output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        image_path = os.path.join(class_dir, f"image_{idx}.jpg")
        save_image(data_transforms['train'](image), image_path)

    # Save validation images
    print(f"Saving {val_size} validation images...")
    for idx, (image, label) in enumerate(val_dataset):
        class_name = full_dataset.classes[label]
        class_dir = os.path.join(val_output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        image_path = os.path.join(class_dir, f"image_{idx}.jpg")
        save_image(data_transforms['val'](image), image_path)

    print(f"Images successfully saved to {output_dir}")

if __name__ == '__main__':
    print("Starting data preprocessing and saving...")
    split_and_save_images(data_dir, output_dir, test_size)