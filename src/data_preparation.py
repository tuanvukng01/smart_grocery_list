import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Paths
data_dir = 'path_to_UECFOOD256/'  # Path to your UECFOOD256 dataset directory
batch_size = 32  # Batch size for DataLoader
test_size = 0.2  # Fraction of data for validation
img_size = (224, 224)  # Image size for resizing

# Define transformations for training and validation sets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Function to load datasets and create PyTorch DataLoaders
def create_dataloaders(data_dir, test_size, batch_size):
    """
    Loads the UECFOOD256 dataset, splits it into training and validation sets,
    and creates DataLoaders for PyTorch.

    :param data_dir: Path to the UECFOOD256 dataset
    :param test_size: Fraction of data for validation
    :param batch_size: Batch size for the DataLoader
    :return: DataLoaders for training and validation sets
    """
    # Load the entire dataset using ImageFolder, without applying transformations yet
    full_dataset = datasets.ImageFolder(data_dir)

    # Split dataset into training and validation
    train_size = int((1 - test_size) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply the transformations after the split
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == '__main__':
    # Create DataLoaders for training and validation sets
    train_loader, val_loader = create_dataloaders(data_dir, test_size, batch_size)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")