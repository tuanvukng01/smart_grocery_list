import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths and configurations
img_size = (224, 224)  # Image size for the input
batch_size = 32  # Number of images to process in each batch
epochs = 10  # Number of epochs to train
train_dir = 'data/processed/train/'  # Directory containing training images
val_dir = 'data/processed/val/'  # Directory containing validation images
model_save_path = 'models/saved_model/food_model.h5'  # Path to save the trained model


# Build the MobileNetV2 model for transfer learning
def build_model(num_classes):
    """
    Builds the MobileNetV2 transfer learning model with added dense layers for classification.

    :param num_classes: Number of output classes
    :return: Compiled Keras model
    """
    # Load the MobileNetV2 model pre-trained on ImageNet, exclude the top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Pooling layer to reduce dimensionality
    x = Dense(1024, activation='relu')(x)  # Dense layer for added learning capacity
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer for classification

    # Define the complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers in the base model to prevent them from being trained
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Function to train the model
def train_model(train_dir, val_dir, model_save_path):
    """
    Trains the model on the dataset and saves it.

    :param train_dir: Directory containing training data
    :param val_dir: Directory containing validation data
    :param model_save_path: Path to save the trained model
    """
    # Data augmentation and normalization for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalize pixel values
        rotation_range=20,  # Augment data with random rotations
        width_shift_range=0.2,  # Augment data with random width shifts
        height_shift_range=0.2,  # Augment data with random height shifts
        shear_range=0.2,  # Augment data with random shearing
        zoom_range=0.2,  # Augment data with random zooms
        horizontal_flip=True  # Augment data with horizontal flips
    )

    # Only normalization for validation
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load training and validation data from the directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Get the number of classes based on the training data
    num_classes = len(train_generator.class_indices)

    # Build the model
    model = build_model(num_classes)

    # Train the model
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=val_generator.samples // batch_size
    )

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")


if __name__ == '__main__':
    # Train the model using the processed dataset
    train_model(train_dir, val_dir, model_save_path)