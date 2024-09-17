import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
img_size = (224, 224)
batch_size = 32
epochs = 10
data_dir = 'data/processed/'
model_save_path = 'models/saved_model/food_model.h5'


def build_model(num_classes = 475):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(data_dir, model_save_path):
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(os.path.join(data_dir, 'train'), target_size=img_size,
                                                        batch_size=batch_size, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(os.path.join(data_dir, 'val'), target_size=img_size,
                                                    batch_size=batch_size, class_mode='categorical')

    model = build_model(num_classes=len(train_generator.class_indices))

    model.fit(train_generator, validation_data=val_generator, epochs=epochs,
              steps_per_epoch=train_generator.samples // batch_size,
              validation_steps=val_generator.samples // batch_size)

    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")


if __name__ == '__main__':
    train_model(data_dir, model_save_path)