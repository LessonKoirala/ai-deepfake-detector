import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.metrics import Precision, Recall

def train_deepfake_detector(data_dir, model_output_path, image_size=(224, 224), batch_size=32, epochs=15):
    """
    Builds a deepfake detector using transfer learning with MobileNetV2.
    It performs a two-stage training process:
    1. Trains only the new top layers.
    2. Fine-tunes the top portion of the base model with a lower learning rate.

    Args:
        data_dir (str): Path to the directory containing 'ai' and 'real' subdirectories.
        model_output_path (str): File path to save the trained model.
        image_size (tuple): The size of the input images for the model.
        batch_size (int): The number of samples per gradient update.
        epochs (int): The number of epochs for each training stage.
    """
    print("Starting model training process...")

    # Data Augmentation and Preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Stage 1: Build the model and train only the top layers
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(image_size[0], image_size[1], 3)
    )

    base_model.trainable = False  # freeze all layers initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Stage 1 compilation
    print("\nStage 1: Training top layers...")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name="precision"), Recall(name="recall")]
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_output_path, monitor='val_accuracy', save_best_only=True)  # save full model
    ]

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Stage 2: Fine-tuning the top layers of the base model
    print("\nStage 2: Fine-tuning the model...")
    base_model.trainable = True

    # Freeze most layers, unfreeze only last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name="precision"), Recall(name="recall")]
    )

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    print(f"\nModel training complete. Best model saved to {model_output_path}")


# Main block to run the script
if __name__ == "__main__":
    processed_data_dir = './processed_image_data'
    output_model_path = './deepfake_detector_model.keras'
    train_deepfake_detector(processed_data_dir, output_model_path)
