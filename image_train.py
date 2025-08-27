import os
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime

# ---------------- CONFIG ----------------
IMG_SIZE = (160, 160)      # small but efficient
BATCH_SIZE = 32
EPOCHS = 5                 # usually enough with transfer learning
DATA_DIR = "archive (1)"   # your dataset folder
MODEL_NAME = "ai_vs_human.keras"

# ---------------- DATA ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------- MODEL ----------------
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights="imagenet"
)
base_model.trainable = False  # freeze backbone

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ---------------- TRAIN ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ---------------- SAVE ----------------
os.makedirs("image_model_output", exist_ok=True)
model.save(os.path.join("image_model_output", MODEL_NAME))
print(f"\n✅ Model saved to image_model_output/{MODEL_NAME}")

# ---------------- EVALUATE ----------------
test_loss, test_acc, test_auc = model.evaluate(val_ds)
print(f"\nTest Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
