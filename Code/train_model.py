"""
3D Print Failure Classifier - Training Script
Mac M2 Pro + TensorFlow Metal GPU

Classes:
  spaghetti_failure, warping, first_layer_failure, under_extrusion,
  stringing, blob_or_zit, layer_shift, nozzle_clog, good_print

Dataset layout expected:
  dataset/
    spaghetti_failure/  *.jpg / *.png
    warping/
    first_layer_failure/
    ...
"""

import os
import json
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURATION  (edit these paths if needed)
# ─────────────────────────────────────────────
DATA_DIR    = pathlib.Path("./dataset")   # one sub-folder per class
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
MODEL_OUT   = "print_failure_model.keras"
NAMES_OUT   = "class_names.json"
# ─────────────────────────────────────────────

def build_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    return train_ds, val_ds

def augment_and_normalize(train_ds, val_ds):
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ])

    normalize = layers.Rescaling(1.0 / 255.0)

    def train_prep(x, y):
        x = normalize(x)
        x = augment(x, training=True)
        return x, y

    def val_prep(x, y):
        return normalize(x), y

    train_ds = train_ds.map(train_prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(val_prep,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def build_model(num_classes):
    base = keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    # Phase 1: freeze base, train head only
    base.trainable = False

    inputs  = keras.Input(shape=IMG_SIZE + (3,))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base

def plot_history(history, phase="phase1"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="train acc")
    axes[0].plot(history.history["val_accuracy"],  label="val acc")
    axes[0].set_title(f"Accuracy ({phase})")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="train loss")
    axes[1].plot(history.history["val_loss"],  label="val loss")
    axes[1].set_title(f"Loss ({phase})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"training_curve_{phase}.png")
    print(f"  Saved training_curve_{phase}.png")

def main():
    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n[INFO] Available GPUs: {gpus}")
    if not gpus:
        print("[WARN] No GPU found — training will use CPU (slower).")
    print()

    # Data
    if not DATA_DIR.exists():
        print(f"[ERROR] dataset folder '{DATA_DIR}' not found.")
        print("  Create it with sub-folders named after each class, e.g.:")
        print("  dataset/spaghetti_failure/   dataset/warping/   ...")
        return

    train_ds, val_ds = build_datasets()
    class_names = train_ds.class_names
    print(f"[INFO] Classes ({len(class_names)}): {class_names}\n")

    # Save class names for the inference script
    with open(NAMES_OUT, "w") as f:
        json.dump(class_names, f)

    train_ds, val_ds = augment_and_normalize(train_ds, val_ds)

    # Model
    model, base = build_model(len(class_names))

    # Phase 1: head only (fast warm-up)
    print("=== Phase 1: training classification head ===")
    model.compile(
        optimizer=keras.optimizers.Adam(LR * 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=5,
                   callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    plot_history(h1, "phase1")

    # Phase 2: fine-tune entire network
    print("\n=== Phase 2: fine-tuning full network ===")
    base.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    plot_history(h2, "phase2")

    # Save
    model.save(MODEL_OUT)
    print(f"\n[INFO] Model saved → {MODEL_OUT}")
    print(f"[INFO] Class names → {NAMES_OUT}")

    # Quick eval
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n[RESULT] Validation accuracy: {acc:.2%}  |  loss: {loss:.4f}")

if __name__ == "__main__":
    main()
