import os
import keras
from keras import layers
from tensorflow import data as tf_data

# Function to delete corrupted images
def clean_data(directory):
    num_skipped = 0
    for folder_name in ("Cancer", "Non_Cancer"):
        folder_path = os.path.join(directory, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
    print(f"Deleted {num_skipped} corrupted images from {directory}")

# Function to create datasets
def create_datasets(base_dir, image_size=(180, 180), batch_size=128, validation_split=0.2, seed=1337):
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        base_dir,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )
    return train_ds, val_ds

# Data augmentation layers
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

# Function to apply data augmentation
def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Function to prepare datasets with augmentation and prefetching
def prepare_datasets(train_ds, val_ds):
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    return train_ds, val_ds

