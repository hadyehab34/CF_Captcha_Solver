from preprocess import *
import numpy as np
from collections import Counter
from pathlib import Path
import os

# Split data into training, validation, and test sets
def split_data(data_dir, codec='png', train_size=0.7, val_size=0.15, test_size=0.15, shuffle=True):
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"
    

    data_dir = Path(data_dir)
    images = data_dir.glob(f"*{codec}")
    # Get list of all the images
    images = (list(map(str, list(data_dir.glob(f"*.{codec}")))))
    labels = [img.split(os.path.sep)[-1].split(f".{codec}")[0] for img in images]

    assert len(images) == len(labels), "X and y must have the same length"

    X, y = np.array(images), np.array(labels)

    # Calculate sizes
    total_size = len(X)
    train_split = int(train_size * total_size)
    val_split = int(val_size * total_size)

    # Shuffle if needed
    if shuffle:
        indices = np.random.permutation(total_size)
        X = X[indices]
        y = y[indices]

    # Split data
    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:train_split + val_split], y[train_split:train_split + val_split]
    X_test, y_test = X[train_split + val_split:], y[train_split + val_split:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Split data into training, validation, and test sets
# X_train, y_train, X_val, y_val, X_test, y_test = split_data(np.array(images), np.array(labels))



# Create TensorFlow datasets for training, validation, and test sets
def create_dataset(X, y, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        dataset = (
            dataset.interleave(
                lambda img_path, label: tf.data.Dataset.from_tensor_slices(
                    encode_augmented_samples(img_path, label)
                ),
                cycle_length=tf.data.experimental.AUTOTUNE,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )
    else:
        dataset = dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    return (
        dataset
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

# Define batch size
# batch_size = 32

# # Create datasets for training, validation, and test sets
# train_dataset = create_dataset(X_train, y_train, batch_size, augment=False)
# val_dataset = create_dataset(X_val, y_val, batch_size, augment=False)  # No augmentation for validation
# test_dataset = create_dataset(X_test, y_test, batch_size, augment=False)  # No augmentation for test