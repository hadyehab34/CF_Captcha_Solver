from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import tensorflow as tf
from tensorflow import keras
import numpy as np

downsample_factor = 2

img_width = 270 // downsample_factor
img_height = 80 // downsample_factor

# Maximum length of any captcha in the dataset
max_length = 6

# Define the vocabulary set
vocabulary_list = ['T', '4', 'k', 'e', 'b', 'o', 'A', 'z', 'C', 'v', 'D', 'E', 'i', 'V', 'L', 'X', 'W', '2', 's', 'j', 't', 'U', 'O', 'N', 'd', '3', 'I', '0', 'K', 'H', 'h', 'P', '6', '8', 'f', 'g', 'J', 'p', '7', 'S', 'n', 'q', '1', 'm', 'r', 'w', 'G', 'a', 'Z', '9', 'B', 'y', 'l', 'Q', 'M', 'x', 'u', 'F', 'R', 'c', 'Y', '5']

# Convert the set to a sorted list to ensure the order is consistent
# vocabulary_list = sorted(vocabulary_set)

# Create the StringLookup layer
char_to_num = StringLookup(vocabulary=vocabulary_list, num_oov_indices=0, mask_token=None)

# Optionally, if you need the inverse mapping
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), num_oov_indices=0, mask_token=None, invert=True)

# Define augmentation parameters using tf.image functions
def augment_image(img):
    img = tf.image.random_flip_left_right(img)  # Random horizontal flip
    img = tf.image.random_flip_up_down(img)  # Random vertical flip
    img = tf.image.random_brightness(img, max_delta=0.1)  # Random brightness adjustment
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)  # Random contrast adjustment

    # # Random zoom
    # img = tf.image.resize_with_crop_or_pad(img, img_height + 6, img_width + 6)  # Add padding
    # img = tf.image.random_crop(img, size=[img_height, img_width, 1])  # Random crop
    return img

# Define encoding functions
def encode_single_sample(img_path, label, img_width=img_width, img_height=img_height):
    img_path = tf.strings.join([img_path])  # Ensure img_path is a string

    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])

    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


# Define encoding functions
def encode_single_sample_pred(img_path, img_width=img_width, img_height=img_height):
    img_path = tf.strings.join([img_path])  # Ensure img_path is a string

    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img}

# Define function to encode augmented samples
def encode_augmented_samples(img_path, label, num_augmented_versions=3, img_width=img_width, img_height=img_height):
    img_path = tf.strings.join([img_path])  # Ensure img_path is a string

    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Create multiple augmented versions
    augmented_images = []
    for _ in range(num_augmented_versions):
        augmented_img = augment_image(img)
        augmented_images.append(augmented_img)
    augmented_images = tf.stack(augmented_images)

    # 6. Transpose the images because we want the time dimension to correspond to the width of the image.
    augmented_images = tf.transpose(augmented_images, perm=[0, 2, 1, 3])

    # 7. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    # 8. Return a dict as our model is expecting two inputs
    return {"image": augmented_images, "label": tf.tile(tf.expand_dims(label, axis=0), [num_augmented_versions, 1])}



# Utility function to decode the output of the network for a single sample
def decode_single_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, : 6]
    
    # Decode the prediction
    decoded_string = tf.strings.reduce_join(num_to_char(results[0])).numpy().decode("utf-8")
    
    return decoded_string