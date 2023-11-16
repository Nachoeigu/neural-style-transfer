import tensorflow as tf
import os
from constants import IMAGE_EXTENSIONS

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def list_images_on_directory(directory):
    # List to hold image file names
    images = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                images.append(os.path.join(root, file))

    return images
