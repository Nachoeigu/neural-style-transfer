import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from functions import load_image

class StyleTransfer:

    def __init__(self, 
                 model_url='https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
                 ):
        self.model = hub.load(model_url)

    def producing_new_img(self, content_image_path, style_image_path):
        content_image = load_image(content_image_path)
        style_image = load_image(style_image_path)
        self.stylized_image = np.squeeze(self.model(tf.constant(content_image), tf.constant(style_image))[0])

    def saving_stylized_image(self, name, directory='output/'):
        cv2.imwrite(f'{directory}{name}.jpg', cv2.cvtColor(np.squeeze(self.stylized_image)*255, cv2.COLOR_BGR2RGB))
 

