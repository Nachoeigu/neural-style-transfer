from functions import list_images_on_directory
from model import StyleTransfer

content_images = list_images_on_directory(directory = 'content_images')
style_images = list_images_on_directory(directory = 'style_images')
model = StyleTransfer()

for n_image, content_image in enumerate(content_images):
    for n_style, style_image in enumerate(style_images):
        model.producing_new_img(content_image_path=content_image, style_image_path=style_image)
        model.saving_stylized_image(f'generated_image_{n_image}_{n_style}')
