import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, warnings
import logging
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2


def test_image(image_path):
    image = Image.open(image_path)
    # image = image.resize((300, 300))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    desired_size = (300, 300)
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = desired_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = desired_size[1]
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height))
    padded_image = Image.new('RGB', desired_size, (0, 0, 0))
    paste_location = ((desired_size[0] - new_width) // 2, (desired_size[1] - new_height) // 2)
    padded_image.paste(resized_image, paste_location)

    image = padded_image

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    prediction = np.max(prediction)
    class_labels = ['Horse', 'Human']
    index = 0
    percentage = 100 - prediction*100

    if prediction >= 0.5:
        index = 1
        percentage = prediction*100

    if index == 0:
        horse_conf.append(percentage)
    else:
        human_conf.append(percentage)

    print(f"Prediction ({image_path}): {class_labels[index]} with {percentage:.2f}% confidence")


def test_folder(folder_path):
    image_files = os.listdir(folder_path)
    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        test_image(image_path)


model = tf.keras.models.load_model('saved_models/vgg16_horse_or_human_SGD')

folder_path = 'test_images'
image_path = 'test_images/test_sandry.jpg'

horse_conf = []
human_conf = []

test_folder(folder_path)

print(['%.2f' % elem for elem in horse_conf])
print(['%.2f' % elem for elem in human_conf])