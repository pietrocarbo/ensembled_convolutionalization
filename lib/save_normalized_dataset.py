import os
import sys
import signal
import time
import json

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

IMG_WIDTH = 299
IMG_HEIGHT = 299
batch_size = 32

datagen = ImageDataGenerator(dict(featurewise_center=True, featurewise_std_normalization=True))

images = []
for category in os.listdir(os.path.join("..", "dataset-ethz101food", "train")):
    for file in os.listdir(os.path.join("..", "dataset-ethz101food", "train", category)):
        img = image.load_img(os.path.join("..", "dataset-ethz101food", "train", category, file))
        img = image.img_to_array(img)
        images.append(img)

datagen.fit(images)

datagen_iterator = datagen.flow_from_directory(
    '../dataset-ethz101food/test',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical',
    save_to_dir='../dataset-ethz101food/augmented', save_prefix='aug', save_format='png')

for X_batch, y_batch in datagen_iterator:
    pass
