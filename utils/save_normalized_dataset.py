import os
import sys
import signal
import time
import json

import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import to_categorical

IMG_WIDTH = 299
IMG_HEIGHT = 299
batch_size = 32

datagen = ImageDataGenerator(dict(featurewise_center=True, featurewise_std_normalization=True))

images = []
for category in os.listdir(os.path.join("..", "dataset-ethz101food", "train")):
    for file in os.listdir(os.path.join("..", "dataset-ethz101food", "train", category)):
        img = Image.open(os.path.join("..", "dataset-ethz101food", "train", category, file))
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        img = image.img_to_array(img)
        images.append(img)

datagen.fit(images)
print("Dataset mean is " + datagen.mean + " std is " + datagen.std)
del images


i = 0
images = []
labels = []
for category in os.listdir(os.path.join("..", "dataset-ethz101food", "train")):
    images = []
    labels = []
    os.makedirs(os.path.join("..", "dataset-ethz101food", "augmented", category), exist_ok=True)
    for file in os.listdir(os.path.join("..", "dataset-ethz101food", "train", category)):
        img = Image.open(os.path.join("..", "dataset-ethz101food", "train", category, file))
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        img = image.img_to_array(img)
        images.append(img)
        labels.append(i)

    i += 1
    np_array = np.array(images)
    np_array = np_array.reshape(len(images), IMG_WIDTH, IMG_HEIGHT, 3)
    labels = to_categorical(labels, 101)

    datagen_iterator = datagen.flow(np_array, labels,
                                   batch_size=batch_size,
                                   save_to_dir=os.path.join("..", "dataset-ethz101food", "augmented", category), save_prefix='aug_', save_format='png')
    for X_batch, y_batch in datagen_iterator:
        pass
