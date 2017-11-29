import os
import math
import time
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import SGD

from sys import platform
if platform == "darwin":
    import matplotlib as mpl
    mpl.use('TkAgg')

def disable_randomization_effects():
    from keras import backend as K
    import random as rn
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.allow_growth = True
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


disable_randomization_effects()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

train_datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        'dataset-ethz101food/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'dataset-ethz101food/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

num_classes = 101

early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
reduce_lr_plateu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
checkpoint_model = keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), 'models', 'mobilenet_finetuned'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

base_model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=num_classes)

last_layer = base_model.output
x = GlobalAveragePooling2D()(last_layer)

x = Dense(512, activation='relu', name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', name='fc-2')(x)
x = Dropout(0.4)(x)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)

custom_model = Model(inputs=base_model.input, outputs=out)
print(custom_model.summary())


def train_top_n_layers(model, n, epochs, optimizer, callbacks):
    for i in range(len(model.layers)):
        if i < n:
            model.layers[i].trainable = False
        else:
            model.layers[i].trainable = True

    custom_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    start = time.time()
    history = model.fit_generator(train_generator, steps_per_epoch=750 // 32, epochs=epochs, verbose=1,
                                       validation_data=validation_generator, validation_steps=250 // 32,
                                       callbacks=callbacks, use_multiprocessing=False)
    print('Training time {0:.2f} minutes'.format(-(start - time.time()) / 60))

    (loss, accuracy) = model.evaluate_generator(validation_generator, 250 // 32)
    print("[EVAL] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    return history

histories = []
histories.append(train_top_n_layers(custom_model, 6, 100, 'rmsprop', [early_stopping, reduce_lr_plateu]))
histories.append(train_top_n_layers(custom_model, 12, 100, 'rmsprop', [early_stopping, reduce_lr_plateu]))
histories.append(train_top_n_layers(custom_model, 18, 100, 'rmsprop', [early_stopping, reduce_lr_plateu, checkpoint_model]))

plt.style.use(['classic'])
fig = plt.figure('Validation Loss/Accurancy')

training_steps = len(histories)

for i in range(training_steps):
    val_acc = histories[i].history['val_acc']
    val_loss = histories[i].history['val_loss']
    x = range(len(val_acc))

    ax = fig.add_subplot(1, training_steps, i+1)
    ax.plot(x, val_loss)
    ax.plot(x, val_acc)
    ax.set_xlabel('Epochs')
    ax.grid(True)
    ax.set_title('Training step ' + str(i+1))
    ax.legend(['loss', 'acc'], loc=0)

fig.subplots_adjust(hspace=.5)
plt.show()

