import os
import time
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model
from keras import Sequential
from keras import optimizers

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

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

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
epochs = 50

# base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=num_classes)
base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=num_classes)

last_layer = base_model.output
x = GlobalAveragePooling2D()(last_layer)

x = Dense(512, activation='relu', name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', name='fc-2')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)

custom_model = Model(inputs=base_model.input, outputs=out)

for layer in custom_model.layers[:-6]:
    layer.trainable = False

print(custom_model.summary())

custom_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

start = time.time()
hist = custom_model.fit_generator(train_generator, steps_per_epoch=750 // 32, epochs=epochs, verbose=1, validation_data=validation_generator, validation_steps=250 // 32)
print('Total training time {0:.2f} minutes'.format(-(start - time.time()) / 60))

print('[HISTORY]', hist.history)

(loss, accuracy) = custom_model.evaluate_generator(validation_generator, 250 // 32)
print("[EVAL] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

custom_model.save(os.path.join(os.getcwd(), 'models', 'vgg16_baseline_rmsprop_' + str(epochs) + 'e_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.h5'))