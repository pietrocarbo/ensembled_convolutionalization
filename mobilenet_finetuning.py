import os
import time
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.mobilenet import preprocess_input

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

# train_datagen = ImageDataGenerator(rotation_range=25,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         rescale=1. / 255,
#         horizontal_flip=True,
#         fill_mode='nearest')
#
# test_datagen = ImageDataGenerator(rotation_range=25,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         rescale=1. / 255,
#         horizontal_flip=True,
#         fill_mode='nearest')

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

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


def checkpointer(filename):
    return keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), 'models', filename), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)


def early_stopper(monitor='val_acc', min_delta=0, patience=50):
    return keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode='auto')


def lr_reducer(monitor='val_loss', factor=0.1, patience=5):
    return keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)


def csv_logger(filename, separator=' ', append=True):
    return keras.callbacks.CSVLogger(os.path.join(os.getcwd(), 'logs', filename), separator=separator, append=append)


base_model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=num_classes)

last_layer = base_model.output
x = GlobalAveragePooling2D()(last_layer)

x = Dense(512, activation='relu', name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', name='fc-2')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)

custom_model = Model(inputs=base_model.input, outputs=out)
print(custom_model.summary())


def train_top_n_layers(model, n, epochs, optimizer, callbacks=None):
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


# optimizers
rmsprop = 'rmsprop'
sgd = keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

# callbacks to use
stopper = early_stopper(patience=20)
lr_reduce = lr_reducer()
model_saver = checkpointer('mobilenet_finetuned_{val_acc:.2f}_{epoch:d}_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.hdf5')
logger = csv_logger('mobilenet_finetuning_started_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.csv')

histories = []

train_time = time.time()
histories.append(train_top_n_layers(custom_model, 6, 1, rmsprop, [stopper, logger]))
histories.append(train_top_n_layers(custom_model, 18, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 24, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 30, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 36, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 42, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 48, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 54, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 60, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 66, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 72, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 78, 1, sgd, [stopper, lr_reduce, logger]))
histories.append(train_top_n_layers(custom_model, 84, 1, sgd, [stopper, lr_reduce, logger, model_saver]))
print('Total training time {0:.2f} minutes'.format(-(train_time - time.time()) / 60))

plt.style.use('seaborn-bright')
fig = plt.figure('Validation Loss/Accurancy')

training_steps = len(histories)
for i in range(training_steps):
    val_acc = histories[i].history['val_acc']
    train_acc = histories[i].history['acc']
    x = range(len(val_acc))

    ax = fig.add_subplot(1, training_steps, i+1)
    ax.plot(x, train_acc)
    ax.plot(x, val_acc)
    ax.set_xlabel('Epochs')
    ax.grid(True)
    ax.set_title('Training step ' + str(i+1))
    ax.legend(['train_acc', 'val_acc'], loc=0)

result_filename = os.path.join(os.getcwd(), 'results', 'mobilenet_finetuning_' + time.strftime("%Y-%m-%d_%H-%M-%S"))
fig.subplots_adjust(hspace=.5)
fig.savefig(result_filename)

