import argparse
import os
import time

#Serve al mio mac <3
from sys import platform
if platform == "darwin":
    import matplotlib as mpl
    mpl.use('TkAgg')

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


def disable_randomization_effects():
    from keras import backend as K
    import numpy as np
    import random as rn
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def pause():
    programPause = input("Press <ENTER> to continue")


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(description='transfer learning from resnet50 on ethz101 food dataset')

parser.add_argument('-e', '--epochs', dest='epochs', default='50', type=int, help='Number of training epochs')

parser.add_argument('-ft', '--fine_tune', dest='ft', action='store_false',
                    help='Flag that indicates to NOT fine tune last 3 FC layers but just the final classifier layer')

args = parser.parse_args()

if args.epochs <= 0:
    parser.error("number of epochs MUST be bigger than 0")

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
fine_tuning = args.ft
epochs = args.epochs
print('fine tune:', str(fine_tuning), ', epochs:', epochs)


# fine tune the last layers of resnet-50
if fine_tuning:
    model = ResNet50(weights='imagenet', include_top=False)
    # model.summary()

    last_layer = model.output
    x = GlobalAveragePooling2D()(last_layer)

    x = Dense(512, activation='relu', name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc-2')(x)
    x = Dropout(0.5)(x)

    # a softmax layer for classification
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)

    # this is the model we will train
    custom_resnet_model = Model(inputs=model.input, outputs=out)

    for layer in custom_resnet_model.layers[:-6]:
        layer.trainable = False

    print(custom_resnet_model.summary())

    for layer in custom_resnet_model.layers[-8:]:
        print('trainable {}'.format(layer.trainable))

    custom_resnet_model.summary()

    custom_resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    start = time.time()
    hist = custom_resnet_model.fit_generator(train_generator, steps_per_epoch=750//32, epochs=epochs, verbose=1, validation_data=validation_generator, validation_steps=250//32)
    print('Total training time {0:.2f} minutes'.format(-(start - time.time()) / 60))

    print('[HISTORY]', hist.history)

    (loss, accuracy) = custom_resnet_model.evaluate_generator(validation_generator, 250//32)
    print("[EVAL] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

# train the classifier alone
else:
    image_input = Input(shape=(224, 224, 3))

    model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
    model.summary()

    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)

    custom_resnet_model = Model(inputs=image_input, outputs=out)
    custom_resnet_model.summary()

    for layer in custom_resnet_model.layers[:-1]:
        layer.trainable = False

    custom_resnet_model.layers[-1].trainable

    custom_resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    start = time.time()
    hist = custom_resnet_model.fit_generator(train_generator, steps_per_epoch=750//32, epochs=epochs, verbose=1, validation_data=validation_generator, validation_steps=250//32)
    print('Total training time {0:.2f} minutes'.format(-(start - time.time()) / 60))

    (loss, accuracy) = custom_resnet_model.evaluate_generator(validation_generator, 250//32)
    print("[EVAL] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

# save model on disk
fine_tuning = 'fineTuned3FC' if fine_tuning else 'justClassifierLayer'
custom_resnet_model.save(os.path.join(os.getcwd(), 'models', 'resnet50_' + fine_tuning + '_' + str(epochs) + 'e_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.h5'))

# visualizing loss and accuracy
xc = range(epochs)
train_acc = hist.history['acc']
train_loss = hist.history['loss']
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

plt.style.use(['classic'])

fig = plt.figure('Loss and Accurancy')

ax = fig.add_subplot(2, 1, 1)
ax.plot(xc, train_loss)
ax.plot(xc, val_loss)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.grid(True)
ax.set_title('training vs validation loss')
ax.legend(['training', 'validation'], loc=0)

ax = fig.add_subplot(2, 1, 2)
ax.plot(xc, train_acc)
ax.plot(xc, val_acc)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.grid(True)
ax.set_title('validation vs validation accurancy')
ax.legend(['training', 'validation'], loc=0)

fig.subplots_adjust(hspace=.5)
fig.savefig(os.path.join(os.getcwd(), 'results', 'resnet50_' + fine_tuning + '_' + str(epochs) + 'e32b_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.png'))
