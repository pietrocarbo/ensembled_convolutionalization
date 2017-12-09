import os
import sys
import signal
import time

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model

from lib.plot_utils import save_acc_loss_plots
from lib.randomization import lower_randomization_effects
from lib.callbacks import checkpointer, early_stopper, lr_reducer, csv_logger

lower_randomization_effects()

from keras.applications.resnet50 import preprocess_input
model_name = 'resnet50'
model_n_layers = 168
num_classes = 101
base_model = keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling=None, classes=num_classes)

# 76% - 88 layers - base_model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=num_classes)
# base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=num_classes)

# base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=num_classes)
# base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=num_classes)
# base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=num_classes)

# memory management
# cpu_parallelism = True
# from keras import backend as K
# if not cpu_parallelism:
#     session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# else:
#     session_conf = tf.ConfigProto()
# session_conf.gpu_options.allow_growth = True
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

batch_size = 32
IMG_WIDTH = 224
IMG_HEIGHT = 224

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        'dataset-ethz101food/train',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,       # larger batch size when training bottom, smaller training upper layers ?
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'dataset-ethz101food/test',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical')

last_layer = base_model.output
x = GlobalAveragePooling2D()(last_layer)

# x = Dense(512, activation='relu', name='fc-1')(x)
# x = Dropout(0.5)(x)
# x = Dense(256, activation='relu', name='fc-2')(x)
# x = Dropout(0.5)(x)
# out = Dense(num_classes, activation='softmax', name='output_layer')(x)

# x = Dense(512, activation='relu', name='fc-1')(x)
# out = Dense(num_classes, activation='softmax', name='output_layer')(x)

out = Dense(num_classes, activation='softmax', name='output_layer')(x)

custom_model = Model(inputs=base_model.input, outputs=out)
# print(custom_model.summary())


def train_top_n_layers(model, n, epochs, optimizer, callbacks=None, train_steps=None, val_steps=None, test_epoch_end=False):
    for i in range(len(model.layers)):
        if i < n:
            model.layers[i].trainable = False
        else:
            model.layers[i].trainable = True

    custom_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    start = time.time()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps or 75750 // batch_size,
                                  epochs=epochs, verbose=1,
                                  validation_data=validation_generator,
                                  validation_steps=val_steps or 25250 // batch_size,
                                  callbacks=callbacks,
                                  use_multiprocessing=False)
    print('Training time {0:.2f} minutes'.format(-(start - time.time()) / 60))

    if test_epoch_end:
        (loss, accuracy) = model.evaluate_generator(validation_generator, 250 // 32)
        print("[EVAL] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    return history


def close_signals_handler(signum, frame):
    sys.stdout.flush()
    print('\n\nReceived KeyboardInterrupt (CTRL-C), preparing to exit')
    save_acc_loss_plots(histories,
                        os.path.join(os.getcwd(), 'results', plot_acc_file),
                        os.path.join(os.getcwd(), 'results', plot_loss_file))
    sys.exit(1)


# filenames
logfile = model_name + '_ft_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
checkpoints_filename = model_name + '_ft_weights_acc{val_acc:.2f}_e{epoch:d}_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.hdf5'
plot_acc_file = model_name + '_ft_acc' + time.strftime("%Y-%m-%d_%H-%M-%S")
plot_loss_file = model_name + '_ft_loss' + time.strftime("%Y-%m-%d_%H-%M-%S")

# optimizers
sgd = keras.optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
adam = 'adam'
rmsprop = 'rmsprop'

# callbacks
stopper = early_stopper(monitor='val_loss', patience=2)
lr_reduce = lr_reducer(factor=0.1, patience=1)
model_saver = checkpointer(checkpoints_filename)
logger = csv_logger(logfile)

# training parameters
train_steps = None  # None to consider all the training set / 1 to test stuff
val_steps = None  # None to consider all the validation set / 1 to test stuff
epochs_fc = 5000  # 5000
epochs_ft = 2000  # 2000
ft_granularity = 24

signal.signal(signal.SIGTERM, close_signals_handler)
signal.signal(signal.SIGINT, close_signals_handler)
train_time = time.time()

histories = [train_top_n_layers(custom_model, 6, epochs_fc, rmsprop, [stopper, logger], train_steps, val_steps)]
for i in range(6+ft_granularity, 6+model_n_layers+1, ft_granularity):
    histories.append(train_top_n_layers(custom_model, i, epochs_ft, adam, [stopper, logger, model_saver], train_steps, val_steps))

print('Total training time {0:.2f} minutes'.format(-(train_time - time.time()) / 60))
save_acc_loss_plots(histories,
                    os.path.join(os.getcwd(), 'results', plot_acc_file),
                    os.path.join(os.getcwd(), 'results', plot_loss_file))
