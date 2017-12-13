import os
import sys
import signal
import time
import json

import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model
from keras.regularizers import l2
from keras.activations import selu

from lib.plot_utils import save_acc_loss_plots
from lib.randomization import lower_randomization_effects
from lib.callbacks import checkpointer, early_stopper, lr_reducer, csv_logger
from lib.memory_management import memory_growth_config


lower_randomization_effects()
memory_growth_config()

from keras.applications.resnet50 import preprocess_input
model_name = 'xception'

base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# 77% - 1dense - 32bs - base_model = keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# 76% - 3dens - 32bs - base_model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet')
# 42% - 3dense - 32bs - base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

base_model_nlayers = len(base_model.layers)
num_classes = 101

base_model_output = base_model.output

dense3, denseLRBN, dense1, *_ = range(10)
top_net = denseLRBN

if top_net == dense3:
    x = GlobalAveragePooling2D()(base_model_output)
    x = Dense(512, activation='relu', name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc-2')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)
    topnn_nlayers = 6  # global pooling/dropout layers count as 1 layer

elif top_net == denseLRBN:
    x = GlobalAveragePooling2D()(base_model_output)
    x = Dense(4096, kernel_initializer='he_uniform', bias_initializer="he_uniform", activation=LeakyReLU(), kernel_regularizer=l2(.0005), bias_regularizer=l2(.0005))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(.4)(x)
    out = Dense(num_classes, kernel_initializer='he_uniform', bias_initializer="he_uniform", activation='softmax')(x)
    topnn_nlayers = 6

elif top_net == dense1:
    x = GlobalAveragePooling2D()(base_model_output)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)
    topnn_nlayers = 2

else:
    raise ValueError('Unspecified top neural network architecture')

custom_model = Model(inputs=base_model.input, outputs=out)
print('Custom model is \n' + custom_model.summary())

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32
IMG_WIDTH = 224
IMG_HEIGHT = 224

data_augmentation_level = 0  # 1, 2, ..

dict_augmentation = {"preprocessing_function": preprocess_input, "rescale": 1./255}

if data_augmentation_level > 0:
    dict_augmentation["horizontal_flip"] = True
    dict_augmentation["fill_mode"] = 'nearest'

if data_augmentation_level > 1:
    dict_augmentation["width_shift_range"] = 0.2
    dict_augmentation["height_shift_range"] = 0.2

if data_augmentation_level > 2:
    dict_augmentation["shear_range"] = 0.2
    dict_augmentation["zoom_range"] = 0.2

if data_augmentation_level > 3:
    dict_augmentation["rotation_range"] = 40

train_datagen = ImageDataGenerator(**dict_augmentation)
test_datagen = ImageDataGenerator(**dict_augmentation)


def train_top_n_layers(model, threshold_trainability, epochs, optimizer, batch_size=32, callbacks=None, train_steps=None, val_steps=None, test_epoch_end=False, max_queue_size=10):
    for i in range(len(model.layers)):
        if i < threshold_trainability:
            model.layers[i].trainable = False
        else:
            model.layers[i].trainable = True

    train_generator = train_datagen.flow_from_directory(
        'dataset-ethz101food/train',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'dataset-ethz101food/test',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical')
    print('Batch size is ' + str(batch_size))

    custom_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    keras.backend.get_session().run(tf.global_variables_initializer())

    start = time.time()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps or 75750 // batch_size,
                                  epochs=epochs, verbose=1,
                                  validation_data=validation_generator,
                                  validation_steps=val_steps or 25250 // batch_size,
                                  callbacks=callbacks,
                                  max_queue_size=max_queue_size)
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


def percentage(whole, part):
    return (whole * part) // 100


# filenames
model_arch_file = model_name + '_architecture_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.json'
logfile = model_name + '_ft_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
checkpoints_filename = model_name + '_ft_weights_acc{val_acc:.2f}_e{epoch:d}_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.hdf5'
plot_acc_file = model_name + '_ft_acc' + time.strftime("%Y-%m-%d_%H-%M-%S")
plot_loss_file = model_name + '_ft_loss' + time.strftime("%Y-%m-%d_%H-%M-%S")

# optimizers
rmsprop = 'rmsprop'
sgd = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
adam = 'adam'

# callbacks
stopper = early_stopper(monitor='val_loss', patience=5)
lr_reduce = lr_reducer(factor=0.1, patience=4)
model_saver = checkpointer(checkpoints_filename)
logger = csv_logger(logfile)

# training parameters
train_steps = None  # None to consider all the training set / 1 to test stuff
val_steps = None  # None to consider all the validation set / 1 to test stuff
epochs_fc = 500
epochs_ft = 200

signal.signal(signal.SIGTERM, close_signals_handler)
signal.signal(signal.SIGINT, close_signals_handler)
train_time = time.time()

with open(os.path.join(os.getcwd(), 'models', model_arch_file), 'w') as outfile:
    json.dump(json.loads(custom_model.to_json()), outfile, indent=2)

twopass, bottomup, *_ = range(10)
ft_type = twopass

if ft_type == twopass:
    histories = [train_top_n_layers(
        model=custom_model,
        threshold_trainability=topnn_nlayers,
        epochs=epochs_fc,
        optimizer=rmsprop,
        batch_size=batch_size,
        callbacks=[stopper, logger, model_saver],
        max_queue_size=10)]
    histories += train_top_n_layers(
        model=custom_model,
        threshold_trainability=percentage(base_model_nlayers+topnn_nlayers, 50),
        epochs=epochs_ft,
        optimizer=sgd,
        batch_size=percentage(batch_size, 80),
        callbacks=[stopper, logger, model_saver],
        max_queue_size=10)

elif ft_type == bottomup:
    histories = [train_top_n_layers(
        model=custom_model,
        threshold_trainability=topnn_nlayers,
        epochs=epochs_fc,
        optimizer=rmsprop,
        callbacks=[stopper, logger, model_saver])]
    ft_granularity = 24
    for trained_layers_idx in range(topnn_nlayers + ft_granularity, topnn_nlayers + base_model_nlayers + 1, ft_granularity):
        histories += train_top_n_layers(
            model=custom_model,
            threshold_trainability=trained_layers_idx,
            epochs=epochs_ft,
            optimizer=adam,
            callbacks=[stopper, logger, model_saver])

else:
    raise ValueError('Unspecified training technique')

print('Total training time {0:.2f} minutes'.format(-(train_time - time.time()) / 60))
save_acc_loss_plots(histories,
                    os.path.join(os.getcwd(), 'results', plot_acc_file),
                    os.path.join(os.getcwd(), 'results', plot_loss_file))
