import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.regularizers import l2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
import numpy as np

# base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
# # base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
# # base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
# # base_model = keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# # base_model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet')
# # base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# # base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#
# x = GlobalAveragePooling2D()(base_model.output)
#
# # x = Dense(1024, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005), bias_regularizer=l2(.0005))(x)
# # x = LeakyReLU()(x)
# # x = BatchNormalization()(x)
# # x = Dropout(0.5)(x)
# # x = Dense(512, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005), bias_regularizer=l2(.0005))(x)
# # x = LeakyReLU()(x)
# # x = BatchNormalization()(x)
# # x = Dropout(0.5)(x)
# # out = Dense(101, kernel_initializer='he_uniform', bias_initializer="he_uniform", activation='softmax', name='output_layer')(x)
#
# # x = Dense(512, activation='relu', name='fc-1')(x)
# # x = Dropout(0.5)(x)
# # x = Dense(256, activation='relu', name='fc-2')(x)
# # x = Dropout(0.5)(x)
#
# out = Dense(101, activation='softmax', name='output_layer')(x)
#
# model = Model(inputs=base_model.input, outputs=out)
#
# model.summary()

# vgg_val_acc = []
# vgg_val_loss = []
# vgg_train_acc = []
# vgg_train_loss = []
# with open("vgg19_ft_2017-12-22_23-55-53.csv") as ftlogfile:
#     for ix, logline in enumerate(ftlogfile):
#         split = logline.strip("\n").split("\t")
#         print(split)
#         if ix == 0: continue
#         vgg_val_acc.append(split[4])
#         vgg_val_loss.append(split[5])
#         vgg_train_acc.append(split[1])
#         vgg_train_loss.append(split[2])
#
# vgg_val_acc = np.array(list(map(float, vgg_val_acc)))
# vgg_train_acc = np.array(list(map(float, vgg_train_acc)))
# vgg_val_loss = np.array(list(map(float, vgg_val_loss)))
# vgg_train_loss = np.array(list(map(float, vgg_train_loss)))
#
# x = range(len(vgg_val_acc))
# plt.plot(x, vgg_val_acc)
# plt.plot(x, vgg_train_acc)
# plt.axvline(x=10, color='red', ls='--')
# # plt.annotate("first pass", xy=(10, 0.31), xytext=(12, 0.31), xycoords='data', verticalalignment='center', arrowprops=dict(color='red', arrowstyle="->", ls='--'))
# plt.grid(True)
# plt.title('VGG19 twopass training')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# # plt.ylim(0, 1)
# plt.legend(['validation accuracy', 'training accuracy'], loc=4)
# plt.savefig("vgg19ftplot.jpg")
# plt.clf()
#
# step_ix = -1
# xcep_epochs_per_step = [0, 0, 0, 0, 0]
# xcep_val_acc = []
# xcep_val_loss = []
# xcep_train_acc = []
# xcep_train_loss = []
# with open("xception_ft_2017-12-24_13-00-22.csv") as ftlogfile:
#     for ix, logline in enumerate(ftlogfile):
#         split = logline.strip("\n").split("\t")
#         print(split)
#         if ix == 0: continue
#         if split[0] == "0":
#             step_ix += 1
#             xcep_epochs_per_step[step_ix] += 1
#         else:
#             xcep_epochs_per_step[step_ix] += 1
#         xcep_val_acc.append(split[4])
#         xcep_val_loss.append(split[5])
#         xcep_train_acc.append(split[1])
#         xcep_train_loss.append(split[2])
#
# print(xcep_epochs_per_step)
# print(xcep_val_acc)
# print(xcep_val_loss)
# print(xcep_train_acc)
# print(xcep_train_loss)
#
# xcep_val_acc = np.array(list(map(float, xcep_val_acc)))
# xcep_train_acc = np.array(list(map(float, xcep_train_acc)))
# xcep_val_loss = np.array(list(map(float, xcep_val_loss)))
# xcep_train_loss = np.array(list(map(float, xcep_train_loss)))
#
# x = range(len(xcep_val_acc))
# plt.plot(x, xcep_val_acc)
# plt.plot(x, xcep_train_acc)
# for ix, epochs in enumerate(np.cumsum(xcep_epochs_per_step)):
#     plt.axvline(x=epochs-1, color='red', ls='--')
#     # plt.annotate(str(ix+1), xy=(epochs-1, 0.41), xytext=(epochs+2, 0.41), xycoords='data',
#     #           verticalalignment='center', arrowprops=dict(color='red', arrowstyle="->", ls='--'))
# plt.grid(True)
# plt.title('Xception bottomup training')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# # plt.ylim(0, 1)
# plt.legend(['validation accuracy', 'training accuracy'], loc=4)
# plt.savefig("xceptionftplot.jpg")
# plt.clf()

