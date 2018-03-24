from keras.layers import Conv2D, AveragePooling2D, Dense, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, \
    Dropout
from keras.models import Model
from keras.models import model_from_json
from keras.regularizers import l2
import keras
import keras.backend as K

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-bright')
import matplotlib.ticker as plticker

import time
import json
import pickle
from random import shuffle
import os
import numpy as np
import matplotlib.patches as patches

import PIL
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from utils.labels_ix_mapping import ix_to_class_name, class_name_to_idx
dataset_path = "dataset-ethz101food"

def prepare_str_file_architecture_syntax(filepath):
    model_str = str(json.load(open(filepath, "r")))
    model_str = model_str.replace("'", '"')
    model_str = model_str.replace("True", "true")
    model_str = model_str.replace("False", "false")
    model_str = model_str.replace("None", "null")
    return model_str


def convolutionalize_architecture(architecture_path, weigths_path, last_layer_name, pool_size, debug=False):
    model = model_from_json(prepare_str_file_architecture_syntax(architecture_path))
    model.load_weights(weigths_path)
    if debug:
        print("IMPORTED MODEL")
        model.summary()

    p_dim = model.get_layer("global_average_pooling2d_1").input_shape
    out_dim = model.get_layer("output_layer").get_weights()[1].shape[0]
    W, b = model.get_layer("output_layer").get_weights()

    weights_shape = (1, 1, p_dim[3], out_dim)

    if debug:
        print("weights old shape", W.shape, "values", W)
        print("biases old shape", b.shape, "values", b)
        print("weights new shape", weights_shape)

    W = W.reshape(weights_shape)

    last_layer = model.get_layer(last_layer_name)
    last_layer.outbound_nodes = []
    model.layers.pop()
    model.layers.pop()

    x = AveragePooling2D(pool_size=(pool_size, pool_size), strides=(1, 1))(last_layer.output)
    x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b], name="conv2d_fcn")(x)
    model = Model(inputs=model.input, outputs=x)

    if debug:
        print("CONVOLUTIONALIZED MODEL")
        model.summary()

    return model


def convolutionalize_incresv2():
    incresv2 = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                        input_shape=(None, None, 3))
    x = GlobalAveragePooling2D()(incresv2.output)
    out = Dense(101, activation='softmax', name='output_layer')(x)
    incresv2 = Model(inputs=incresv2.input, outputs=out)
    incresv2.load_weights(
        "trained_models/top2_incresnetv2_acc79_2017-12-22/incv2resnet_ft_weights_acc0.79_e4_2017-12-21_09-02-16.hdf5")
    # incresv2.summary()

    out_dim = incresv2.get_layer("output_layer").get_weights()[1].shape[0]
    p_dim = incresv2.get_layer("global_average_pooling2d_1").input_shape
    W, b = incresv2.get_layer("output_layer").get_weights()
    weights_shape = (1, 1, p_dim[3], out_dim)
    W = W.reshape(weights_shape)
    last_layer = incresv2.get_layer("conv_7b_ac")
    last_layer.outbound_nodes = []
    incresv2.layers.pop()
    incresv2.layers.pop()
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1))(last_layer.output)
    x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b], name="conv2d_fcn")(x)
    incresv2 = Model(inputs=incresv2.input, outputs=x)
    return incresv2


def convolutionalize_incv3():
    incv3 = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                        input_shape=(None, None, 3))
    x = GlobalAveragePooling2D()(incv3.output)
    x = Dense(1024, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005),
              bias_regularizer=l2(.0005))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005),
              bias_regularizer=l2(.0005))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(101, kernel_initializer='he_uniform', bias_initializer="he_uniform", activation='softmax',
                name='output_layer')(x)
    incv3 = Model(inputs=incv3.input, outputs=out)
    incv3.load_weights(
        "trained_models/top3_inceptionv3_acc79_2017-12-27/inceptionv3_ft_weights_acc0.79_e10_2017-12-25_22-10-02.hdf5")
    # incv3.summary()

    W1, b1 = incv3.get_layer("dense_1").get_weights()
    W2, b2 = incv3.get_layer("dense_2").get_weights()
    W3, b3 = incv3.get_layer("output_layer").get_weights()

    BN1 = incv3.get_layer("batch_normalization_298").get_weights()
    BN2 = incv3.get_layer("batch_normalization_299").get_weights()

    W1 = W1.reshape((1, 1, 2048, 1024))
    W2 = W2.reshape((1, 1, 1024, 512))
    W3 = W3.reshape((1, 1, 512, 101))

    last_layer = incv3.get_layer("mixed10")
    last_layer.outbound_nodes = []
    for i in range(10):
        incv3.layers.pop()

    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1))(last_layer.output)

    x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid', weights=[W1, b1],
               name="conv2d_fcn1")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(weights=BN1)(x)
    x = Dropout(0.5)(x)

    x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', weights=[W2, b2],
               name="conv2d_fcn2")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(weights=BN2)(x)
    x = Dropout(0.5)(x)

    x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W3, b3],
               name="conv2d_fcn3")(x)
    incv3 = Model(inputs=incv3.input, outputs=x)
    return incv3


# K.clear_session()

# -----------------------------------
# FCNs

vgg16FCN = convolutionalize_architecture(
    architecture_path="trained_models/top5_vgg16_acc77_2017-12-24/vgg16_architecture_2017-12-23_22-53-03.json",
    weigths_path="trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5",
    last_layer_name="block5_pool",
    pool_size=9)

xceptionFCN = convolutionalize_architecture(
    architecture_path="trained_models/top1_xception_acc80_2017-12-25/xception_architecture_2017-12-24_13-00-22.json",
    weigths_path="trained_models/top1_xception_acc80_2017-12-25/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5",
    last_layer_name="block14_sepconv2_act",
    pool_size=10)

incresv2FCN = convolutionalize_incresv2()

incv3FCN = convolutionalize_incv3()


def predict_from_imgarray(model, img, input_size, preprocess):
    img = image.array_to_img(img)
    img = img.resize((input_size[0], input_size[1]), PIL.Image.BICUBIC)  # width, height order here!
    img = image.img_to_array(img)
    img_expandedim = np.expand_dims(img, axis=0)
    img_preprocessed_image = preprocess(img_expandedim)
    preds = model.predict(img_preprocessed_image)
    return preds


def predict_from_filename(model, filename, input_size, preprocess):
    input_img = image.load_img(filename, target_size=input_size)
    input_img = image.img_to_array(input_img)
    input_image_expandedim = np.expand_dims(input_img, axis=0)
    input_preprocessed_image = preprocess(input_image_expandedim)
    preds = model.predict(input_preprocessed_image)
    return preds


def get_top1data(preds, extraClass):
    maxix = np.argmax(preds)
    return (maxix, ix_to_class_name(maxix), preds[maxix], preds[class_name_to_idx(extraClass)])


def dim_size(w, k, s):
    return ((w - k) // s + 1)

def process_image(fcns, kernels, prep_funcs, input_fn, input_cix, img_shape, upsampling_step=1.2, max_scale_factor=3.0):
    results = []
    if (os.path.exists(input_fn)):
        base_kernel_size = sum(kernels) // len(kernels)
        scale_factor = float(base_kernel_size) / min(img_shape[0], img_shape[1])
        maxcn = 0

        while scale_factor < max_scale_factor and maxcn < len(fcns):
            # definiamo la dimensione attesa della heatmap a questa scala usando il kernel del primo fcn
            # riscaleremo poi le immagini alle dimensioni giuste per gli altri cropper,
            # in modo da avere in output un heatmap della dimensione prevista
            base_kernel_size = min(kernels)
            heatmap_h = dim_size(round(img_shape[0] * scale_factor), base_kernel_size, 32)
            heatmap_w = dim_size(round(img_shape[1] * scale_factor), base_kernel_size, 32)
            # print("Heatmapdim scale dim:", heatmap_h, heatmap_w)

            # cerchiamo a questa scala il crop che ha il numero massimo di croppatori che lo classificano come classe input_ix
            heatmaps = []
            bool_cix_maps = []
            for ix, fcn in enumerate(fcns):
                scaled_w = kernels[ix] + (heatmap_w - 1) * 32
                scaled_h = kernels[ix] + (heatmap_h - 1) * 32
                # print("Scaled input dim:", scaled_h, scaled_w)

                heatmaps.append(predict_from_filename(fcn, input_fn, (scaled_h, scaled_w), prep_funcs[ix])[0])

                bool_cix_map = np.argmax(heatmaps[-1], axis=2) == input_cix
                bool_cix_maps.append(bool_cix_map)

            # ncix_max_map è la mappa che mi dice quanti croppatori hanno classificato un crop come input_ix. ha valori da 0 a 4 quindi
            ncix_max_map = np.zeros(bool_cix_maps[-1].shape, dtype=int)
            for bool_cix_map in bool_cix_maps:
                ncix_max_map += bool_cix_map

            maxcn = np.max(ncix_max_map)  # valore massimo della mappa ncix_max_map
            positions = np.nonzero(ncix_max_map == maxcn)  # tupla con indici relativi a ncix_max_map dove è presente il valore maxcn
            positions = list(zip(positions[0], positions[1]))

            # print(positions)

            def sum_crop_score(x):
                res = 0
                for map in heatmaps:
                    res += map[x[0], x[1], input_cix]
                return res

            ordpositions = sorted(positions, key=sum_crop_score)
            best_crop_ix = ordpositions[-1]
            best_crop_score = sum_crop_score(best_crop_ix) / len(fcns)
            correct_fcn = [bool_cix_map[best_crop_ix[0], best_crop_ix[1]] for bool_cix_map in
                           bool_cix_maps]  # array booleano
            results.append({"factor": scale_factor, "heatmap_shape": heatmaps[-1].shape[0:2], "ix": best_crop_ix,
                            "score": best_crop_score, "nfcn_clf_ix": maxcn, "fcn_clf_ix": correct_fcn})

            # si passa ora alla prossima scala
            scale_factor *= upsampling_step

        # for result in results:
        #     print(result)
        # print("\n")

    else:
        print("The image file " + str(input_fn) + " does not exist")

    return results


# seleziona il best crop della return list
def select_best_crop(res_list):
    sort_list = sorted(res_list, key=lambda res: (res["nfcn_clf_ix"], res["score"]), reverse=True)
    return (sort_list[0])


def traslation(heat_coord, factor, fcn_stride=32):
    return (int(fcn_stride * heat_coord / factor))


file_list = []
set = "test"
class_folders = os.listdir(os.path.join(dataset_path, set))
folder_to_scan = 101
instances_per_folder = 250

for i_folder, class_folder in enumerate(class_folders[0:folder_to_scan]):
    instances = os.listdir(os.path.join(dataset_path, set, class_folder))
    for i_instance, instance in enumerate(instances[0:instances_per_folder]):
        filename = os.path.join(dataset_path, set, class_folder, instance)
        file_list.append((filename, class_folder))

kernel_sizes = [288, 295, 299, 299]
FCNs = [vgg16FCN, xceptionFCN, incresv2FCN, incv3FCN]
preprocess_funcs = [keras.applications.vgg16.preprocess_input
    , keras.applications.xception.preprocess_input
    , keras.applications.inception_resnet_v2.preprocess_input
    , keras.applications.inception_v3.preprocess_input]

crops_vgg16 = []
crops_xce = []
crops_incrnv2 = []
crops_incv3 = []

i_processed = 0
for filename, class_folder in file_list:
    ix_label = class_name_to_idx(class_folder)
    img = image.load_img(filename)
    img = image.img_to_array(img)
    imgh, imgw = img.shape[0:2]

    for ix, excluded_fcn in enumerate(FCNs):

        ensemble = FCNs[:ix] + FCNs[ix+1:]
        kernels = kernel_sizes[:ix] + kernel_sizes[ix + 1:]
        preprocesses = preprocess_funcs[:ix] + preprocess_funcs[ix + 1:]

        res_list = process_image(ensemble, kernels, preprocesses, filename, class_name_to_idx(class_folder), (imgh, imgw))
        crop = select_best_crop(res_list)
        coordh = traslation(crop["ix"][0], crop["factor"])
        coordw = traslation(crop["ix"][1], crop["factor"])
        rect_dim = int(295 / crop["factor"])

        # debug-purpose
        # if not is_square_in_img(coordh, coordw, rect_dim, imgh, imgw):
        #     print("Crop out of img bound! File:", filename, "Crop data:", coordh, coordw, rect_dim, imgh, imgw)
        # print("Wrong sample", str(count) + "/" + str(len(file_list)), "->", filename, "factor", crop["factor"], "score", crop["score"], "nets classifing correct", crop["nfcn_clf_ix"], crop["fcn_clf_ix"])
        # print("Max confidence", crop["score"], "at scale", crop["factor"],
        #       "heatmap crop", (crop["ix"][0], crop["ix"][1]),
        #       "in range [" + str(crop["heatmap_shape"][0]) + ", " + str(crop["heatmap_shape"][1]) + "] ->",
        #       "relative img point", (coordh, coordw), "in range [" + str(imgh) + ", " + str(imgw) + "]")
        # fig, ax = plt.subplots(1)
        # ax.imshow(img / 255.)
        # ax.set_title(class_folder)
        # rect = patches.Rectangle((coordw, coordh), rect_dim, rect_dim, linewidth=2, edgecolor='g', facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

        if ix == 0:
            crops_list = crops_vgg16
        elif ix == 1:
            crops_list = crops_xce
        elif ix == 2:
            crops_list = crops_incrnv2
        elif ix == 3:
            crops_list = crops_incv3
        else:
            raise Exception("Unkown fcn index", ix, excluded_fcn)

        crops_list.append(dict(filename=str(filename),
                              label=str(class_folder),
                              crop=dict(
                                  factor=float(crop["factor"]),
                                  heath=int(crop["heatmap_shape"][0]),
                                  heatw=int(crop["heatmap_shape"][1]),
                                  cropixh=int(crop["ix"][0]),
                                  cropixw=int(crop["ix"][1]),
                                  score=float(crop["score"]),
                                  nfcn=int(crop["nfcn_clf_ix"]),
                                  fcn=dict(fcn1=str(crop["fcn_clf_ix"][0]),
                                           fcn2=str(crop["fcn_clf_ix"][1]),
                                           fcn3=str(crop["fcn_clf_ix"][2])
                                           )
                                        ),
                              rect=dict(lower_left=(int(coordh), int(coordw)), side=int(rect_dim))
                          )
         )

    i_processed += 1
    if i_processed % instances_per_folder == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S") + " started class " + str(i_processed // instances_per_folder) + " of " + str(folder_to_scan))


pickle.dump(crops_xce, open("crops_xce.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(crops_vgg16, open("crops_vgg16.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(crops_incv3, open("crops_incv3.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(crops_incrnv2, open("crops_incrnv2.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
