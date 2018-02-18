from keras.layers import Conv2D, AveragePooling2D, Dense, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model
from keras.models import model_from_json
from keras.regularizers import l2
import keras

import shutil
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')

import json
import pickle
import os
import numpy as np
import matplotlib.patches as patches

from PIL import Image
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.xception import preprocess_input as inception_preprocess
from keras.preprocessing import image

def prepare_str_file_architecture_syntax(filepath):
    model_str = str(json.load(open(filepath, "r")))
    model_str = model_str.replace("'", '"')
    model_str = model_str.replace("True", "true")
    model_str = model_str.replace("False", "false")
    model_str = model_str.replace("None", "null")
    return model_str


def load_VGG16(architecture_path, weigths_path, debug=False):
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

    last_layer = model.get_layer("block5_pool")
    last_layer.outbound_nodes = []
    model.layers.pop()
    model.layers.pop()

    if debug:
        for i, l in enumerate(model.layers):
            print(i, ":", l.name)

    return model, last_layer, W, b

baseVGG16, last_layer, W, b = load_VGG16("trained_models/top5_vgg16_acc77_2017-12-24/vgg16_architecture_2017-12-23_22-53-03.json", "trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")
x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(last_layer.output)
x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b])(x)
VGG16FCN = Model(inputs=baseVGG16.input, outputs=x)
# VGG16FCN.summary()

xception = model_from_json(prepare_str_file_architecture_syntax("trained_models/top1_xception_acc80_2017-12-25/xception_architecture_2017-12-24_13-00-22.json"))
xception.load_weights("trained_models/top1_xception_acc80_2017-12-25/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5")

# incresv2 = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
# x = GlobalAveragePooling2D()(incresv2.output)
# out = Dense(101, activation='softmax', name='output_layer')(x)
# incresv2 = Model(inputs=incresv2.input, outputs=out)
# incresv2.load_weights("trained_models/top2_incresnetv2_acc79_2017-12-22/incv2resnet_ft_weights_acc0.79_e4_2017-12-21_09-02-16.hdf5")
#
# incv3 = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
# x = GlobalAveragePooling2D()(incv3.output)
# x = Dense(1024, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005), bias_regularizer=l2(.0005))(x)
# x = LeakyReLU()(x)
# x = BatchNormalization()(x)
# x = Dropout(0.5)(x)
# x = Dense(512, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005), bias_regularizer=l2(.0005))(x)
# x = LeakyReLU()(x)
# x = BatchNormalization()(x)
# x = Dropout(0.5)(x)
# out = Dense(101, kernel_initializer='he_uniform', bias_initializer="he_uniform", activation='softmax', name='output_layer')(x)
# incv3 = Model(inputs=incv3.input, outputs=out)
# incv3.load_weights("trained_models/top3_inceptionv3_acc79_2017-12-27/inceptionv3_ft_weights_acc0.79_e10_2017-12-25_22-10-02.hdf5")
#
# model_list = [xception, incresv2, incv3]
# ensemble_input = Input(shape=xception.input_shape[1:])
# outputs = [model(ensemble_input) for model in model_list]
# ensemble_output = keras.layers.average(outputs)
# ensemble = Model(inputs=ensemble_input, outputs=ensemble_output)
# ensemble.summary()

classifier = xception

def ix_to_class_name(idx):
    with open("dataset-ethz101food/meta/classes.txt") as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]

def class_name_to_idx(name):
    with open("dataset-ethz101food/meta/classes.txt") as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
        for i, label_name in enumerate(class_labels):
            if label_name == name:
                return i
        else:
            print("class idx not found!")
            exit(-1)

def traslation(heat_coord):
    return(int(32 * heat_coord / factor)) #32 is the stride of the whole convolutive net


upsampling_step = 1.2
max_upsampling_factor = 3

def process_image(input_fn, input_ix):
    results = []
    if (os.path.exists(input_fn)):
        input_img_reference = image.load_img(input_fn)
        input_img_reference = image.img_to_array(input_img_reference)
        # print("image", input_fn, "has shape", input_img_reference.shape)
        min_dim = min(input_img_reference.shape[0], input_img_reference.shape[1])
        # print("mdim = {}\n".format(min_dim))

        min_upsampling_factor = 224. / min_dim
        upsampling_factor = min_upsampling_factor

        while upsampling_factor < max_upsampling_factor:
            # print("\nupsampling factor", upsampling_factor)

            img_size = (int(max(224, input_img_reference.shape[0] * upsampling_factor)), int(max(224, input_img_reference.shape[1] * upsampling_factor)))
            input_img = image.load_img(input_fn, target_size=img_size)
            input_img = image.img_to_array(input_img)
            # fig, ax = plt.subplots(1)
            # ax.imshow(input_img / 255.)
            # plt.show()
            input_image_expandedim = np.expand_dims(input_img, axis=0)
            input_preprocessed_image = vgg_preprocess(input_image_expandedim)
            preds = VGG16FCN.predict(input_preprocessed_image)
            # print("input_img shape (height, width)", input_img.shape, "-> preds shape", preds.shape)

            # stop al primo crop che è massimo per la classe input_ix
            # seg_map = np.argmax(preds[0], axis=2)
            # bool_map_ix = seg_map == input_ix
            # if np.any(bool_map_ix):
            #     heatmaps_values = preds[0, :, :, input_ix]
            #     max_heatmap = np.amax(heatmaps_values)
            #     max_coordinates = np.unravel_index(np.argmax(heatmaps_values, axis=None), heatmaps_values.shape)
            #     # print("max value for input_ix found at", max_coordinates)
            #
            #     results.append((upsampling_factor, (preds.shape[1], preds.shape[2]), max_heatmap, max_coordinates, max_heatmap, input_ix))
            #     break
            #
            # elif results == []:  # default value
            #     heatmap_values = preds[0, :, :, input_ix]
            #     max_heatmap = np.amax(heatmap_values)
            #     max_coordinates = np.unravel_index(np.argmax(heatmap_values, axis=None), heatmap_values.shape)
            #
            #     crop_heatmaps = preds[0, max_coordinates[0], max_coordinates[1], :]
            #     max_crop = np.amax(crop_heatmaps)
            #     max_crop_ix = np.argmax(crop_heatmaps)
            #
            #     results.append((upsampling_factor, (preds.shape[1], preds.shape[2]), max_heatmap, max_coordinates, max_crop, max_crop_ix))
                # print("adding default element:\n", results)

            # produzione result ad ogni scala
            heatmaps_values = preds[0, :, :, input_ix]
            max_heatmap = np.amax(heatmaps_values)
            max_coordinates = np.unravel_index(np.argmax(heatmaps_values, axis=None), heatmaps_values.shape)
            # print("max value", max_heatmap, "found at", max_coordinates)

            crop_heatmaps = preds[0, max_coordinates[0], max_coordinates[1], :]
            max_crop = np.amax(crop_heatmaps)
            max_crop_ix = np.argmax(crop_heatmaps)

            results.append((upsampling_factor, (preds.shape[1], preds.shape[2]), max_heatmap, max_coordinates, max_crop, max_crop_ix))

            upsampling_factor *= upsampling_step

    else:
        print ("The specified image " + input_fn + " does not exist")
    return results


def threshold_max(rst_list, threshold=0.5):
    for ix, rst in enumerate(rst_list):
        if rst[2] > threshold:
            return ix
    return 0

def factor_weighted_max(rst_list, weight=1.25):
    max_ix = 0
    max_score = 0
    for ix, rst in enumerate(rst_list):
        score = (weight / rst[0]) * rst[2]
        # print("element", ix, " has factor: {:f}".format(rst[0]), ", prob: {:f}".format(rst[2]), "-> score {:f}".format(score))
        if score < max_score:
            max_score = score
            max_ix = ix
    return max_ix


# ciclo per un set di immagini
dump_list = []
set = "test"
class_folders = os.listdir("dataset-ethz101food/" + set)
folder_to_scan = 101
instances_per_folder = 250
for i_folder, class_folder in enumerate(class_folders[0:folder_to_scan]):
    instances = os.listdir("dataset-ethz101food/" + set + "/" + class_folder)
    for i_instance, instance in enumerate(instances[0:instances_per_folder]):
        filename = "dataset-ethz101food/" + set + "/" + class_folder + "/" + instance

        # classificazione
        wtrain, htrain = (299, 299)
        img_classify = image.load_img(filename, target_size=(wtrain, htrain))
        img_classify = image.img_to_array(img_classify)
        img_classify_expandedim = np.expand_dims(img_classify, axis=0)
        img_classify_preprocessed = inception_preprocess(img_classify_expandedim)
        clf = classifier.predict(img_classify_preprocessed).flatten()
        clf_cix = np.argmax(clf)
        clf_class = ix_to_class_name(clf_cix)
        clf_score = clf[clf_cix]
        # print("\nImage classified as", clf_class, "with score", clf_score)
        clf_true_label = clf[class_name_to_idx(class_folder)]

        # processamento a varie scale
        rst_list = process_image(filename, class_name_to_idx(class_folder))
        baseSize_ix = 0
        bestSize_ix = threshold_max(rst_list)  # SELEZIONE MAX: last item of the list / cmax(list)
        factor, (hdim, wdim), prob, (hcoordh, hcoordw), max_crop, max_crop_ix = rst_list[bestSize_ix]
        rect_dim = int(224 / factor)
        coordh = traslation(hcoordh)
        coordw = traslation(hcoordw)
        # img_localize = image.load_img(filename)
        # img_localize = image.img_to_array(img_localize)
        # print("Max confidence", prob, "found at scale factor", factor, " size [" + str(int(max(224, img_localize.shape[0] * factor))) + ", " +  str(int(max(224, img_localize.shape[1] * factor))) + "]:",
        #       "heatmap cell", (hcoordh, hcoordw), "in range [" + str(hdim) + ", " + str(wdim) + "] ->",
        #       "relative img point", (coordh, coordw), "in range [" + str(img_localize.shape[0])+ ", " + str(img_localize.shape[1]) + "]")


        # classificazione su crop
        img = image.load_img(filename)
        img = image.img_to_array(img)
        img_crop = img[coordh:coordh+rect_dim, coordw:coordw+rect_dim]
        img_crop = image.array_to_img(img_crop)
        img_crop = img_crop.resize((wtrain, htrain))
        img_crop = image.img_to_array(img_crop)
        # fig, ax = plt.subplots(1)  # mostra il crop
        # ax.imshow(img_crop / 255.)
        # plt.show()
        img_crop_expandedim = np.expand_dims(img_crop, axis=0)
        img_crop_preprocessed = inception_preprocess(img_crop_expandedim)
        clf_crop = classifier.predict(img_crop_preprocessed).flatten()
        crop_cix = np.argmax(clf_crop)
        crop_class = ix_to_class_name(crop_cix)
        crop_score = clf_crop[crop_cix]
        # print("Crop classified as", crop_class, "with score", crop_score)
        crop_true_label = clf_crop[class_name_to_idx(class_folder)]


        # dumping dei dati
        data = dict(filename = str(filename),
            label = str(class_folder),
            scale_factor = float(factor),
            square_crop = dict(lower_left = (int(coordh), int(coordw)), side = int(rect_dim)),
            originalSize = dict(
                vgg16 = dict(
                    score = float(rst_list[baseSize_ix][2]),
                    labelGuessed = str(ix_to_class_name(rst_list[baseSize_ix][5])),
                    scoreGuessed = float(rst_list[baseSize_ix][4])
                ),
                xception = dict(
                    score = float(clf_true_label),
                    labelGuessed = str(clf_class),
                    scoreGuessed = float(clf_score)
                )
            ),
            croppedSize = dict(
                vgg16 = dict(
                    score = float(prob),
                    labelGuessed = str(ix_to_class_name(max_crop_ix)),
                    scoreGuessed = float(max_crop)
                ),
                xception=dict(
                    score = float(crop_true_label),
                    labelGuessed = str(crop_class),
                    scoreGuessed = float(crop_score)
                )
            )
        )

        print("processed " + str(instances_per_folder * i_folder + i_instance + 1) + "/" + str(instances_per_folder * folder_to_scan))
        # print(json.dumps(data, indent=2))
        #
        # fig, ax = plt.subplots(1)
        # ax.imshow(img / 255.)
        # rect = patches.Rectangle((coordw, coordh), rect_dim, rect_dim, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

        dump_list.append(data)

with open(set + "Set" + str(instances_per_folder * folder_to_scan) + ".json", "w+") as file:
    json.dump(dump_list, file, indent=2)
# with open("testSet.pkl", "wb") as file:
#     pickle.dump(dump_list, file)