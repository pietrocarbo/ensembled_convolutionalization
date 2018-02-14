from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model
from keras.models import model_from_json

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
from keras.applications.vgg16 import preprocess_input
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

xception_notFCN = model_from_json(prepare_str_file_architecture_syntax("trained_models/top1_xception_acc80_2017-12-25/xception_architecture_2017-12-24_13-00-22.json"))
xception_notFCN.load_weights("trained_models/top1_xception_acc80_2017-12-25/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5")
# xception_notFCN.summary()

def idx_to_class_name(idx):
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
            input_preprocessed_image = preprocess_input(input_image_expandedim)

            preds = VGG16FCN.predict(input_preprocessed_image)
            # print("input_img shape (height, width)", input_img.shape, "-> preds shape", preds.shape)

            heatmaps_values = preds[0, :, :, input_ix]
            max_heatmap = np.amax(heatmaps_values)
            max_coordinates = np.unravel_index(np.argmax(heatmaps_values, axis=None), heatmaps_values.shape)
            # print("max value", max_heatmap, "found at", max_coordinates)

            results.append((upsampling_factor, (preds.shape[1], preds.shape[2]), max_heatmap, max_coordinates))

            upsampling_factor *= upsampling_step
    else:
        print ("The specified image " + input_fn + " does not exist")
    return results


# ciclo per un set di immagini
dump_list = []
set = "test"
class_folders = os.listdir("dataset-ethz101food/" + set)
folder_to_scan = 101
instances_per_folder = 5
for i_folder, class_folder in enumerate(class_folders[0:folder_to_scan]):
    instances = os.listdir("dataset-ethz101food/" + set + "/" + class_folder)
    for i_instance, instance in enumerate(instances[0:instances_per_folder]):
        filename = "dataset-ethz101food/" + set + "/" + class_folder + "/" + instance

        # processamento immagine a varie scale
        rst_list = process_image(filename, class_name_to_idx(class_folder))
        factor, (hdim, wdim), prob, (hcoordh, hcoordw) = max(rst_list, key=lambda x: x[2])
        rect_dim = int(224 / factor)
        coordh = traslation(hcoordh)
        coordw = traslation(hcoordw)

        # classificazione
        wtrain, htrain = (299, 299)
        img_classify = image.load_img(filename, target_size=(wtrain, htrain))
        img_classify = image.img_to_array(img_classify)
        # fig, ax = plt.subplots(1)
        # ax.imshow(img_classify / 255.)
        # plt.show()
        img_classify_expandedim = np.expand_dims(img_classify, axis=0)
        img_classify_preprocessed = preprocess_input(img_classify_expandedim)
        clf = xception_notFCN.predict(img_classify_preprocessed).flatten()
        clf_cix = np.argmax(clf)
        clf_class = idx_to_class_name(clf_cix)
        clf_score = clf[clf_cix]
        # print("\nImage classified as", clf_class, "with score", clf_score)

        # output localizzazione
        img_localize = image.load_img(filename)
        img_localize = image.img_to_array(img_localize)
        # fig, ax = plt.subplots(1)
        # ax.imshow(img_localize / 255.)
        # rect = patches.Rectangle((coordw, coordh), rect_dim, rect_dim, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # print("Max confidence", prob, "found at scale factor", factor, " size [" + str(int(max(224, img_localize.shape[0] * factor))) + ", " +  str(int(max(224, img_localize.shape[1] * factor))) + "]:",
        #       "heatmap cell", (hcoordh, hcoordw), "in range [" + str(hdim) + ", " + str(wdim) + "] ->",
        #       "relative img point", (coordh, coordw), "in range [" + str(img_localize.shape[0])+ ", " + str(img_localize.shape[1]) + "]")
        plt.show()


        img_crop = img_localize[coordh:coordh+rect_dim, coordw:coordw+rect_dim]
        img_crop = image.array_to_img(img_crop)
        img_crop = img_crop.resize((wtrain, htrain))
        img_crop = image.img_to_array(img_crop)
        # fig, ax = plt.subplots(1)
        # ax.imshow(img_crop / 255.)
        # plt.show()
        img_crop_expandedim = np.expand_dims(img_crop, axis=0)
        img_crop_preprocessed = preprocess_input(img_crop_expandedim)
        clf_crop = xception_notFCN.predict(img_crop_preprocessed).flatten()
        crop_cix = np.argmax(clf_crop)
        crop_class = idx_to_class_name(crop_cix)
        crop_score = clf_crop[crop_cix]
        # print("Crop classified as", crop_class, "with score", crop_score)

        dump_list.append(dict(filename = filename,
            scale_factor = factor,
            square_crop = dict(lower_left = (coordh, coordw), side = rect_dim),
            scoreFCNtrainSize = rst_list[0][2],
            scoreFCNbestSize = prob,
            scoreCLF = clf_score,
            scoreCLFcrop = crop_score))
        print("added to dump list " + str(instances_per_folder * i_folder + i_instance) + "/" + str(instances_per_folder * folder_to_scan))

with open("testSet.json", "w+") as file:
    json.dump(dump_list, file, indent=2)
# with open("testSet.pkl", "wb") as file:
#     pickle.dump(dump_list, file)