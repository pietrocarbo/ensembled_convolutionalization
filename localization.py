from keras.layers import Conv2D, AveragePooling2D, Dense, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model
from keras.models import model_from_json
from keras.regularizers import l2
import keras
import keras.backend as K

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

import PIL
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

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
def prepare_str_file_architecture_syntax(filepath):
    model_str = str(json.load(open(filepath, "r")))
    model_str = model_str.replace("'", '"')
    model_str = model_str.replace("True", "true")
    model_str = model_str.replace("False", "false")
    model_str = model_str.replace("None", "null")
    return model_str


def convolutionalize_net(architecture_path, weigths_path, last_layer_name, pool_size, debug=False):
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

dump_list = []
# K.clear_session()

# -----------------------------------
# FCNs

# vgg16FCN = convolutionalize_net(architecture_path="trained_models/top5_vgg16_acc77_2017-12-24/vgg16_architecture_2017-12-23_22-53-03.json",
#                                 weigths_path="trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5",
#                                 last_layer_name="block5_pool",
#                                 pool_size=7)

# xceptionFCN = convolutionalize_net(architecture_path="trained_models/xception_architecture_2017-12-24_13-00-22.json",
#                                    weigths_path="trained_models/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5",
#                                    last_layer_name="block14_sepconv2_act",
#                                    pool_size=10)

# -----------------------------------
# CLASSIFIERS
for input_size in [267,
                267+32-1, 267+32, 267+32+1,
                267+32*2-1, 267+32*2, 267+32*2+1,
                267+32*3-1, 267+32*3, 267+32*3+1,
                267+32*4-1, 267+32*4, 267+32*4+1,
                267+32*4-1, 267+32*4, 267+32*4+1]:

                   # INCRESV2 steps: 224, (+11) 235, (+32) 267,
                   # XCEPTION steps: 224, (+7) 231, (+32) 263, (+32) 295, (+32) 327, (+32) 359, (+32) 391, (+32) 423,
                   # VGG steps: 224, 224+32, 224+32*2, 224+32*3, 224+32*4, 224+32*5, 224+32*8, 224+32*7,
    wclf, hclf = (input_size, input_size)

    # vgg19 = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(wclf, hclf, 3))
    # x = GlobalAveragePooling2D()(vgg19.output)
    # out = Dense(101, activation='softmax', name='output_layer')(x)
    # vgg19 = Model(inputs=vgg19.input, outputs=out)
    # vgg19.load_weights("trained_models/top4_vgg19_acc78_2017-12-23/vgg19_ft_weights_acc0.78_e26_2017-12-22_23-55-53.hdf5")

    # xception = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(wclf, hclf, 3))
    # x = GlobalAveragePooling2D()(xception.output)
    # out = Dense(101, activation='softmax', name='output_layer')(x)
    # xception = Model(inputs=xception.input, outputs=out)
    # xception.load_weights("trained_models/top1_xception_acc80_2017-12-25/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5")

    incresv2 = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(wclf, hclf, 3))
    x = GlobalAveragePooling2D()(incresv2.output)
    out = Dense(101, activation='softmax', name='output_layer')(x)
    incresv2 = Model(inputs=incresv2.input, outputs=out)
    incresv2.load_weights("trained_models/top2_incresnetv2_acc79_2017-12-22/incv2resnet_ft_weights_acc0.79_e4_2017-12-21_09-02-16.hdf5")

    # incv3 = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(wclf, hclf, 3))
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

    # model_list = [xception, incresv2, incv3]
    # ensemble_input = Input(shape=xception.input_shape[1:])
    # outputs = [model(ensemble_input) for model in model_list]
    # ensemble_output = keras.layers.average(outputs)
    # ensemble = Model(inputs=ensemble_input, outputs=ensemble_output)

    print("INPUT SIZE", input_size)
    # print("vgg19 gap input shape", vgg19.get_layer("block5_pool").output_shape)
    # print("xce gap input shape", xception.get_layer("block14_sepconv2_act").output_shape)
    # print("incv3 gap input shape", incv3.get_layer("mixed10").output_shape)
    print("incresv2 gap input shape", incresv2.get_layer("conv_7b_ac").output_shape , "\n")

# classifiers = {"xception": xception, "incresv2": incresv2, "incv3": incv3}
# CLF = xception
from keras.applications.xception import preprocess_input as clf_preprocess

# FCN = xceptionFCN
# fcn_window = 299
# from keras.applications.xception import preprocess_input as fcn_preprocess

set = "test"
class_folders = os.listdir("dataset-ethz101food/" + set)
folder_to_scan = 101
instances_per_folder = 250

max_scale_factor = 3
upsampling_step = 1.2
crop_selection_policy = "max_input_ix"    #  "input_ix>=0.5"

def traslation(heat_coord, factor, fcn_stride=32):
    return(int(fcn_stride * heat_coord / factor))

# def process_image(input_img_reference, input_fn, input_ix, crop_policy):
#     results = []
#     if (os.path.exists(input_fn)):
#         scale_factor = float(fcn_window) / min(input_img_reference.shape[0], input_img_reference.shape[1])
#
#         while scale_factor < max_scale_factor:
#             img_size = (int(max(fcn_window, input_img_reference.shape[0] * scale_factor)),
#                         int(max(fcn_window, input_img_reference.shape[1] * scale_factor)))
#             input_img = image.load_img(input_fn, target_size=img_size)
#             input_img = image.img_to_array(input_img)
#             input_image_expandedim = np.expand_dims(input_img, axis=0)
#             input_preprocessed_image = fcn_preprocess(input_image_expandedim)
#             preds = FCN.predict(input_preprocessed_image)
#             # print("scale:", scale_factor, "input_img shape (height, width)", input_img.shape, "-> preds shape", preds.shape)
#
#             # valore default alla scala di base
#             if results == []:
#                 heatmap_values = preds[0, :, :, input_ix]
#                 max_heatmap = np.amax(heatmap_values)
#                 max_coordinates = np.unravel_index(np.argmax(heatmap_values, axis=None), heatmap_values.shape)
#
#                 crop_heatmaps = preds[0, max_coordinates[0], max_coordinates[1], :]
#                 max_crop = np.amax(crop_heatmaps)
#                 max_crop_ix = np.argmax(crop_heatmaps)
#
#                 results.append((scale_factor, (preds.shape[1], preds.shape[2]), max_heatmap, max_coordinates,
#                                 max_crop, max_crop_ix))
#
#             # stop al primo crop che è massimo per la classe input_ix
#             if crop_policy == "max_input_ix":
#                 seg_map = np.argmax(preds[0], axis=2)
#                 bool_map_ix = seg_map == input_ix
#                 if np.any(bool_map_ix):
#                     heatmaps_values = preds[0, :, :, input_ix]
#                     max_heatmap = np.amax(heatmaps_values)
#                     max_coordinates = np.unravel_index(np.argmax(heatmaps_values, axis=None), heatmaps_values.shape)
#
#                     results.append((scale_factor, (preds.shape[1], preds.shape[2]), max_heatmap, max_coordinates,
#                                     max_heatmap, input_ix))
#                     # print("crop max_input_ix found:", results[-1])
#                     break
#
#             # stop al primo crop >= 0.5 per la classe input_ix
#             elif crop_policy == "input_ix>=0.5":
#                 heatmaps_values = preds[0, :, :, input_ix]
#                 max_heatmap = np.amax(heatmaps_values)
#                 if max_heatmap >= 0.5:
#                     max_coordinates = np.unravel_index(np.argmax(heatmaps_values, axis=None), heatmaps_values.shape)
#
#                     crop_heatmaps = preds[0, max_coordinates[0], max_coordinates[1], :]
#                     max_crop = np.amax(crop_heatmaps)
#                     max_crop_ix = np.argmax(crop_heatmaps)
#
#                     results.append((scale_factor, (preds.shape[1], preds.shape[2]), max_heatmap, max_coordinates,
#                                     max_crop, max_crop_ix))
#                     # print("crop input_ix>=0.5 found:", results[-1])
#                     break
#
#             else:
#                 print("Unspecified crop policy. Exiting.")
#                 exit(-1)
#
#             scale_factor *= upsampling_step
#     else:
#         print ("The specified image " + input_fn + " does not exist")
#
#     # if len(results) == 1:
#     #     print("crop default", results[0])
#     return results

def get_random_crop(x, random_crop_size, sync_seed=None):
    np.random.seed(sync_seed)
    h, w = x.shape[0], x.shape[1]
    rangeh = (h - random_crop_size) // 2
    rangew = (w - random_crop_size) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    # print("img shape", x.shape, "crop_dim", rect_dim, "crop: H", offseth, ":", offseth+random_crop_size, ", W", offsetw, ":", offsetw+random_crop_size)
    return x[offseth:offseth+random_crop_size, offsetw:offsetw+random_crop_size]


def classify(classifier, filename, custom_size=None, input_classix=None):
    if custom_size:
        img = image.load_img(filename, target_size=(custom_size[0], custom_size[1]))
    else:
        img = image.load_img(filename)

    img_array = image.img_to_array(img)
    img_array_expandedim = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = clf_preprocess(img_array_expandedim)
    clf = classifier.predict(img_array_preprocessed).flatten()
    clf_ixmax = np.argmax(clf)
    clf_labelmax = ix_to_class_name(clf_ixmax)
    clf_scoremax = clf[clf_ixmax]

    if input_classix:
        clf_scoreinputix = clf[class_name_to_idx(input_classix)]
        return (clf_ixmax, clf_labelmax, clf_scoremax, clf_scoreinputix)
    else:
        return (clf_ixmax, clf_labelmax, clf_scoremax)

def resize_arrayimg(imgarray, new_width, new_height):
    imgarray = image.array_to_img(imgarray)
    imgarray = imgarray.resize((new_width, new_height), PIL.Image.BILINEAR)
    imgarray = image.img_to_array(imgarray)
    return imgarray


# for key in classifiers:
#     print("Validating model", key, "at", classifiers[key])
#     classifiers[key].summary()
#     dict_augmentation = dict(preprocessing_function=clf_preprocess)
#     test_datagen = ImageDataGenerator(**dict_augmentation)
#     validation_generator = test_datagen.flow_from_directory(
#         'dataset-ethz101food/test',
#         target_size=(wclf, hclf),
#         batch_size=32,
#         class_mode='categorical')
#     classifiers[key].compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
#     (loss, acc, top5acc) = classifiers[key].evaluate_generator(validation_generator, (instances_per_folder * folder_to_scan) // 32)
#     print("[Model", key, "] test-set: loss={:.4f}, top-1 acc={:.4f}%, top-5 acc={:.4f}%".format(loss, acc * 100, top5acc * 100))
#     dump_list.append(dict(
#         model = key,
#         loss = loss,
#         acc = acc,
#         top5acc = top5acc
#     ))
# with open("inceptions256testSet.json", "w+") as file:
#     json.dump(dump_list, file, indent=2, sort_keys=True)
#


# ciclo per un set di immagini
# for i_folder, class_folder in enumerate(class_folders[0:folder_to_scan]):
#     instances = os.listdir("dataset-ethz101food/" + set + "/" + class_folder)
#     for i_instance, instance in enumerate(instances[0:instances_per_folder]):
#         filename = "dataset-ethz101food/" + set + "/" + class_folder + "/" + instance

        # img_classify = image.img_to_array(img_classify)
        # img_classify_expandedim = np.expand_dims(img_classify, axis=0)
        # img_classify_preprocessed = clf_preprocess(img_classify_expandedim)
        # clf = CLF.predict(img_classify_preprocessed).flatten()
        # clf_cix = np.argmax(clf)
        # clf_class = ix_to_class_name(clf_cix)
        # clf_score = clf[clf_cix]
        # clf_true_label = clf[class_name_to_idx(class_folder)]

        # estrazione best crop
        # img = image.load_img(filename)
        # img = image.img_to_array(img)
        # rst_list = process_image(img, filename, class_name_to_idx(class_folder), crop_selection_policy)
        # factor, (hdim, wdim), prob, (hcoordh, hcoordw), max_crop, max_crop_ix = rst_list[-1]
        # # if hdim > 1 or wdim > 1:
        # #     countCnb +=1
        # #     factorCnb += rst_list[-1][0]
        # #     print("n.", count, "img:", filename, "rst_list (len", len(rst_list), ")", rst_list)
        # rect_dim = int(fcn_window / factor)
        # coordh = traslation(hcoordh, factor)
        # coordw = traslation(hcoordw, factor)
        # # img_localize = image.load_img(filename)
        # img_localize = image.img_to_array(img_localize)
        # print("Max confidence", prob, "found at scale factor", factor, " size [" + str(int(max(224, img_localize.shape[0] * factor))) + ", " +  str(int(max(224, img_localize.shape[1] * factor))) + "]:",
        #       "heatmap cell", (hcoordh, hcoordw), "in range [" + str(hdim) + ", " + str(wdim) + "] ->",
        #       "relative img point", (coordh, coordw), "in range [" + str(img_localize.shape[0])+ ", " + str(img_localize.shape[1]) + "]")

        # classificazione sul best crop
        # crop = img[coordh:coordh + rect_dim, coordw:coordw + rect_dim]
        # crop = image.array_to_img(crop)
        # crop = crop.resize((wtrain, htrain))
        # crop = image.img_to_array(crop)
        # crop_expandedim = np.expand_dims(crop, axis=0)
        # crop_preprocessed = clf_preprocess(crop_expandedim)
        # crop_clf = CLF.predict(crop_preprocessed).flatten()
        # crop_cix = np.argmax(crop_clf)
        # crop_class = ix_to_class_name(crop_cix)
        # crop_score = crop_clf[crop_cix]


        # classificazione su random crop
        # random_crop = get_random_crop(img, rect_dim)
        # random_crop = image.array_to_img(random_crop)
        # random_crop = random_crop.resize((wtrain, htrain))
        # random_crop = image.img_to_array(random_crop)
        # random_crop_expandedim = np.expand_dims(random_crop, axis=0)
        # random_crop_preprocessed = clf_preprocess(random_crop_expandedim)
        # random_crop_clf = CLF.predict(random_crop_preprocessed).flatten()
        # random_crop_cix = np.argmax(random_crop_clf)
        # random_crop_class = ix_to_class_name(crop_cix)
        # random_crop_score = random_crop_clf[crop_cix]

        # dumping dei dati
        # data = dict(filename = str(filename),
        #     label = str(class_folder),
        #     scale_factor = float(factor),
        #     square_crop = dict(lower_left = (int(coordh), int(coordw)), side = int(rect_dim)),
        #     originalSize = dict(
        #         vgg16 = dict(
        #             score = float(rst_list[0][2]),
        #             labelGuessed = str(ix_to_class_name(rst_list[0][5])),
        #             scoreGuessed = float(rst_list[0][4])
        #         ),
        #         xception = dict(
        #             score = float(clf_true_label),
        #             labelGuessed = str(clf_class),
        #             scoreGuessed = float(clf_score)
        #         )
        #     ),
        #     croppedSize = dict(
        #         vgg16 = dict(
        #             score = float(prob),
        #             labelGuessed = str(ix_to_class_name(max_crop_ix)),
        #             scoreGuessed = float(max_crop)
        #         ),
        #         xception=dict(
        #             score = float(crop_true_label),
        #             labelGuessed = str(crop_class),
        #             scoreGuessed = float(crop_score)
        #         )
        #     )
        # )
        # data = dict(filename=str(filename),
        #     label=str(class_folder),
        #     scale_factor=float(factor),
        #     square_crop=dict(lower_left=(int(coordh), int(coordw)), side=int(rect_dim)),
        #     predictions=dict(
        #         randomCrop=dict(
        #             scoreTrueLabel=float(random_crop_clf[class_name_to_idx(class_folder)]),
        #             labelGuessed=str(ix_to_class_name(random_crop_cix)),
        #             scoreGuessed=float(random_crop_clf[random_crop_cix])
        #         ),
        #         cropFcn=dict(
        #             scoreTrueLabel=float(crop_clf[class_name_to_idx(class_folder)]),
        #             labelGuessed=str(ix_to_class_name(crop_cix)),
        #             scoreGuessed=float(crop_clf[crop_cix])
        #         )
        #     )
        # )
        # dump_list.append(data)
        #
        # # stampa dei risultati
        # # if i_instance == 0:
        # #     print("processing " + str(instances_per_folder * i_folder + i_instance + 1) + "/" + str(instances_per_folder * folder_to_scan))
        # #     print("#imgs cropped at non-original size", countCnb, ", avg factor", factorCnb / countCnb)
        # print(json.dumps(data, indent=2, sort_keys=True))
        #
        # fig, (ax0, ax1, ax2) = plt.subplots(1, 3) #, figsize=(8, 8))
        # ax0.set_title("FCN crop")
        # ax0.imshow((crop + 1) / 2)
        #
        # ax1.set_title("Random crop")
        # ax1.imshow((random_crop + 1) / 2)
        #
        # ax2.set_title("Img + FCN crop")
        # ax2.imshow(img / 255.)
        # rect = patches.Rectangle((coordw, coordh), rect_dim, rect_dim, linewidth=1, edgecolor='r', facecolor='none')
        # ax2.add_patch(rect)
        #
        # plt.show()

# with open(set + "Set" + str(instances_per_folder * folder_to_scan) + "_Crop-RandomCrop-CLF" + ".json", "w+") as file:
#     json.dump(dump_list, file, indent=2, sort_keys=True)