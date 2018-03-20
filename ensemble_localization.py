from keras.layers import Conv2D, AveragePooling2D, Dense, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dropout, Input
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

dataset_path = "dataset-ethz101food"

def ix_to_class_name(idx):
    with open(os.path.join(dataset_path, "meta", "classes.txt")) as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]

def class_name_to_idx(name):
    with open(os.path.join(dataset_path, "meta", "classes.txt")) as file:
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

vgg16FCN = convolutionalize_architecture(architecture_path="trained_models/top5_vgg16_acc77_2017-12-24/vgg16_architecture_2017-12-23_22-53-03.json",
                                         weigths_path="trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5",
                                         last_layer_name="block5_pool",
                                         pool_size=9)

xceptionFCN = convolutionalize_architecture(architecture_path="trained_models/xception_architecture_2017-12-24_13-00-22.json",
                                            weigths_path="trained_models/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5",
                                            last_layer_name="block14_sepconv2_act",
                                            pool_size=10)

incresv2FCN = convolutionalize_incresv2()

incv3FCN = convolutionalize_incv3()


# -----------------------------------
# CLFs
vgg16CLF = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(vgg16CLF.output)
out = Dense(101, activation='softmax', name='output_layer')(x)
vgg16CLF = Model(inputs=vgg16CLF.input, outputs=out)
vgg16CLF.load_weights("trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")

vgg19CLF = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(vgg19CLF.output)
out = Dense(101, activation='softmax', name='output_layer')(x)
vgg19CLF = Model(inputs=vgg19CLF.input, outputs=out)
vgg19CLF.load_weights("trained_models/top4_vgg19_acc78_2017-12-23/vgg19_ft_weights_acc0.78_e26_2017-12-22_23-55-53.hdf5")

xceptionCLF = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(xceptionCLF.output)
out = Dense(101, activation='softmax', name='output_layer')(x)
xceptionCLF = Model(inputs=xceptionCLF.input, outputs=out)
xceptionCLF.load_weights("trained_models/top1_xception_acc80_2017-12-25/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5")

incresv2CLF = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(incresv2CLF.output)
out = Dense(101, activation='softmax', name='output_layer')(x)
incresv2CLF = Model(inputs=incresv2CLF.input, outputs=out)
incresv2CLF.load_weights("trained_models/top2_incresnetv2_acc79_2017-12-22/incv2resnet_ft_weights_acc0.79_e4_2017-12-21_09-02-16.hdf5")

incv3CLF = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(incv3CLF.output)
x = Dense(1024, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005), bias_regularizer=l2(.0005))(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005), bias_regularizer=l2(.0005))(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
out = Dense(101, kernel_initializer='he_uniform', bias_initializer="he_uniform", activation='softmax', name='output_layer')(x)
incv3CLF = Model(inputs=incv3CLF.input, outputs=out)
incv3CLF.load_weights("trained_models/top3_inceptionv3_acc79_2017-12-27/inceptionv3_ft_weights_acc0.79_e10_2017-12-25_22-10-02.hdf5")


def predict_from_imgarray(model, img, input_size, preprocess):
    # if img.shape[0] != input_size[0] or img.shape[1] != input_size[1]:
    img = image.array_to_img(img)
    img = img.resize((input_size[0], input_size[1]), PIL.Image.BICUBIC)  # width, height order here!
    img = image.img_to_array(img)
    # fig, ax = plt.subplots(1)
    # ax.imshow(img / 255.)
    # plt.show()
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

#inc_list = [0, 32, 64, 96, 160, 224, 288, 384, 512]
kernel_sizes = [288, 295, 299, 299]
FCNs = [vgg16FCN, xceptionFCN, incresv2FCN, incv3FCN]
preprocess_func = [  keras.applications.vgg16.preprocess_input
                   , keras.applications.xception.preprocess_input
                   , keras.applications.inception_resnet_v2.preprocess_input
                   , keras.applications.inception_v3.preprocess_input]

def dim_size(w, k, s):
  return ((w - k) // s + 1)
        
def process_image(input_fn, input_cix, img_shape, upsampling_step = 1.2, max_scale_factor = 2.5):
    results = []
    if (os.path.exists(input_fn)):
        base_kernel_size = 295 # any of the kernels would do
        scale_factor = float(base_kernel_size) / min(img_shape[0], img_shape[1])
        maxcn = 0
        
        while scale_factor < max_scale_factor and maxcn < 4:
            # definiamo la dimensione attesa della heatmap a questa scala usando il kernel del primo fcn
            # riscaleremo poi le immagini alle dimensioni giuste per gli altri cropper,
            # in modo da avere in output un heatmap della dimensione prevista
            base_kernel_size = kernel_sizes[0]
            heatmap_h = dim_size(round(img_shape[0]*scale_factor), base_kernel_size, 32)
            heatmap_w = dim_size(round(img_shape[1]*scale_factor), base_kernel_size, 32)
            # print("Heatmapdim scale dim:", heatmap_h, heatmap_w)
            
            # cerchiamo a questa scala il crop che ha il numero massimo di croppatori che lo classificano come classe input_ix
            heatmaps = []
            bool_cix_maps = []
            for ix, fcn in enumerate(FCNs):
                scaled_w = kernel_sizes[ix] + (heatmap_w - 1) * 32
                scaled_h = kernel_sizes[ix] + (heatmap_h - 1) * 32
                # print("Scaled input dim:", scaled_h, scaled_w)

                heatmaps.append(predict_from_filename(fcn, input_fn, (scaled_h, scaled_w), preprocess_func[ix])[0])

                bool_cix_map = np.argmax(heatmaps[-1], axis=2) == input_cix
                bool_cix_maps.append(bool_cix_map)

            # ncix_max_map è la mappa che mi dice quanti croppatori hanno classificato un crop come input_ix. ha valori da 0 a 4 quindi
            ncix_max_map = np.zeros(bool_cix_maps[-1].shape, dtype=int)
            for bool_cix_map in bool_cix_maps:
                ncix_max_map += bool_cix_map

            maxcn = np.max(ncix_max_map)    # valore massimo della mappa ncix_max_map
            positions = np.nonzero(ncix_max_map == maxcn)  # tupla con indici relativi a ncix_max_map dove è presente il valore maxcn
            positions = list(zip(positions[0], positions[1]))
            #print(positions)

            def sum_crop_score(x):
                res = 0
                for map in heatmaps:
                    res += map[x[0], x[1], input_cix]
                return res

            ordpositions = sorted(positions, key=sum_crop_score)
            best_crop_ix = ordpositions[-1]
            best_crop_score = sum_crop_score(best_crop_ix) / 4
            correct_fcn = [bool_cix_map[best_crop_ix[0], best_crop_ix[1]] for bool_cix_map in bool_cix_maps] # array booleano
            results.append({"factor": scale_factor, "heatmap_shape": heatmaps[-1].shape[0:2], "ix": best_crop_ix,
                            "score": best_crop_score, "nfcn_clf_ix": maxcn, "fcn_clf_ix": correct_fcn})

            # si passa ora alla prossima scala
            scale_factor *= upsampling_step

        # for result in results:
        #     print(result)
        # print("\n")

    else:
        print ("The image file " + str(input_fn) + " does not exist")

    return results

# seleziona il best crop della return list
def select_best_crop(res_list):
    sort_list = sorted(res_list, key=lambda res: (res["nfcn_clf_ix"], res["score"]), reverse=True)
    return(sort_list[0])

def traslation(heat_coord, factor, fcn_stride=32):
    return(int(fcn_stride * heat_coord / factor))

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

# with open(os.path.join("test_images", "wronglabels.txt")) as file:
#     file_list = [(os.path.join(dataset_path,
#                                line.strip("\"\n").split("\\")[0],
#                                line.strip("\"\n").split("\\")[1],
#                                line.strip("\"\n").split("\\")[2]),
#                                line.strip("\"\n").split("\\")[1]) for line in file.readlines()]
# shuffle(file_list)

factors = np.empty(len(file_list))
scores = np.empty(len(file_list))
nfcns = np.empty(len(file_list), dtype=int)

crop_data = []

vgg16_orig_data = []
vgg16_crop_data = []

vgg19_orig_data = []
vgg19_crop_data = []

xce_orig_data = []
xce_crop_data = []

incv3_orig_data = []
incv3_crop_data = []

incrv2_orig_data = []
incrv2_crop_data = []

i_processed = 0

for filename, class_folder in file_list:

    img = image.load_img(filename)
    img = image.img_to_array(img)
    imgh, imgw = img.shape[0:2]

    res_list = process_image(filename, class_name_to_idx(class_folder), (imgh, imgw))
    crop = select_best_crop(res_list)  # factor, (hdim, wdim), (hcoordh, hcoordw), correct_fcn, score, cn_no
    coordh = traslation(crop["ix"][0], crop["factor"])
    coordw = traslation(crop["ix"][1], crop["factor"])
    rect_dim = int(295 / crop["factor"])

    factors[i_processed] = crop["factor"]
    scores[i_processed] = crop["score"]
    nfcns[i_processed] = crop["nfcn_clf_ix"]

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


    ix_label = class_name_to_idx(class_folder)

    def append_predictions(modelCLF, filename, croparray, input_size, preprocess, orig_export_list, crop_export_list):
        preds_original = predict_from_filename(modelCLF, filename, input_size, preprocess).flatten()
        porig_maxix, porig_maxname, porig_maxscore, porigin_labelscore = get_top1data(preds_original, class_folder)
        # n.classe corretta, prob, n.classe predetta, prob
        orig_export_list.append(dict(ix_label=int(ix_label), pr_label=float(porigin_labelscore), ix_predicted=int(porig_maxix), pr_predicted=float(porig_maxscore)))

        preds_crop = predict_from_imgarray(modelCLF, croparray, input_size, preprocess).flatten()
        pcrop_maxix, pcrop_maxname, pcrop_maxscore, pcrop_labelscore = get_top1data(preds_crop, class_folder)
        crop_export_list.append(dict(ix_label=int(ix_label), pr_label=float(pcrop_labelscore), ix_predicted=int(pcrop_maxix), pr_predicted=float(pcrop_maxscore)))

    # VGG16
    append_predictions(vgg16CLF, filename, img[coordh:coordh + rect_dim, coordw:coordw + rect_dim], (224, 224),
                       keras.applications.vgg16.preprocess_input, vgg16_orig_data, vgg16_crop_data)

    # VGG19
    append_predictions(vgg19CLF, filename, img[coordh:coordh + rect_dim, coordw:coordw + rect_dim], (224, 224),
                       keras.applications.vgg19.preprocess_input, vgg19_orig_data, vgg19_crop_data)
    # INCV3
    append_predictions(incv3CLF, filename, img[coordh:coordh + rect_dim, coordw:coordw + rect_dim], (299, 299),
                       keras.applications.inception_v3.preprocess_input, incv3_orig_data, incv3_crop_data)

    # INCRESNETV2
    append_predictions(incresv2CLF, filename, img[coordh:coordh + rect_dim, coordw:coordw + rect_dim], (299, 299),
                       keras.applications.inception_resnet_v2.preprocess_input, incrv2_orig_data, incrv2_crop_data)

    # XCEPTION
    append_predictions(xceptionCLF, filename, img[coordh:coordh + rect_dim, coordw:coordw + rect_dim], (299, 299),
                       keras.applications.xception.preprocess_input, xce_orig_data, xce_crop_data)

    crop_data.append(dict(filename=str(filename),
                label=str(class_folder),
                crop=dict(
                    factor=float(crop["factor"]),
                    heath=int(crop["heatmap_shape"][0]),
                    heatw=int(crop["heatmap_shape"][1]),
                    cropixh=int(crop["ix"][0]),
                    cropixw=int(crop["ix"][1]),
                    score=float(crop["score"]),
                    nfcn=int(crop["nfcn_clf_ix"]),
                    fcn=dict(vgg16FCN=str(crop["fcn_clf_ix"][0]),
                             xceptionFCN=str(crop["fcn_clf_ix"][1]),
                             incresv2FCN=str(crop["fcn_clf_ix"][2]),
                             incv3FCN=str(crop["fcn_clf_ix"][3])
                    )
                ),
                rect=dict(lower_left=(int(coordh), int(coordw)), side=int(rect_dim))
        )
    )

    i_processed += 1
    if i_processed % instances_per_folder == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S") + " started class " + str(i_processed//instances_per_folder) + " of " + str(folder_to_scan))

print("Averages: score", np.mean(scores), "nfcn", np.mean(nfcns), "factor", np.mean(factors))

pickle.dump(vgg16_orig_data, open("vgg16_orig_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(vgg16_crop_data, open("vgg16_crop_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(vgg19_orig_data, open("vgg19_orig_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(vgg19_crop_data, open("vgg19_crop_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(incv3_orig_data, open("incv3_orig_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(incv3_crop_data, open("incv3_crop_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(incrv2_orig_data, open("incrv2_orig_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(incrv2_crop_data, open("incrv2_crop_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(xce_orig_data, open("xce_orig_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(xce_crop_data, open("xce_crop_data.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(crop_data, open("cropsdata.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)