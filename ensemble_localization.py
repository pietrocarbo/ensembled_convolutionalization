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
import matplotlib.ticker as plticker

import json
import pickle
import os
import numpy as np
import matplotlib.patches as patches

import PIL
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

dataset_path = "dataset-ethz101food/"

def ix_to_class_name(idx):
    with open(dataset_path + "meta/classes.txt") as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]

def class_name_to_idx(name):
    with open(dataset_path + "meta/classes.txt") as file:
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

    W1 = W1.reshape((1, 1, 2048, 1024))
    W2 = W2.reshape((1, 1, 1024, 512))
    W3 = W3.reshape((1, 1, 512, 101))

    last_layer = incv3.get_layer("mixed10")
    last_layer.outbound_nodes = []
    for i in range(10):
        incv3.layers.pop()

    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1))(last_layer.output)

    x = Conv2D(1024, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W1, b1],
               name="conv2d_fcn1")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(512, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W2, b2],
               name="conv2d_fcn2")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
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
# vgg16FCN.summary()

xceptionFCN = convolutionalize_architecture(architecture_path="trained_models/xception_architecture_2017-12-24_13-00-22.json",
                                            weigths_path="trained_models/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5",
                                            last_layer_name="block14_sepconv2_act",
                                            pool_size=10)
# xceptionFCN.summary()


incresv2FCN = convolutionalize_incresv2()
# incresv2FCN.summary()
incv3FCN = convolutionalize_incv3()
# incv3FCN.summary()

def predict(model, filename, input_size, preprocess):
    input_img = image.load_img(filename, target_size=input_size)
    input_img = image.img_to_array(input_img)
    input_image_expandedim = np.expand_dims(input_img, axis=0)
    input_preprocessed_image = preprocess(input_image_expandedim)
    preds = model.predict(input_preprocessed_image)
    return preds

max_scale_factor = 3 
upsampling_step = 1.2

#inc_list = [0, 32, 64, 96, 160, 224, 288, 384, 512]
kernel_sizes = [288, 295, 299, 299]
FCNs = [vgg16FCN, xceptionFCN, incresv2FCN, incv3FCN]
preprocess_func = [  keras.applications.vgg16.preprocess_input
                   , keras.applications.xception.preprocess_input
                   , keras.applications.inception_resnet_v2.preprocess_input
                   , keras.applications.inception_v3.preprocess_input]

def dim_size(w,k,s):
  return((w-k)//s)
        
def process_image(input_fn, input_cix):
    results = []
    if (os.path.exists(input_fn)):
        img = image.load_img(input_fn)
        scale_factor = float(base_kernel_size) / min(img.shape[0], img.shape[1])

        while scale_factor < max_scale_factor:
            #definiamo la dimensione attesa della heatmap a questa scala
            #usando il kernel del primo fcn
            #riscaleremo poi le immagini alle dimensioni giuste per gli
            #altri cropper, im modo da avere in output un heatmap della
            #dimensione prevista
            base_kernel_size = kernel_sizes[0]
            heat_map_w = dim_size(int(round(img.shape[0]*scale_factor)),base_kernel_size,32)
            heat_map_h = dim_size(int(round(img.shape[1]*scale_factor)),base_kernel_size,32)    
            print(heat_map_w,heat_map_h)
            
            # cerchiamo a questa scala il crop che ha il numero massimo di croppatori che lo classificano come classe input_ix
            heatmaps = []
            bool_cix_maps = []
            for ix, fcn in enumerate(FCNs):
                new_dim_w = kernel_sizes[ix]+(heat_map_w-1)*32
                new_dim_h = kernel_sizes[ix]+(heat_map_h-1)*32
                heatmaps.append(predict(fcn, input_fn, (new_dim_w,new_dim_h), preprocess_func[ix])[0])

                bool_cix_map = np.argmax(heatmaps[-1], axis=2) == input_cix
                bool_cix_maps.append(bool_cix_map)

            # ncix_max_map e la mappa che mi dice quanti croppatori hanno classificato un crop come input_ix. ha valori da 0 a 4 quindi
            ncix_max_map = np.zeros(bool_cix_maps[-1].shape, dtype=int)
            for bool_cix_map in bool_cix_maps:
                ncix_max_map += bool_cix_map

            maxcn = np.max(ncix_max_map)    # valore massimo della mappa ncix_max_map
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
            best_crop_score = sum_crop_score(best_crop_ix) / 4

            results.append((scale_factor, heatmaps[-1].shape[0:1], best_crop_ix, best_crop_score, maxcn))
            # si passa ora alla prossima scala
            scale_factor *= upsampling_step
        for result in results:
            print(result)
        print("\n")

    else:
        print ("The image file " + str(input_fn) + " does not exist")

    return results

def best_crop(res_list):
    #seleziona il best crop della return list
    def mykey(factor, dims, hdims, score, cn_no):
      return (cn_no, score)
    sort_list = sorted(res_list, key=mykey)
    return(sort_list[-1])

def traslation(heat_coord, factor, fcn_stride=32):
    return(int(fcn_stride * heat_coord / factor))
    
        # # if hdim > 1 or wdim > 1:
        # #     countCnb +=1
        # #     factorCnb += rst_list[-1][0]
        # #     print("n.", count, "img:", filename, "rst_list (len", len(rst_list), ")", rst_list)
        # rect_dim = int(fcn_window / factor)
        # coordh = traslation(hcoordh, factor)
        # coordw = traslation(hcoordw, factor)
        
set = "test"
class_folders = os.listdir(dataset_path + set)
folder_to_scan = 5
instances_per_folder = 1
# "dataset-ethz101food/train/cup_cakes/46500.jpg"
for i_folder, class_folder in enumerate(class_folders[0:folder_to_scan]):
    instances = os.listdir("dataset-ethz101food/" + set + "/" + class_folder)
    for i_instance, instance in enumerate(instances[0:instances_per_folder]):
        filename = "dataset-ethz101food/" + set + "/" + class_folder + "/" + instance
        res_list = process_image(filename, class_name_to_idx(class_folder))
        factor, (hdim, wdim), (hcoordh, hcoordw), score, cn_no = best_crop(res_list)
        coordh = traslation(hcoordh, factor)
        coordw = traslation(hcoordw, factor)
        rect_dim = int(295 / factor)

        if True: #set to True to draw
          img = image.load_img(input_fn)
          fig, ax = plt.figure()
          ax.imshow(img / 255.)
          rect = patches.Rectangle((coordw, coordh), rect_dim, rect_dim, linewidth=1, edgecolor='r', facecolor='none')
          ax.add_patch(rect)
          plt.show()
