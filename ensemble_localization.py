from keras.preprocessing import image
import keras
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Conv2D, AveragePooling2D, Dense, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dropout
# non-graphical plot backend
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
import matplotlib.patches as patches
import time
import pickle
import os
import numpy as np
import PIL
from PIL import Image
from utils.labels_ix_mapping import ix_to_class_name, class_name_to_idx
dataset_path = "dataset-ethz101food"


# Function used to convolutionalize the VGG16 architecture
def convolutionalize_vgg16():
    vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))

    x = GlobalAveragePooling2D(name="global_average_pooling2d_1")(vgg16.output)
    out = Dense(101, activation='softmax', name='output_layer')(x)
    vgg16 = Model(inputs=vgg16.input, outputs=out)

    vgg16.load_weights("trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")

    p_dim = vgg16.get_layer("global_average_pooling2d_1").input_shape
    out_dim = vgg16.get_layer("output_layer").get_weights()[1].shape[0]
    W, b = vgg16.get_layer("output_layer").get_weights()

    weights_shape = (1, 1, p_dim[3], out_dim)

    W = W.reshape(weights_shape)

    last_layer = vgg16.get_layer("block5_pool")   # name of last VGG16 Keras layer
    last_layer.outbound_nodes = []
    vgg16.layers.pop()
    vgg16.layers.pop()

    x = AveragePooling2D(pool_size=(9, 9), strides=(1, 1))(last_layer.output)
    x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b], name="conv2d_fcn")(x)
    vgg16 = Model(inputs=vgg16.input, outputs=x)

    return vgg16


# Function used to convolutionalize the Xception architecture
def convolutionalize_xception():
    xce = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(None, None, 3))

    x = GlobalAveragePooling2D(name="global_average_pooling2d_1")(xce.output)
    out = Dense(101, activation='softmax', name='output_layer')(x)
    xce = Model(inputs=xce.input, outputs=out)

    xce.load_weights("trained_models/top1_xception_acc80_2017-12-25/xception_ft_weights_acc0.81_e9_2017-12-24_13-00-22.hdf5")

    p_dim = xce.get_layer("global_average_pooling2d_1").input_shape
    out_dim = xce.get_layer("output_layer").get_weights()[1].shape[0]
    W, b = xce.get_layer("output_layer").get_weights()

    weights_shape = (1, 1, p_dim[3], out_dim)

    W = W.reshape(weights_shape)

    last_layer = xce.get_layer("block14_sepconv2_act")
    last_layer.outbound_nodes = []
    xce.layers.pop()
    xce.layers.pop()

    x = AveragePooling2D(pool_size=(10, 10), strides=(1, 1))(last_layer.output)
    x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b], name="conv2d_fcn")(x)
    xce = Model(inputs=xce.input, outputs=x)

    return xce


# Function used to convolutionalize the InceptionResNetV2 architecture
def convolutionalize_incresv2():
    incresv2 = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                        input_shape=(None, None, 3))
    x = GlobalAveragePooling2D(name="global_average_pooling2d_1")(incresv2.output)
    out = Dense(101, activation='softmax', name='output_layer')(x)
    incresv2 = Model(inputs=incresv2.input, outputs=out)
    incresv2.load_weights("trained_models/top2_incresnetv2_acc79_2017-12-22/incv2resnet_ft_weights_acc0.79_e4_2017-12-21_09-02-16.hdf5")

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

# Function used to convolutionalize the InceptionV3 architecture
def convolutionalize_incv3():
    incv3 = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                        input_shape=(None, None, 3))
    x = GlobalAveragePooling2D()(incv3.output)
    x = Dense(1024, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005),
              bias_regularizer=l2(.0005), name="fully-connected1")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(name="batch-normalization-1")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, kernel_initializer='he_uniform', bias_initializer="he_uniform", kernel_regularizer=l2(.0005),
              bias_regularizer=l2(.0005), name="fully-connected2")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(name="batch-normalization-2")(x)
    x = Dropout(0.5)(x)
    out = Dense(101, kernel_initializer='he_uniform', bias_initializer="he_uniform", activation='softmax',
                name='output_layer')(x)
    incv3 = Model(inputs=incv3.input, outputs=out, name="output_layer")
    incv3.load_weights("trained_models/top3_inceptionv3_acc79_2017-12-27/inceptionv3_ft_weights_acc0.79_e10_2017-12-25_22-10-02.hdf5")

    W1, b1 = incv3.get_layer("fully-connected1").get_weights()
    W2, b2 = incv3.get_layer("fully-connected2").get_weights()
    W3, b3 = incv3.get_layer("output_layer").get_weights()

    BN1 = incv3.get_layer("batch-normalization-1").get_weights()
    BN2 = incv3.get_layer("batch-normalization-2").get_weights()

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

# FCNs declarations
vgg16FCN = convolutionalize_vgg16()

xceptionFCN = convolutionalize_xception()

incresv2FCN = convolutionalize_incresv2()

incv3FCN = convolutionalize_incv3()

# CLFs (classifiers) declarations
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

def get_top1data(preds, additionalClassIx):
    maxix = np.argmax(preds)
    return (maxix, ix_to_class_name(maxix), preds[maxix], preds[class_name_to_idx(additionalClassIx)])

# ensemble declaration
kernel_sizes = [288, 295, 299, 299]
FCNs = [vgg16FCN, xceptionFCN, incresv2FCN, incv3FCN]
preprocess_func = [  keras.applications.vgg16.preprocess_input
                   , keras.applications.xception.preprocess_input
                   , keras.applications.inception_resnet_v2.preprocess_input
                   , keras.applications.inception_v3.preprocess_input]

# Formula to comput the output size after application of a convolutional kernel
def dim_size(w, k, s):
  return ((w - k) // s + 1)

# Ensemble image processing at different scales and heatmaps informations extraction.
# Returns a list with the best heatmap element and relative score at each scale
def process_image(input_fn, input_cix, img_shape, upsampling_step = 1.2, max_scale_factor = 3.0):
    results = []
    if (os.path.exists(input_fn)):
        base_kernel_size = 295 # any of the kernels would do
        scale_factor = float(base_kernel_size) / min(img_shape[0], img_shape[1])
        maxcn = 0
        
        while scale_factor < max_scale_factor and maxcn < 4:
            # we define the expected heatmap dimension at this scale using the kernel size of the first FCN
            base_kernel_size = kernel_sizes[0]
            heatmap_h = dim_size(round(img_shape[0]*scale_factor), base_kernel_size, 32)
            heatmap_w = dim_size(round(img_shape[1]*scale_factor), base_kernel_size, 32)
            
            # we search, at this scale, the heatmap element (crop) that maximize the label for highest number of FNCs
            heatmaps = []
            bool_cix_maps = []
            for ix, fcn in enumerate(FCNs):
                # we adjust the input size for each FCN to get comparable (equal-size) heatmaps
                scaled_w = kernel_sizes[ix] + (heatmap_w - 1) * 32
                scaled_h = kernel_sizes[ix] + (heatmap_h - 1) * 32

                heatmaps.append(predict_from_filename(fcn, input_fn, (scaled_h, scaled_w), preprocess_func[ix])[0])

                bool_cix_map = np.argmax(heatmaps[-1], axis=2) == input_cix   # boolean map that indicate label maximization
                bool_cix_maps.append(bool_cix_map)

            # ncix_max_map is a int map, that will have the number of FCN that maximize the label (values from 0 to 4)
            ncix_max_map = np.zeros(bool_cix_maps[-1].shape, dtype=int)
            for bool_cix_map in bool_cix_maps:
                ncix_max_map += bool_cix_map

            maxcn = np.max(ncix_max_map)
            positions = np.nonzero(ncix_max_map == maxcn)  # tuple with the indices of max_cn relative to ncix_max_map
            positions = list(zip(positions[0], positions[1]))

            def sum_crop_score(x):
                res = 0
                for map in heatmaps:
                    res += map[x[0], x[1], input_cix]
                return res

            best_crop_ix = max(positions, key=sum_crop_score)
            best_crop_score = sum_crop_score(best_crop_ix) / 4
            correct_fcn = [bool_cix_map[best_crop_ix[0], best_crop_ix[1]] for bool_cix_map in bool_cix_maps]

            results.append({"factor": scale_factor, "heatmap_shape": heatmaps[-1].shape[0:2], "ix": best_crop_ix,
                            "score": best_crop_score, "nfcn_clf_ix": maxcn, "fcn_clf_ix": correct_fcn})

            # step to the next scale
            scale_factor *= upsampling_step

    else:
        print ("The image file " + str(input_fn) + " does not exist")

    return results

def select_best_crop(res_list):
    return max(res_list, key=lambda res: (res["nfcn_clf_ix"], res["score"]))

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

# for statics
factors = np.empty(len(file_list))
scores = np.empty(len(file_list))
nfcns = np.empty(len(file_list), dtype=int)

# for exporting crops coordinates
crops_list = []

i_processed = 0
for filename, class_folder in file_list:

    img = image.load_img(filename)
    img = image.img_to_array(img)
    imgh, imgw = img.shape[0:2]

    res_list = process_image(filename, class_name_to_idx(class_folder), (imgh, imgw))
    crop = select_best_crop(res_list)
    coordh = traslation(crop["ix"][0], crop["factor"])
    coordw = traslation(crop["ix"][1], crop["factor"])
    rect_dim = int(295 / crop["factor"])

    factors[i_processed] = crop["factor"]
    scores[i_processed] = crop["score"]
    nfcns[i_processed] = crop["nfcn_clf_ix"]

    # debug-purpose
    print("Max confidence", crop["score"], "at scale", crop["factor"],
          "heatmap crop", (crop["ix"][0], crop["ix"][1]),
          "in range [" + str(crop["heatmap_shape"][0]) + ", " + str(crop["heatmap_shape"][1]) + "] ->",
          "relative img point", (coordh, coordw), "in range [" + str(imgh) + ", " + str(imgw) + "]")
    # fig, ax = plt.subplots(1)
    # ax.imshow(img / 255.)
    # ax.set_title(class_folder)
    # rect = patches.Rectangle((coordw, coordh), rect_dim, rect_dim, linewidth=2, edgecolor='g', facecolor='none')
    # ax.add_patch(rect)
    # plt.show()

    ix_label = class_name_to_idx(class_folder)

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
pickle.dump(crops_list, open("cropsdata.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)