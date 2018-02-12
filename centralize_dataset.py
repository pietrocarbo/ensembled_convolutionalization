from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model
from keras.models import model_from_json
from PIL import Image

import shutil
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
import matplotlib.ticker as plticker

import json
import os
import numpy as np
import matplotlib.patches as patches

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

baseVGG16_1, last_layer, W, b = load_VGG16("trained_models/vgg16_architecture_2017-12-23_22-53-03.json", "trained_models/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")
x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(last_layer.output)
x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b])(x)
model = Model(inputs=baseVGG16_1.input, outputs=x)
# model.summary()

# baseVGG16_2, last_layer, W, b = load_VGG16("trained_models/top5_vgg16_acc77_2017-12-24/vgg16_architecture_2017-12-23_22-53-03.json", "trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")
# x = AveragePooling2D(pool_size=(7, 7))(last_layer.output)
# x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b])(x)
# upsample_fcnVGG16 = Model(inputs=baseVGG16_2.input, outputs=x)


def idx_to_class_name(idx):
    with open('trained_models/classes.txt') as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]

def class_name_to_idx(name):
    with open('trained_models/classes.txt') as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
        for i, label_name in enumerate(class_labels):
            if label_name == name:
                return i
        else:
            print("class idx not found!")
            exit(-1)

def save_map(heatmap, resultfname, input_size, tick_interval=None, is_input_img=False):
    if is_input_img:
        image = Image.open(heatmap)
        image = image.resize(input_size, Image.ANTIALIAS)
        imgarray = np.asarray(image)
    else:
        pixels = 255 * (1.0 - heatmap)
        image = Image.fromarray(pixels.astype(np.uint8), mode='L')
        image = image.resize(input_size, Image.NEAREST)
        imgarray = np.asarray(image)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if tick_interval:
        myInterval = tick_interval
        loc = plticker.MultipleLocator(base=myInterval)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.grid(which='both', axis='both', linestyle='-')

    if is_input_img:
        ax.imshow(imgarray)
    else:
        ax.imshow(imgarray, cmap='Greys_r', interpolation='none')

    if tick_interval:
        nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(myInterval)))
        ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(myInterval)))
        for jj in range(ny):
            y = myInterval / 2 + jj * myInterval
            for ii in range(nx):
                x = myInterval / 2. + float(ii) * myInterval
                ax.text(x, y, '{:d}'.format((ii + jj * nx) + 1), color='tab:blue', ha='center', va='center')

    fig.savefig(resultfname)
    plt.close()


input_set = "train"
input_class =  "cup_cakes" #"beignets" #"apple_pie" #"cannoli"  
input_instance = "46500" #"beignets_2918213" #"cannoli_1163058" #"apple_pie_68383"
input_filename = "test_images/"+ input_instance + ".jpg"

class_label = class_name_to_idx(input_class)

input = image.load_img(input_filename)
input = image.img_to_array(input)
print("image", input_filename, "has shape", input.shape)

mdim = min(input.shape[0],input.shape[1])
print("mdim = {}".format(mdim))
upsampling_step = 1.2
max_upsampling_factor = 3
min_upsampling_factor = 224./mdim
upsampling_factor = min_upsampling_factor

results = []

if (os.path.exists(input_filename)):
    while upsampling_factor < max_upsampling_factor:
        print("upsampling factor", upsampling_factor)

        img_size = (int(input.shape[0] * upsampling_factor), int(input.shape[1] * upsampling_factor))
        input_img = image.load_img(input_filename, target_size=img_size)
        input_img = image.img_to_array(input_img)
        input_image_expandedim = np.expand_dims(input_img, axis=0)
        input_preprocessed_image = preprocess_input(input_image_expandedim)

        preds = model.predict(input_preprocessed_image)
        print("input shape (height, width)", input_img.shape, "-> preds shape", preds.shape)

        heatmaps_values = preds[0, :, :, class_label]
        max_heatmap = np.amax(heatmaps_values)
        max_coordinates = np.unravel_index(np.argmax(heatmaps_values, axis=None), heatmaps_values.shape)
        print("max value", max_heatmap, "found at", max_coordinates)
        results.append((upsampling_factor, (preds.shape[1],preds.shape[2]), max_heatmap, max_coordinates))

        upsampling_factor = upsampling_factor * upsampling_step
else:
    print ("The specified image " + input_filename + " does not exist")


factor, (hdim,wdim), pred, (hcoordh, hcoordw) = max(results, key=lambda x:x[2])

input_img = input

def traslation(heat_coord):
    return(int(32 * heat_coord / factor)) #32 is the stride of the whole convolutive net

rect_dim = int(224/ factor)
stride = int(rect_dim / factor * 7)

coordh = coordinate_fix(hcoordh, hdim, input_img.shape[0]-rect_dim)
coordw = coordinate_fix(hcoordw, wdim, input_img.shape[1]-rect_dim)

coordh = traslation(hcoordh)
coordw = traslation(hcoordw)

print(coordh,coordw)

print("\nMax confidence", pred, "found at upscale factor", factor, ";",
      "heatmap cell", (hcoordh, hcoordw), "in range [", hdim, ",", wdim, "] ->",
      "relative img point", (coordh, coordw), "in range [", input_img.shape[0], ",", input_img.shape[1], "]")


fig, ax = plt.subplots(1)

ax.imshow(input_img / 255.)

#rect_dim = int(input_img.shape[0] / factor)
#half_rect_dim = int(112/factor) #int(96/factor) #"half" is in fact 3/7 of 224

#circle = patches.Circle((coordw, coordh), int(16/factor))
rect = patches.Rectangle((coordw, coordh), rect_dim, rect_dim, linewidth=3, edgecolor='b', facecolor='none')


ax.add_patch(rect)
#ax.add_patch(circle)

plt.show()
