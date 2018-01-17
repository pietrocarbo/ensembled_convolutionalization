from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, AveragePooling2D
from keras.models import Model
from keras.models import model_from_json
from PIL import Image

import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
import matplotlib.ticker as plticker

import json
import os
import numpy as np

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

def prepare_str_file_architecture_syntax(filepath):
    model_str = str(json.load(open(filepath, "r")))
    model_str = model_str.replace("'", '"')
    model_str = model_str.replace("True", "true")
    model_str = model_str.replace("False", "false")
    model_str = model_str.replace("None", "null")
    return model_str


model = model_from_json(prepare_str_file_architecture_syntax("2017-12-24_acc77_vgg16/vgg16_architecture_2017-12-23_22-53-03.json"))
model.load_weights("2017-12-24_acc77_vgg16/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")
print("IMPORTED MODEL")
model.summary()

p_dim = model.get_layer("global_average_pooling2d_1").input_shape  # None,7,7,512
out_dim = model.get_layer("output_layer").get_weights()[1].shape[0]  # None,101
W, b = model.get_layer("output_layer").get_weights()
print("weights old shape", W.shape, "values", W)
print("biases old shape", b.shape, "values", b)

weights_shape = (1, 1, p_dim[3], out_dim)
print("weights new shape", weights_shape)

new_W = W.reshape(weights_shape)

last_pool_layer = model.get_layer("block5_pool")
last_pool_layer.outbound_nodes = []
model.layers.pop()
model.layers.pop()

for i, l in enumerate(model.layers):
    print(i, ":", l.name)

x = AveragePooling2D(pool_size=(7, 7))(last_pool_layer.output)

x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[new_W, b])(x)

model = Model(inputs=model.get_layer("input_1").input, outputs=x)

print("CONVOLUTIONALIZATED MODEL")
model.summary()


def idx_to_class_name(idx):
    with open(os.path.join('dataset-ethz101food', 'meta', 'classes.txt')) as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]

def save_map(heatmap, resultfname, is_input_img=False, grid=True):
    if is_input_img:
        pil_input = Image.open(heatmap)
        pil_input = pil_input.resize(img_size)
        imgarray = np.asarray(pil_input)
    else:
        pixels = 255 * (1.0 - heatmap)
        image = Image.fromarray(pixels.astype(np.uint8), mode='L')
        image = image.resize(img_size)
        imgarray = np.asarray(image)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    myInterval = 224.
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    if grid:
        ax.grid(which='major', axis='both', linestyle='-')

    if is_input_img:
        ax.imshow(imgarray)
    else:
        ax.imshow(imgarray, cmap='Greys_r', interpolation='none')

    nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(myInterval)))
    ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(myInterval)))
    for jj in range(ny):
        y = myInterval / 2 + jj * myInterval
        for ii in range(nx):
            x = myInterval / 2. + float(ii) * myInterval
            ax.text(x, y, '{:d}'.format((ii + jj * nx) + 1), color='tab:blue', ha='center', va='center')

    fig.savefig(resultfname)

max_upsampling_factor = 6
min_upsampling_factor = 1
top_n_show = 3
threshold_accuracy_stop = 0.80

# "dataset-ethz101food/train/cup_cakes/13821.jpg"
# "dataset-ethz101food/train/cannoli/1163058.jpg"
# "dataset-ethz101food/train/apple_pie/68383.jpg"
# "dataset-ethz101food/train/red_velvet_cake/1664681.jpg"
# "dataset-ethz101food/train/cup_cakes/46500.jpg"
input_class = "cannoli"
input_instance = "1706697"
input_set = "test"
input_filename = "dataset-ethz101food/" + input_set + "/" + input_class + "/" + input_instance + ".jpg"

if (os.path.exists(input_filename)):
    for upsampling_factor in range (min_upsampling_factor, max_upsampling_factor + 1):
        img_size = (224 * upsampling_factor, 224 * upsampling_factor)

        resultdir = os.path.join(os.getcwd(), "results", input_class + "_" + input_instance, "upsampled" + str(upsampling_factor) + "_heatmaps")

        input_image = image.load_img(input_filename, target_size=img_size)
        input_image = image.img_to_array(input_image)
        input_image_expandedim = np.expand_dims(input_image, axis=0)
        input_preprocessed_image = preprocess_input(input_image_expandedim)
        preds = model.predict(input_preprocessed_image)

        heatmaps_values = [preds[0, :, :, i] for i in range(101)]

        max_heatmaps = np.amax(heatmaps_values, axis=(1,2))

        top_n_idx = np.argsort(max_heatmaps)[-3:][::-1]

        if (os.path.isdir(resultdir)):
            print("Deleting older version of this folder...\n")
            shutil.rmtree(resultdir)

        os.makedirs(resultdir)

        save_map(input_filename, os.path.join(resultdir, input_class + "_" + input_instance + ".jpg"), is_input_img=True)
        for i, idx in enumerate(top_n_idx):
            name_class = idx_to_class_name(idx)
            print("Top", i, "category is: id", idx, "name", name_class)
            resultfname = os.path.join(resultdir, str(i + 1) + "_" + name_class + "_acc" + str(max_heatmaps[idx]) + ".jpg")
            save_map(heatmaps_values[idx], resultfname)
            print("heatmap saved at", resultfname)

        if (max_heatmaps[top_n_idx[0]] >= threshold_accuracy_stop):
            print ("\nThreshold accuracy stop detected -> " + str(max_heatmaps[top_n_idx[0]]) + " with upsampling_factor -> " + str(upsampling_factor))
            break
        else:
            print ("\nMax accuracy -> " + str(max_heatmaps[top_n_idx[0]]) + " with upsampling_factor -> " + str(upsampling_factor) + "\n")

        # ---------------------------------------------------------------------------
        # wrong way to get the most probable category
        # summed_heatmaps = np.sum(heatmaps_values, axis=(1, 2))
        # idx_classmax = np.argmax(summed_heatmaps).astype(int)
else:
    print ("The specified image " + input_filename + " not exist")