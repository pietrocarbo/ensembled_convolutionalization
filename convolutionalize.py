from keras.layers import Conv2D, AveragePooling2D
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


def load_VGG16(architecture_path, weigths_path, debug=True):
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

baseVGG16_1, last_layer, W, b = load_VGG16("trained_models/top5_vgg16_acc77_2017-12-24/vgg16_architecture_2017-12-23_22-53-03.json", "trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")
x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(last_layer.output)
x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b])(x)
overlap_fcnVGG16 = Model(inputs=baseVGG16_1.input, outputs=x)


baseVGG16_2, last_layer, W, b = load_VGG16("trained_models/top5_vgg16_acc77_2017-12-24/vgg16_architecture_2017-12-23_22-53-03.json", "trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")
x = AveragePooling2D(pool_size=(7, 7))(last_layer.output)
x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b])(x)
upsample_fcnVGG16 = Model(inputs=baseVGG16_2.input, outputs=x)


def idx_to_class_name(idx):
    with open(os.path.join('dataset-ethz101food', 'meta', 'classes.txt')) as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]

def save_map(heatmap, resultfname, tick_interval=None, is_input_img=False):
    if is_input_img:
        pil_input = Image.open(heatmap)
        # pil_input = pil_input.resize(img_size)
        imgarray = np.asarray(pil_input)
    else:
        pixels = 255 * (1.0 - heatmap)
        image = Image.fromarray(pixels.astype(np.uint8), mode='L')
        # image = image.resize(img_size)
        imgarray = np.asarray(image)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if tick_interval:
        myInterval = tick_interval
        loc = plticker.MultipleLocator(base=myInterval)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.grid(which='both', axis='both', linestyle='-')
        nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(myInterval)))
        ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(myInterval)))
        for jj in range(ny):
            y = myInterval / 2 + jj * myInterval
            for ii in range(nx):
                x = myInterval / 2. + float(ii) * myInterval
                ax.text(x, y, '{:d}'.format((ii + jj * nx) + 1), color='tab:blue', ha='center', va='center')

    if is_input_img:
        ax.imshow(imgarray)
    else:
        ax.imshow(imgarray, cmap='Greys_r', interpolation='none')

    fig.savefig(resultfname)
    plt.close()


# dataset-ethz101food/test/apple_pie/101251.jpg
# "dataset-ethz101food/train/cup_cakes/13821.jpg"
# "dataset-ethz101food/train/cannoli/1163058.jpg"
# "dataset-ethz101food/train/apple_pie/68383.jpg"
# "dataset-ethz101food/train/red_velvet_cake/1664681.jpg"
# "dataset-ethz101food/train/cup_cakes/46500.jpg"
#  test/cannoli/1706697.jpg -> potenzialmente due cibi nella stessa foto
#  test/beignets/2918213.jpg -> cibo in basso a sinistra
#  test/french_fries/796641.jpg -> cibo in alto e molto piccolo
#  train/pizza/2687575.jpg -> potenzialmente lo stesso cibo in più posti
#  train/spaghetti_bolognese/1331330.jpg -> cibo i ndue posizioni diverse
#  train/cup_cakes/9256.jpg -> potrebbe dare buoni frutti con upsampling anche a 8
#  train/cup_cakes/451074.jpg -> molto difficile da trovare ed è lontano ( sopra il bus)
#  train/cup_cakes/1265596.jpg -> cibo in più posti
input_set = "train"
input_class = "cup_cakes"
input_instance = "46500"
input_filename = "dataset-ethz101food/" + input_set + "/" + input_class + "/" + input_instance + ".jpg"

threshold_accuracy_stop = 0.80
max_upsampling_factor = 7
min_upsampling_factor = 1
top_n_show = 5

model = overlap_fcnVGG16

if (os.path.exists(input_filename)):
    if model == overlap_fcnVGG16:
        input_image = image.load_img(input_filename)
        input_image = image.img_to_array(input_image)
        input_image_expandedim = np.expand_dims(input_image, axis=0)
        input_preprocessed_image = preprocess_input(input_image_expandedim)

        preds = model.predict(input_preprocessed_image)
        print("input img shape (height, width)", input_image.shape, "preds shape", preds.shape)

        heatmaps_values = [preds[0, :, :, i] for i in range(101)]
        max_heatmaps = np.amax(heatmaps_values, axis=(1, 2))
        top_n_idx = np.argsort(max_heatmaps)[-top_n_show:][::-1]

        resultdir = os.path.join(os.getcwd(), "results", input_class + "_" + input_instance + "_standard-heatmaps")
        if (os.path.isdir(resultdir)):
            print("Deleting older version of the folder " + resultdir)
            shutil.rmtree(resultdir)
        os.makedirs(resultdir)

        save_map(input_filename, os.path.join(resultdir, input_class + "_" + input_instance + ".jpg"), is_input_img=True)
        for i, idx in enumerate(top_n_idx):
            name_class = idx_to_class_name(idx)
            print("Top", i, "category is: id", idx, ", name", name_class)
            resultfname = os.path.join(resultdir, str(i + 1) + "_" + name_class + "_acc" + str(max_heatmaps[idx]) + ".jpg")
            save_map(heatmaps_values[idx], resultfname)
            print("heatmap saved at", resultfname)

    else:
        for upsampling_factor in range (min_upsampling_factor, max_upsampling_factor):
            img_size = (255 * upsampling_factor, 255 * upsampling_factor)
            input_image = image.load_img(input_filename, target_size=img_size)
            input_image = image.img_to_array(input_image)
            input_image_expandedim = np.expand_dims(input_image, axis=0)
            input_preprocessed_image = preprocess_input(input_image_expandedim)

            preds = model.predict(input_preprocessed_image)
            print("input img shape (height, width)", input_image.shape, "preds shape", preds.shape)

            heatmaps_values = [preds[0, :, :, i] for i in range(101)]
            max_heatmaps = np.amax(heatmaps_values, axis=(1,2))
            top_n_idx = np.argsort(max_heatmaps)[-top_n_show:][::-1]

            resultdir = os.path.join(os.getcwd(), "results", input_class + "_" + input_instance, "upsampled" + str(upsampling_factor) + "-heatmaps")
            if (os.path.isdir(resultdir)):
                print("Deleting older version of the folder " + resultdir)
                shutil.rmtree(resultdir)
            os.makedirs(resultdir)

            save_map(input_filename, os.path.join(resultdir, input_class + "_" + input_instance + ".jpg"), tick_interval=224, is_input_img=True)
            for i, idx in enumerate(top_n_idx):
                name_class = idx_to_class_name(idx)
                print("Top", i, "category is: id", idx, ", name", name_class)
                resultfname = os.path.join(resultdir, str(i + 1) + "_" + name_class + "_acc" + str(max_heatmaps[idx]) + ".jpg")
                save_map(heatmaps_values[idx], resultfname, tick_interval=224)
                print("heatmap saved at", resultfname)

            if (max_heatmaps[top_n_idx[0]] >= threshold_accuracy_stop):
                print("Upsampling step " + str(upsampling_factor) + " finished -> accuracy threshold stop detected (accuracy: " + str(max_heatmaps[top_n_idx[0]]) + ")\n")
                break
            else:
                print("Upsampling step " + str(upsampling_factor) + " finished -> low accuracy, continuing... (accuracy: " + str(max_heatmaps[top_n_idx[0]]) + ")\n")

        # ---------------------------------------------------------------------------
        # wrong way to get the most probable category
        # summed_heatmaps = np.sum(heatmaps_values, axis=(1, 2))
        # idx_classmax = np.argmax(summed_heatmaps).astype(int)
else:
    print ("The specified image " + input_filename + " does not exist")
