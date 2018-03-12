from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model
from keras.models import model_from_json
from PIL import Image

from lib.plot_utils import plot_confusion_matrix
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-bright')
import matplotlib.ticker as plticker

import json
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
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


baseVGG16_1, last_layer, W, b = load_VGG16(
    "trained_models/top5_vgg16_acc77_2017-12-24/vgg16_architecture_2017-12-23_22-53-03.json",
    "trained_models/top5_vgg16_acc77_2017-12-24/vgg16_ft_weights_acc0.78_e15_2017-12-23_22-53-03.hdf5")
x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(last_layer.output)
x = Conv2D(101, (1, 1), strides=(1, 1), activation='softmax', padding='valid', weights=[W, b])(x)
overlap_fcnVGG16 = Model(inputs=baseVGG16_1.input, outputs=x)


def idx_to_class_name(idx):
    with open(os.path.join('dataset-ethz101food', 'meta', 'classes.txt')) as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]


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


top_n_show = 1
model = overlap_fcnVGG16

base_dir = "dataset-ethz101food/"
# with open("test_images/smallfoodpics.txt") as file:
#     file_list = [base_dir + line.strip('"\n') for line in file.readlines()]
# # for file in file_list:
# #     print(file)
with open("dataset-ethz101food/meta/test_revisited.txt") as file:
    file_list = [base_dir + "test/" + line.strip('\n') + ".jpg" for line in file.readlines()]


y_test = [y//250 for y in range(250*101)]

# dict_augmentation = dict(preprocessing_function=preprocess_input)
# def images_generator(directory, batch_size):
#     """Replaces Keras' native ImageDataGenerator."""
#     i = 0
#     file_list = os.listdir(directory)
#     while True:
#         image_batch = []
#         for b in range(batch_size):
#             if i == len(file_list):
#                 i = 0
#                 random.shuffle(file_list)
#             sample = file_list[i]
#             i += 1
#             image = cv2.resize(cv2.imread(sample[0]), INPUT_SHAPE)
#             image_batch.append((image.astype(float) - 128) / 128)
#
#         yield np.array(image_batch)
#
#
# for ix_pred, input_filename in enumerate(file_list):
#     if (os.path.exists(input_filename)):
#         input_image = image.load_img(input_filename)
#         img_original_size = input_image.size
#         input_image = image.img_to_array(input_image)
#         input_image_expandedim = np.expand_dims(input_image, axis=0)
#         input_preprocessed_image = preprocess_input(input_image_expandedim)
#
#         preds = model.predict(input_preprocessed_image, batch_size=1)
#         # print("input img shape (height, width)", input_image.shape, "preds shape", preds.shape)
#
#         heatmaps_values = [preds[0, :, :, i] for i in range(101)]
#         max_heatmaps = np.amax(heatmaps_values, axis=(1, 2))
#         top_n_ix = np.argsort(max_heatmaps)[-top_n_show:][::-1]
#
#         # for i, ix in enumerate(top_n_ix):
#             # name_class = idx_to_class_name(ix)
#         y_pred[ix_pred] = top_n_ix[0]
#         if ix_pred % 250 == 0:
#             print("Image processed: number", ix_pred, "name", input_filename)  #,"Top", i+1, "category is: id", ix, ", name", name_class)
#     else:
#         print("The specified image " + input_filename + " does not exist")
#
# with open('y_pred', 'wb') as f:
#     pickle.dump(y_pred, f)

with open('y_pred', 'rb') as f:
    y_pred = pickle.load(f)

# print("length pred", len(y_pred), "test", len(y_test))
# for i in y_pred:
#     print(i)

y_pred.insert(15863, 63)
y_pred.pop()

# print("length pred", len(y_pred), "test", len(y_test))
# for i in y_pred:
#     print(i)

y_pred = np.asarray(y_pred).astype(int)
y_test = np.asarray(y_test).astype(int)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names = [idx_to_class_name(i) for i in range(101)]

plt.figure(figsize=(8, 6), dpi=400)
plt.rcParams.update({'font.size': 2})
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig("results/testSetConfusionMatrixNN.jpg")

for i in range(101):
    print("Category", class_names[i], "correct preds", cnf_matrix[i][i])