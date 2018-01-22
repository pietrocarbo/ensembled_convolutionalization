from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, AveragePooling2D, Flatten
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

import keras
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

def prepare_str_file_architecture_syntax(filepath):
    model_str = str(json.load(open(filepath, "r")))
    model_str = model_str.replace("'", '"')
    model_str = model_str.replace("True", "true")
    model_str = model_str.replace("False", "false")
    model_str = model_str.replace("None", "null")
    return model_str


#model = model_from_json(prepare_str_file_architecture_syntax("trained_models/top5_inv2resnet_flatten_acc70_2018-01-12/inception_resnet_2_architecture_2018-01-12_08-41-00.json"))
base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = Flatten(name='flatten')(base_model.output)
topnet_output = Dense(101, activation='softmax', name='output_layer')(x)
model = Model(inputs=base_model.input, outputs=topnet_output)
model.summary()


model.load_weights("trained_models/top5_inv2resnet_flatten_acc70_2018-01-12/inception_resnet_2_ft_weights_acc0.70_e2_2018-01-12_08-41-00.hdf5")
print("IMPORTED MODEL")
#Fino a qui tutto ok


p_dim = model.get_layer("conv_7b_ac").input_shape  # None,5,5,1536
out_dim = model.get_layer("output_layer").get_weights()[1].shape[0]  # None,101
W, b = model.get_layer("output_layer").get_weights()
print("weights old shape", W.shape, "values", W)
print("biases old shape", b.shape, "values", b)

weights_shape = (5, 5, p_dim[3], out_dim)
print("weights new shape", weights_shape)

new_W = W.reshape(weights_shape)

last_pool_layer = model.get_layer("conv_7b_ac")
last_pool_layer.outbound_nodes = []
model.layers.pop()
model.layers.pop()

for i, l in enumerate(model.layers):
    print(i, ":", l.name)

#x = AveragePooling2D(pool_size=(7, 7))(last_pool_layer.output)

x = Conv2D(101, (5, 5),strides=(1, 1), activation='softmax', padding='valid', weights=[new_W, b])(last_pool_layer.output)

model = Model(inputs=model.get_layer("input_1").input, outputs=x)

print("CONVOLUTIONALIZATED MODEL")
model.summary()


def idx_to_class_name(idx):
    with open(os.path.join('dataset-ethz101food', 'meta', 'classes.txt')) as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]

input_class = "cannoli"
input_instance = "1706697"
input_set = "test"
input_filename = "dataset-ethz101food/" + input_set + "/" + input_class + "/" + input_instance + ".jpg"

if (os.path.exists(input_filename)):
    #for upsampling_factor in range (min_upsampling_factor, max_upsampling_factor + 1):
    img_size = (224, 224)

    #resultdir = os.path.join(os.getcwd(), "results", input_class + "_" + input_instance, "upsampled" + str(upsampling_factor) + "_heatmaps")

    input_image = image.load_img(input_filename, target_size=img_size)
    input_image = image.load_img(input_filename)
    input_image = image.img_to_array(input_image)
    input_image_expandedim = np.expand_dims(input_image, axis=0)
    input_preprocessed_image = preprocess_input(input_image_expandedim)
    preds = model.predict(input_preprocessed_image)

        # heatmaps_values = [preds[0, :, :, i] for i in range(101)]
        #
        # max_heatmaps = np.amax(heatmaps_values, axis=(1,2))
        #
        # top_n_idx = np.argsort(max_heatmaps)[-3:][::-1]
        #
        # if (os.path.isdir(resultdir)):
        #     print("Deleting older version of the folder " + resultdir)
        #     shutil.rmtree(resultdir)
        # os.makedirs(resultdir)
        #
        # save_map(input_filename, os.path.join(resultdir, input_class + "_" + input_instance + ".jpg"), is_input_img=True)
        # for i, idx in enumerate(top_n_idx):
        #     name_class = idx_to_class_name(idx)
        #     print("Top", i, "category is: id", idx, ", name", name_class)
        #     resultfname = os.path.join(resultdir, str(i + 1) + "_" + name_class + "_acc" + str(max_heatmaps[idx]) + ".jpg")
        #     save_map(heatmaps_values[idx], resultfname)
        #     print("heatmap saved at", resultfname)
        #
        # if (max_heatmaps[top_n_idx[0]] >= threshold_accuracy_stop):
        #     print("Upsampling step " + str(upsampling_factor) + " finished -> accuracy threshold stop detected (accuracy: " + str(max_heatmaps[top_n_idx[0]]) + ")\n")
        #     # break
        # else:
        #     print("Upsampling step " + str(upsampling_factor) + " finished -> low accuracy, continuing... (accuracy: " + str(max_heatmaps[top_n_idx[0]]) + ")\n")
        #
        # # ---------------------------------------------------------------------------
        # # wrong way to get the most probable category
        # # summed_heatmaps = np.sum(heatmaps_values, axis=(1, 2))
        # # idx_classmax = np.argmax(summed_heatmaps).astype(int)
else:
    print ("The specified image " + input_filename + " does not exist")

plt.close('all')
