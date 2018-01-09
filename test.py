from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, AveragePooling2D
from keras.models import Model
from keras.models import model_from_json
import json
import numpy as np
import os

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

x = Conv2D(101, (1, 1), strides=(1, 1), activation='relu', padding='valid', weights=[new_W, b])(x)

model = Model(inputs=model.get_layer("input_1").input, outputs=x)

print("CONVOLUTIONALIZATED MODEL")
model.summary()


input_file = "test_images/pizza.jpg"
img = image.load_img(input_file)  # , target_size=(224, 224)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)

preds = preds.flatten()
sorted_classes_ix = np.argsort(preds).flatten()
result = 'Image ' + str(input_file) + ' results:\n'
with open(os.path.join('dataset-ethz101food', 'meta', 'classes.txt')) as file:
    class_labels = [line.strip('\n') for line in file.readlines()]
for i in range(5):
    idx = sorted_classes_ix[-i]
    result += '\t prediction {:d}/{:d}  -->  classified as: {:s}({:d}) with a confidence of {:f}\n'.format(i + 1, 5, class_labels[idx], idx, preds[idx])
