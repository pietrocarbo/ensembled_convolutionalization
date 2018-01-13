from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, AveragePooling2D
from keras.models import Model
from keras.models import model_from_json
from PIL import Image
import json
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

factor = 6
img_size = (224*factor, 224*factor)
class_idx = 29   # 12: cannoli, 83: red velvet

# input_file = "dataset-ethz101food/train/cannoli/1163058.jpg"
# input_file = "dataset-ethz101food/train/apple_pie/68383.jpg"
# input_file = "dataset-ethz101food/train/red_velvet_cake/1664681.jpg"
input_file = "dataset-ethz101food/train/cup_cakes/46500.jpg"
img = image.load_img(input_file, target_size=img_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)

unflatten_pred_size = preds.shape

heatmaps_values = [preds[0, :, :, i] for i in range(101)]

pixels = 255 * (1.0 - heatmaps_values[class_idx])
im = Image.fromarray(pixels.astype(np.uint8), mode='L')
im = im.resize(img_size)
im.show()

preds = preds.flatten()
print("preds size", unflatten_pred_size, "flattened into", preds.shape, "values\n", preds)