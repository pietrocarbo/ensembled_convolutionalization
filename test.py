from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D
from keras.models import Model
from keras.models import model_from_json
import json

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

global_pool_layer = model.get_layer("global_average_pooling2d_1")
global_pool_layer.outbound_nodes = []
model.layers.pop()

for i, l in enumerate(model.layers):
    print(i, ":", l.name)

model.layers[19].output_shape

pool_output = global_pool_layer.output

output = Conv2D(out_dim, p_dim[3], strides=1, activation='relu', padding='valid', weights=[new_W, b], input_shape=global_pool_layer.output_shape)(pool_output)
# error here!
print("conv2d attached to global pooling")

model = Model(inputs=model.get_layer("input_1").input, outputs=output)

print("CONVOLUTIONALIZATED MODEL")
model.summary()