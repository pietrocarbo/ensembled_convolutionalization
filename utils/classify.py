import os
import sys
import argparse
import numpy as np
import keras
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image

parser = argparse.ArgumentParser(description='script used to classify food images')
parser.add_argument('architecture_fn', type=str, help='file name containing the model architecture')
parser.add_argument('--weights_fn', type=str, help='file name containing the model weights to load, if NOT given assume that architecture_fn contain also the weights')
parser.add_argument('--input_size', type=int, default=224, help='the shape to resize input images. Default: 224x224')
parser.add_argument('--preprocess_func', type=str, default=keras.applications.vgg16.preprocess_input, help='the preprocess function to apply to input images')
parser.add_argument('input', type=str, help='the path of the image or the directory containing images to classify')
parser.add_argument('-topN', type=int, default=5, help='print the top-n predictions')
args = parser.parse_args()

with open(os.path.join('dataset-ethz101food', 'meta', 'classes.txt')) as file:
    class_labels = [line.strip('\n') for line in file.readlines()]

# Output classification results on image_file in a string format
def classify_image(image_file, preprocess, input_size):
    img = image.load_img(image_file, target_size=input_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess(x)
    preds = model.predict(x).flatten()
    sorted_classes_ix = np.argsort(preds).flatten()
    result = 'Image ' + str(image_file) + ' results:\n'
    for i in range(args.top):
        idx = sorted_classes_ix[-i]
        result += '\t prediction {:d}/{:d}  -->  classified as: {:s}({:d}) with a confidence of {:f}\n'.format(i + 1, args.top, class_labels[idx], idx, preds[idx])
    return result

if os.path.exists(args.architecture_fn) and os.path.exists(args.input) and 0 < args.top < 101:

    # Model assembling
    if "weights_fn" in args:
        model = model_from_json(args.architecture_fn)
        model = model.load_weights(args.weights_fn)
    else:
        model = load_model(args.architecture_fn)

    input_size = (args.input_size, args.input_size)
    preprocess_func = args.preprocess_func
    if not 'input_size' in args or not 'preprocess_func' in args:
        print("Input size or preprocess function not given. Using defaults: shape (224, 224) ang VGG16 preprocessing")

    # Classification
    if os.path.isdir(args.input):
        output = [classify_image(os.path.join(args.input, image), preprocess_func, input_size) for image in os.listdir(args.input)]
        print(*output, sep='\n')
    else:
        print(classify_image(args.input, preprocess_func, input_size))

else:
    print("One or more parameters are not correct")
    sys.exit(-1)
