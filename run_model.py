import os
import argparse
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

parser = argparse.ArgumentParser(description='script used to classify a food image')

parser.add_argument('model', type=str, help='the file path of the model to use')

parser.add_argument('input', type=str, help='the image path to pass to the model')

parser.add_argument('-top', type=int, default=5, help='print the top-n predictions')

args = parser.parse_args()

with open(os.path.join('dataset-ethz101food', 'meta', 'classes.txt')) as file:
    class_labels = [line.strip('\n') for line in file.readlines()]

if os.path.exists(args.model) and os.path.exists(args.input) and 0 < args.top < 102:
    model = load_model(args.model)
    img = image.load_img(args.input, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x).flatten()
    top_classes_idx = np.argsort(preds).flatten()
    result = 'Image ' + str(args.input) + ' results:\n'
    for i in range(args.top):
        idx = top_classes_idx[100-i]
        result += '\t predictions {:d}/{:d}  -->  classified as: {:s}({:d}) with a confidence of {:f}\n'.format(i+1, args.top, class_labels[idx], idx, preds[idx])
    print(result)
else:
    print("one of more parameters path are not correct")
