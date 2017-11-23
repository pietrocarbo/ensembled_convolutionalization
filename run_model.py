#! C:\Users\Pietro\Desktop\Machine Learning\Progetto\project_machine_learning\env\Scripts python

import os
import argparse
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

parser = argparse.ArgumentParser(description='transfer learning from resnet50 on ethz101 food dataset')

parser.add_argument('model', type=str, help='the file path of the model to use')

parser.add_argument('input', type=str, help='the file path of the input to pass to the model')

args = parser.parse_args()

if os.path.exists(args.model) and os.path.exists(args.input):
    model = load_model(args.model)
    img = image.load_img(args.input, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=5))
else:
    print("model or input path don't exists")