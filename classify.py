import os
import sys
import argparse
import numpy as np
import keras
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image

parser = argparse.ArgumentParser(description='script used to classify a food image')
parser.add_argument('model-architecture_name', type=str, help='name of the model or file containing it\'s architecture')  # TODO distinguish between name or filename
parser.add_argument('model', type=str, help='the file path of the model to use')
parser.add_argument('-w', '--weights', action='store_true', help='flag that indicates if we are loading only weights')
parser.add_argument('input', type=str, help='the path of the image or the directory containing images to classify')
parser.add_argument('-top', type=int, default=5, help='print the top-n predictions')
args = parser.parse_args()

with open(os.path.join('dataset-ethz101food', 'meta', 'classes.txt')) as file:
    class_labels = [line.strip('\n') for line in file.readlines()]


def classify_image(image_file, preprocessing):
    img = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocessing(x)
    preds = model.predict(x).flatten()
    sorted_classes_ix = np.argsort(preds).flatten()
    result = 'Image ' + str(image_file) + ' results:\n'
    for i in range(args.top):
        idx = sorted_classes_ix[-i]
        result += '\t prediction {:d}/{:d}  -->  classified as: {:s}({:d}) with a confidence of {:f}\n'.format(i + 1, args.top, class_labels[idx], idx, preds[idx])
    return result


def assemble_net(base_model):
    last_layer = base_model.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(512, activation='relu', name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc-2')(x)
    x = Dropout(0.5)(x)
    out = Dense(101, activation='softmax', name='output_layer')(x)
    return Model(inputs=base_model.input, outputs=out)


if os.path.exists(args.model) and os.path.exists(args.input) and 0 < args.top < 101:

    if args.model_name == 'resnet50':
        from keras.applications.resnet50 import preprocess_input
        from keras.applications.mobilenet import preprocess_input
        if args.weights:
            base_model = keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', classes=101)
            model = assemble_net(base_model)
            model.load_weights(args.model)
        else:
            model = load_model(args.model)

    elif args.model_name == 'mobilenet':
        from keras.applications.mobilenet import preprocess_input
        if args.weights:
            base_model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', classes=101)
            model = assemble_net(base_model)
            model.load_weights(args.model)
        else:
            model = load_model(args.model)

    elif args.model_name == 'incv3':
        from keras.applications.inception_v3 import preprocess_input
        if args.weights:
            base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=101)
            model = assemble_net(base_model)
            model.load_weights(args.model)
        else:
            model = load_model(args.model)

    elif args.model_name == 'xcept':
        from keras.applications.xception import preprocess_input
        if args.weights:
            base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=101)
            model = assemble_net(base_model)
            model.load_weights(args.model)
        else:
            model = load_model(args.model)

    elif args.model_name == 'vgg19':
        from keras.applications.vgg19 import preprocess_input
        if args.weights:
            base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=101)
            model = assemble_net(base_model)
            model.load_weights(args.model)
        else:
            model = load_model(args.model)

    elif args.model_name == 'incresv2':
        from keras.applications.inception_resnet_v2 import preprocess_input
        if args.weights:
            base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=101)
            model = assemble_net(base_model)
            model.load_weights(args.model)
        else:
            model = load_model(args.model)

    else:
        print("model name unrecognizable")
        sys.exit(-1)

    if os.path.isdir(args.input):
        output = [classify_image(os.path.join(args.input, image), preprocess_input) for image in os.listdir(args.input)]
        print(*output, sep='\n')
    else:
        print(classify_image(args.input, preprocess_input))

else:
    print("one of more parameters are not correct")
    sys.exit(-1)
