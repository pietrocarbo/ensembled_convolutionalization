import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input as preprocess

with open("dataset-ethz101food/meta/classes.txt") as file:
    map_label_ix = {label.strip('\n'): ix for (ix, label) in enumerate(file.readlines())}

def yield_crops(cropfilename):

    while True:
        with open(cropfilename, "rb") as cropfile:

            crops = pickle.load(cropfile)
            print("Crops in pickle file:", len(crops))

            for crop in crops:
                # create numpy arrays of input data
                # and labels, from each line in the file

                coordh = int(crop["rect"]["lower_left"][0])
                coordw = int(crop["rect"]["lower_left"][1])
                rect_dim = int(crop["rect"]["side"])

                img = image.load_img(crop["filename"], target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess(img[coordh:coordh + rect_dim, coordw:coordw + rect_dim])

                y = map_label_ix[str(crop["label"])]

                # print("Input", img, "output", y)
                #
                # fig, ax = plt.subplots(1)
                # ax.imshow(img / 255.)
                # ax.set_title(crop["filename"])
                # plt.show()

                yield ({'input_1': img}, {'output': y})

yield_crops("results/cropping_eval/cropsdata.pickle")