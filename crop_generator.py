import PIL.Image
import pickle
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input as preprocess

with open("dataset-ethz101food/meta/classes.txt") as file:
    map_label_ix = {label.strip('\n'): ix for (ix, label) in enumerate(file.readlines())}

def is_square_in_img(llh, llw, edge, imgh, imgw):
    def inside(width, height, x, y):
        if 0 <= x <= width and 0 <= y <= height: return True
        else: return False
    if inside(imgw, imgh, llw, llh) and inside(imgw, imgh, llw+edge, llh) and inside(imgw, imgh, llw, llh+edge) and inside(imgw, imgh, llw+edge, llh+edge):
        return True
    else:
        return False

def yield_crops(cropfilename):

    count = 0
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

                img = image.load_img(crop["filename"])
                img = image.img_to_array(img)
                imgh, imgw = img.shape[0:2]

                img = img[coordh:coordh + rect_dim, coordw:coordw + rect_dim]

                # if not is_square_in_img(coordh, coordw, rect_dim, imgh, imgw):
                #     print("Crop out of img bound, file:", crop["filename"], imgh, imgw, "Crop data:", crop)
                #     fig, ax = plt.subplots(1)
                #     ax.imshow(img / 255.)
                #     ax.set_title(crop["filename"])
                #     plt.show()

                img = image.array_to_img(img)
                img = img.resize((224, 224), PIL.Image.BICUBIC)
                img = image.img_to_array(img)

                img = np.expand_dims(img, axis=0)
                img = preprocess(img)

                y = map_label_ix[str(crop["label"])]

                print("File", count, "label", y)

                count += 1
                yield ({'input_1': img}, {'output_layer': np.expand_dims(to_categorical(y, num_classes=101), axis=0)})

yield_crops("results/cropping_eval/cropsdata.pickle")