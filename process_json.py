import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.preprocessing import image


def show_samples(samples_ix, dumps):
    for ix in samples_ix:
        print("True label:", dumps[ix]["label"])
        print("Original size label:", dumps[ix]["originalSize"]["xception"]["labelGuessed"],
              ", confidence", dumps[ix]["originalSize"]["xception"]["scoreGuessed"])
        print("Cropped size label:", dumps[ix]["croppedSize"]["xception"]["labelGuessed"],
              ", confidence", dumps[ix]["croppedSize"]["xception"]["scoreGuessed"], "\n")

        fig, ax = plt.subplots(1)
        img = image.load_img(dumps[ix]["filename"])
        img = image.img_to_array(img)
        ax.imshow(img / 255.)
        rect = patches.Rectangle((dumps[ix]["square_crop"]["lower_left"][1], dumps[ix]["square_crop"]["lower_left"][0]),
                                 dumps[ix]["square_crop"]["side"], dumps[ix]["square_crop"]["side"], linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()


def process(filename):
    with open(filename, "r") as json_file:
        dumps = json.load(json_file)
        vgg16_origAcc = 0.
        vgg16_cropAcc = 0.
        xce_origAcc = 0.
        xce_cropAcc = 0.
        missed_samples = []
        corrected_samples = []
        for ix, sample in enumerate(dumps):
            true_label = sample["label"]
            if true_label == sample["originalSize"]["vgg16"]["labelGuessed"]:
                vgg16_origAcc += 1
            if true_label == sample["originalSize"]["xception"]["labelGuessed"]:
                xce_origAcc += 1
            if true_label == sample["croppedSize"]["vgg16"]["labelGuessed"]:
                vgg16_cropAcc += 1
            if true_label == sample["croppedSize"]["xception"]["labelGuessed"]:
                xce_cropAcc += 1

            if true_label == sample["originalSize"]["xception"]["labelGuessed"] and true_label != sample["croppedSize"]["xception"]["labelGuessed"]:
               missed_samples.append(ix)
            if true_label != sample["originalSize"]["xception"]["labelGuessed"] and true_label == sample["croppedSize"]["xception"]["labelGuessed"]:
               corrected_samples.append(ix)


        n_samples = len(dumps)
        print("Total samples", n_samples)
        print("Accuracy VGG {:f}, {:f} Accuracy XCE {:f}, {:f}".format(vgg16_origAcc/n_samples, vgg16_cropAcc/n_samples, xce_origAcc/n_samples, xce_cropAcc/n_samples))
        print("XCE missed",  len(missed_samples), "samples:", missed_samples)
        print("XCE corrected", len(corrected_samples), "samples:", corrected_samples)

        # show_samples(missed_samples, dumps)
        show_samples(corrected_samples, dumps)

process("imagesSet101000cmaxMaxInputIX_Ensemble.json")