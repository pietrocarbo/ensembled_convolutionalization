import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report

def show_samples(samples_ix, dumps):
    for ix in samples_ix:
        print("True label:", dumps[ix]["label"])
        print("Original size label:", dumps[ix]["originalSize"]["xception"]["labelGuessed"],
              ", confidence", dumps[ix]["originalSize"]["xception"]["scoreGuessed"])
        print("Cropped size label:", dumps[ix]["croppedSize"]["xception"]["labelGuessed"],
              ", confidence", dumps[ix]["croppedSize"]["xception"]["scoreGuessed"], "\n")

        fig, ax = plt.subplots(1)
        from keras.preprocessing import image
        img = image.load_img(dumps[ix]["filename"])
        img = image.img_to_array(img)
        ax.imshow(img / 255.)
        rect = patches.Rectangle((dumps[ix]["square_crop"]["lower_left"][1], dumps[ix]["square_crop"]["lower_left"][0]),
                                 dumps[ix]["square_crop"]["side"], dumps[ix]["square_crop"]["side"], linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

def process_v1(filename):
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

def process_ensdata(filename):
    with open(filename, "r") as json_file:
        dumps = json.load(json_file)

        n_samples = len(dumps)

        factors = np.empty(n_samples)
        scores = np.empty(n_samples)
        nfcns = np.empty(n_samples, dtype=int)
        cropixs = np.empty((n_samples, 2), dtype=int)
        heatshapes = np.empty((n_samples, 2), dtype=int)
        rectdims = np.empty(n_samples, dtype=int)
        rectlls = np.empty((n_samples, 2), dtype=int)


        with open("dataset-ethz101food/meta/classes.txt") as file:
            dict_labels = {label.strip('\n'): ix for (ix, label) in enumerate(file.readlines())}

        y_true = []
        y_original = []
        y_cropped = []

        origTrueLabelScores = np.empty(n_samples)
        cropTrueLabelScores = np.empty(n_samples)

        missed_samples = []
        corrected_samples = []

        for ix, sample in enumerate(dumps):

            true_label = sample["label"]
            y_true.append(dict_labels[true_label])
            y_original.append(dict_labels[sample["clf"]["original"]["labelGuessed"]])
            y_cropped.append(dict_labels[sample["clf"]["crop"]["labelGuessed"]])

            origTrueLabelScores[ix] = sample["clf"]["original"]["scoreTrueLabel"]
            cropTrueLabelScores[ix] = sample["clf"]["crop"]["scoreTrueLabel"]

            factors[ix] = sample["ensemble"]["factor"]
            scores[ix] = sample["ensemble"]["score"]
            nfcns[ix] = sample["ensemble"]["nfcn"]
            heatshapes[ix][0] = sample["ensemble"]["heath"]
            heatshapes[ix][1] = sample["ensemble"]["heatw"]
            cropixs[ix][0] = sample["ensemble"]["cropixh"]
            cropixs[ix][1] = sample["ensemble"]["cropixw"]
            rectdims[ix] = sample["rect"]["side"]
            rectlls[ix][0] = sample["rect"]["lower_left"][0]
            rectlls[ix][1] = sample["rect"]["lower_left"][1]

            if true_label == sample["clf"]["original"]["labelGuessed"] and true_label != sample["clf"]["crop"]["labelGuessed"]:
               missed_samples.append(ix)
            if true_label != sample["clf"]["original"]["labelGuessed"] and true_label == sample["clf"]["crop"]["labelGuessed"]:
               corrected_samples.append(ix)

        n_samples = len(dumps)
        print("File:", filename, "contain", n_samples, "samples")

        print("Cropping metrics (avg+-std): factor", "{:.4f}+-{:.4f},".format(np.mean(factors), np.std(factors)),
                                "score", "{:.4f}+-{:.4f},".format(np.mean(scores), np.std(scores)),
                                "nfcn", "{:.4f}+-{:.4f},".format(np.mean(nfcns), np.std(nfcns)),
                        "index", "({:.4f}+-{:.4f}, {:.4f}+-{:.4f})".format(np.mean(cropixs, axis=0)[0], np.std(cropixs, axis=0)[0],
                                                                      np.mean(cropixs, axis=0)[1], np.std(cropixs, axis=0)[1]))

        print("Label scores: original", "(avg: {:.4f}, std: {:.4f});".format(np.mean(origTrueLabelScores), np.std(origTrueLabelScores)),
                           "cropped", "(avg: {:.4f}, std: {:.4f})".format(np.mean(cropTrueLabelScores), np.std(cropTrueLabelScores)),)

        print("Mistaken", len(missed_samples), "samples:", missed_samples)
        print("Corrected", len(corrected_samples), "samples:", corrected_samples)

        print("\nImage classification report\n", classification_report(y_true, y_original, target_names=[lab for lab in sorted(dict_labels)]))
        print("\nCrop classification report\n", classification_report(y_true, y_cropped, target_names=[lab for lab in sorted(dict_labels)]))
# process_ensdata("testSet25250_ENSEMBLE.json")

with open("dataset-ethz101food/meta/classes.txt") as file:
    dict_labels = {label.strip('\n'): ix for (ix, label) in enumerate(file.readlines())}

def classification_eval(orig_clf_fn, crop_clf_fn):
    orig_clf = pickle.load(open(orig_clf_fn, "rb"))
    crop_clf = pickle.load(open(crop_clf_fn, "rb"))

    y_true = []
    y_original = []
    y_cropped = []

    origTrueLabelScores = np.empty(len(orig_clf))
    cropTrueLabelScores = np.empty(len(crop_clf))

    missed_samples = []
    corrected_samples = []

    for ix, clf in enumerate(list(zip(orig_clf, crop_clf))):

        true_label = clf[0]["ix_label"]
        y_true.append(dict_labels[true_label])

        y_original.append(dict_labels[clf[0]["ix_predicted"]])
        y_cropped.append(dict_labels[clf[1]["ix_predicted"]])

        origTrueLabelScores[ix] = clf[0]["pr_label"]
        cropTrueLabelScores[ix] = clf[1]["pr_label"]

        if true_label == clf[0]["ix_predicted"] and true_label != clf[1]["ix_predicted"]:
            missed_samples.append(ix)
        if true_label != clf[0]["ix_predicted"] and true_label == clf[1]["ix_predicted"]:
            corrected_samples.append(ix)

    print("True label score: on original imgs",
          "(avg: {:.4f}, std: {:.4f}),".format(np.mean(origTrueLabelScores), np.std(origTrueLabelScores)),
          "on cropped imgs",
          "(avg: {:.4f}, std: {:.4f})".format(np.mean(cropTrueLabelScores), np.std(cropTrueLabelScores)), )

    print("Mistaken", len(missed_samples), "samples:", missed_samples)
    print("Corrected", len(corrected_samples), "samples:", corrected_samples)

    print("\nImage classification accuracy\n", accuracy_score(y_true, y_original))
    print("\nCrop classification accuracy\n", accuracy_score(y_true, y_cropped))
    # print("\nImage classification report\n",
    #       classification_report(y_true, y_original, target_names=[lab for lab in sorted(dict_labels)]))
    # print("\nCrop classification report\n",
    #       classification_report(y_true, y_cropped, target_names=[lab for lab in sorted(dict_labels)]))


def final_evaluation(foldername):

    # crops metrics
    ensfn = os.path.join(foldername, "cropsdata.pickle")
    with open(ensfn, "rb") as ensfile:
        ens_crop_data = pickle.load(ensfile)

        n_samples = len(ens_crop_data)
        factors = np.empty(n_samples)
        scores = np.empty(n_samples)
        nfcns = np.empty(n_samples, dtype=int)
        cropixs = np.empty((n_samples, 2), dtype=int)
        heatshapes = np.empty((n_samples, 2), dtype=int)
        rectdims = np.empty(n_samples, dtype=int)
        rectlls = np.empty((n_samples, 2), dtype=int)

        for ix, sample in enumerate(ens_crop_data):
            factors[ix] = sample["crop"]["factor"]
            scores[ix] = sample["crop"]["score"]
            nfcns[ix] = sample["crop"]["nfcn"]
            heatshapes[ix][0] = sample["crop"]["heath"]
            heatshapes[ix][1] = sample["crop"]["heatw"]
            cropixs[ix][0] = sample["crop"]["cropixh"]
            cropixs[ix][1] = sample["crop"]["cropixw"]
            rectdims[ix] = sample["rect"]["side"]
            rectlls[ix][0] = sample["rect"]["lower_left"][0]
            rectlls[ix][1] = sample["rect"]["lower_left"][1]

        print("File:", ensfn, "contains", len(ens_crop_data), "samples")
        print("Cropping metrics (avg+-std): factor", "{:.4f}+-{:.4f},".format(np.mean(factors), np.std(factors)),
                                "score", "{:.4f}+-{:.4f},".format(np.mean(scores), np.std(scores)),
                                "nfcn", "{:.4f}+-{:.4f},".format(np.mean(nfcns), np.std(nfcns)),
                        "index", "({:.4f}+-{:.4f}, {:.4f}+-{:.4f})".format(np.mean(cropixs, axis=0)[0], np.std(cropixs, axis=0)[0],
                                                                      np.mean(cropixs, axis=0)[1], np.std(cropixs, axis=0)[1]))




        # ----------------------------------------------
        # crops evaluation using several classifiers

        # classification_eval(os.path.join(foldername, "vgg16_orig_data.pickle"), os.path.join(foldername, "vgg16_crop_data.pickle"))

        classification_eval(os.path.join(foldername, "vgg19_orig_data.pickle"), os.path.join(foldername, "vgg19_crop_data.pickle"))

        # classification_eval(os.path.join(foldername, "xce_orig_data.pickle"), os.path.join(foldername, "xce_crop_data.pickle"))

        # classification_eval(os.path.join(foldername, "incv3_orig_data.pickle"), os.path.join(foldername, "incv3_crop_data.pickle"))

        # classification_eval(os.path.join(foldername, "incrv2_orig_data.pickle"), os.path.join(foldername, "incrv2_crop_data.pickle"))


final_evaluation("results/cropping_eval")