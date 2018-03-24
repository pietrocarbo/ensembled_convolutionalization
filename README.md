# Automatic point-of-interest image cropping via ensembled convolutionalization

Convolutionalization of discriminative neural networks, introduced [here](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) for segmentation purposes, is a simple technique allowing to generate _heat-maps_ relative to the location of a given object in a larger image. 
In this article, we apply this technique to automatically crop images at their actual point of interest, fine tuning them with the final aim to improve the quality of a dataset.
The use of an _ensemble_ of fully convolutional nets sensibly reduce the risk of overfitting, resulting in reasonably accurate croppings. The methodology has been tested on a well known dataset, particularly renowned for containing badly centered and noisy images: the [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) dataset, composed of 101K images spread over 101 food categories. The quality of croppings can be testified
by a sensible and uniform improvement (3-5%) in the classification accuracy of classifiers, even external to the ensemble.

## Setup instructions

1. Clone this repository

2. Download the Food-101 dataset [here](https://www.vision.ee.ethz.ch/datasets_extra/food-101/), uncompress it and place it in `ensembled_convolutionalization/dataset-ethz101food`

3. Create a Python 3.x virtual environment and install the dependencies using the commands:
  * (optional, recommended) `python3 -m pip install --upgrade pip`
  * `pip install virtualenv`
  * `virtualenv -p python3 venv`
  * `source venv/bin/activate` on Linux or `venv\Scripts\activate.bat` on Windows
  * `pip install -r requirements.txt`

4. Run the script `copy_splitdataset.py` to copy the dataset images in the train/test folders (then delete the `images` directory if you want to save disk space)