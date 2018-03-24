import os
from shutil import copyfile

# script to use to split the Food-101 images in the train and test sets using the dataset original splitting indication

def split_dataset(splitfile):
    with open(os.path.join('dataset-ethz101food','meta', splitfile + '.txt')) as file:
        lines = file.readlines()
        ensure_dir(os.path.join('dataset-ethz101food', splitfile))
        for i in lines:
            class_folder, image_file = i.split('/')
            image_file = image_file.strip('\n')
            ensure_dir(os.path.join('dataset-ethz101food', splitfile, class_folder))
            copyfile(os.path.join('dataset-ethz101food', 'images', class_folder, image_file + '.jpg'), os.path.join('dataset-ethz101food', splitfile, class_folder, image_file + '.jpg'))
            print(class_folder, image_file)

def ensure_dir(pathdir):
    if not os.path.exists(pathdir):
        print('Creating folder', pathdir)
        os.makedirs(pathdir)

split_dataset('train')
split_dataset('test')