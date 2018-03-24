import os


def create_empty_directories(dirlist, exist_ok=True, empty_dirs=False):
    for dir in dirlist:
        os.makedirs(dir, exist_ok=exist_ok)
        if len(os.listdir(dir)) > 0 and empty_dirs:
            raise FileExistsError('Output directory ' + str(dir) + ' is not empty')