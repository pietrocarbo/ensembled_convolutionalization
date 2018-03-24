import os

dataset_path = "dataset-ethz101food"

# Helper function to construct labels array
def ix_to_class_name(idx):
    with open(os.path.join(dataset_path, "meta", "classes.txt")) as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
    return class_labels[idx]

# Helper function to get the label index given its name
def class_name_to_idx(name):
    with open(os.path.join(dataset_path, "meta", "classes.txt")) as file:
        class_labels = [line.strip('\n') for line in file.readlines()]
        for i, label_name in enumerate(class_labels):
            if label_name == name:
                return i
        else:
            print("class idx not found!")
            exit(-1)
