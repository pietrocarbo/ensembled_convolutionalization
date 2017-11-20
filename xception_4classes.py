import os
from PIL import Image
import tensorflow as tf
import keras
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard

def disable_randomization_effects():
    from keras import backend as K
    import random as rn
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.allow_growth = True
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


disable_randomization_effects()

class_names = ['cats', 'dogs', 'horses', 'humans']
num_classes = len(class_names)

IMG_SIZE = 299

X = []
y = []


def one_hot_encoding(class_dir_name):
    if class_dir_name == class_names[0]:
        return [1, 0, 0, 0]
    elif class_dir_name == class_names[1]:
        return [0, 1, 0, 0]
    elif class_dir_name == class_names[2]:
        return [0, 0, 1, 0]
    elif class_dir_name == class_names[3]:
        return [0, 0, 0, 1]


for class_dir in os.listdir('dataset-animals4classes'):
    print('importing from ' + class_dir)
    for img_file in tqdm(os.listdir(os.path.join('dataset-animals4classes', class_dir))):
      img = Image.open(os.path.join('dataset-animals4classes', class_dir, img_file))
      X.append(np.array((img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS))))
      y.append(np.array(one_hot_encoding(class_dir)))
      img.close()

# import matplotlib.pyplot as plt
# plt.imshow(x[800])
# plt.show()

X, X_test, y, y_test = train_test_split(X, y, test_size=0.05)
X = np.array(X)
y = np.array(y)
X_test = np.array(X_test)
y_test = np.array(y_test)

base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', classes=num_classes)

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)
# print(base_model.summary())

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

early_stopping = EarlyStopping(monitor='acc', patience=3, verbose=1)
tensor_logs = TensorBoard(log_dir='./tensorboard_logs', histogram_freq=0, batch_size=32, write_graph=True,
                          write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                          embeddings_metadata=None)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=15, verbose=1, validation_split=0.2, callbacks=[early_stopping, tensor_logs])

for layer in base_model.layers[:116]:
    layer.trainable = False
for layer in base_model.layers[116:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100000, verbose=1, validation_split=0.2, callbacks=[early_stopping, tensor_logs])

hist = model.evaluate(X_test, y_test)
print(hist)


