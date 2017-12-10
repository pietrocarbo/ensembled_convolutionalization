import tensorflow as tf
from keras import backend as K

cpu_parallelism = True

if not cpu_parallelism:
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
else:
    session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
