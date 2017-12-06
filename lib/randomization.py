import tensorflow as tf
import numpy as np
import os


def disable_randomization_effects(cpu_parallelism=True):
    from keras import backend as K
    import random as rn
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    tf.set_random_seed(1234)
    # if not cpu_parallelism:
    #     session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # else:
    #     session_conf = tf.ConfigProto()
    # session_conf.gpu_options.allow_growth = True
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)
