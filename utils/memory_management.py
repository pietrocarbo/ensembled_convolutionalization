import tensorflow as tf
from keras import backend as K


def memory_growth_config(cpu_parallelism=True, allow_growth=True, memory_fraction=None):
    K.clear_session()
    if not cpu_parallelism:
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    else:
        session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = allow_growth
    if memory_fraction:
        session_conf.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
