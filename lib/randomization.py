import tensorflow as tf
import numpy as np
import os


def lower_randomization_effects(cpu_parallelism=True):
    import random as rn
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    tf.set_random_seed(1234)
