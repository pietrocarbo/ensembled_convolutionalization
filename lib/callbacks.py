import os
import keras

def checkpointer(filename, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, period=1):
    return keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), 'models', filename), monitor='val_acc',
                                           verbose=verbose, save_best_only=save_best_only, save_weights_only=save_weights_only, mode='auto', period=period)

def early_stopper(monitor='val_acc', min_delta=0, patience=50):
    return keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode='auto')


def lr_reducer(monitor='val_loss', factor=0.1, patience=5, epsilon=0.0001, cooldown=0, min_lr=0):
    return keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, verbose=1, mode='auto', epsilon=epsilon, cooldown=cooldown, min_lr=min_lr)


def csv_logger(filename, separator='\t', append=True):
    return keras.callbacks.CSVLogger(os.path.join(os.getcwd(), 'logs', filename), separator=separator, append=append)
