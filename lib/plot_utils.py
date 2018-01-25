import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
import itertools
import numpy as np

def save_acc_loss_plots(histories, acc_fn, loss_fn):
    epochs_per_step = []
    val_acc = []
    val_loss = []
    train_loss = []
    train_acc = []
    training_steps = len(histories)
    for i in range(training_steps):
        epochs_per_step.append(len(histories[i].history['val_categorical_accuracy']))
        for j in range(len(histories[i].history['val_categorical_accuracy'])):
            train_acc.append(histories[i].history['categorical_accuracy'][j])
            train_loss.append(histories[i].history['loss'][j])
            val_acc.append(histories[i].history['val_categorical_accuracy'][j])
            val_loss.append(histories[i].history['val_loss'][j])

    x = range(len(train_acc))
    plt.plot(x, train_acc)
    plt.plot(x, val_acc)
    plt.ylim(ymin=0)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train_acc', 'val_acc'], loc=0)
    # epochs_per_step = np.cumsum(epochs_per_step)
    # for i in range(len(epochs_per_step) - 1):
    #     plt.annotate(str(epochs_per_step[i]), xy=(epochs_per_step[i+1], 0.2), xytext=(epochs_per_step[i], 0.2), xycoords='data',
    #             verticalalignment='center', arrowprops=dict(color='red', arrowstyle="->", ls='--'))
    plt.savefig(acc_fn)

    plt.clf()

    x = range(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, val_loss)
    plt.ylim(ymin=0)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train_loss', 'val_loss'], loc=0)
    # for i in range(len(epochs_per_step) - 1):
    #     plt.annotate(str(epochs_per_step[i]), xy=(epochs_per_step[i+1], 0.2), xytext=(epochs_per_step[i], 0.2), xycoords='data',
    #             verticalalignment='center', arrowprops=dict(color='red', arrowstyle="->", ls='--'))
    plt.savefig(loss_fn)
    plt.close()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
