import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')

def save_acc_loss_plots(histories, acc_fn, loss_fn):
    epochs_per_step = []
    val_acc = []
    val_loss = []
    train_loss = []
    train_acc = []
    training_steps = len(histories)
    for i in range(training_steps):
        epochs_per_step.append(len(histories[i].history['val_acc']))
        for j in range(len(histories[i].history['val_acc'])):
            train_acc.append(histories[i].history['acc'][j])
            train_loss.append(histories[i].history['loss'][j])
            val_acc.append(histories[i].history['val_acc'][j])
            val_loss.append(histories[i].history['val_loss'][j])

    x = range(len(train_acc))
    plt.plot(x, train_acc)
    plt.plot(x, val_acc)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train_acc', 'val_acc'], loc=0)
    for i in range(len(epochs_per_step)):
        plt.annotate(str(epochs_per_step[i]), xy=(epochs_per_step[i+1], 0.2), xytext=(epochs_per_step[i], 0.2), xycoords='data',
                verticalalignment='center', arrowprops=dict(color='red', arrowstyle="->", ls='--'))
    plt.savefig(acc_fn)

    plt.clf()

    x = range(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, val_loss)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train_loss', 'val_loss'], loc=0)
    for i in range(len(epochs_per_step)):
        plt.annotate(str(epochs_per_step[i]), xy=(epochs_per_step[i+1], 0.2), xytext=(epochs_per_step[i], 0.2), xycoords='data',
                verticalalignment='center', arrowprops=dict(color='red', arrowstyle="->", ls='--'))
    plt.savefig(loss_fn)
    plt.close()
