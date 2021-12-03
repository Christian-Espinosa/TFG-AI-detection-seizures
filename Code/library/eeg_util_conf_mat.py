import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

def confusion_matrix_plot_skplt(y_true, y_pred, title=None):
    import scikitplot as skplt
    if title is not None:
        skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True, cmap='Blues')
    else:
        skplt.metrics.plot_confusion_matrix(y_true, y_pred, title= title, normalize=True, cmap='Blues')

def confusion_matrix_calculate(y_true, y_pred, tags_categ=None):

    if tags_categ is None:
        tags_categ = list(set(y_true))

    tags_ord = np.arange(len(tags_categ))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=tags_ord, normalize='true')
    return cm

def confusion_matrix_plot(y_true, y_pred, title=None, tags_categ=None, colormap=None):

    display_bar = True
    if title is None:
        title = 'Confusion Matrix'
    if tags_categ is None:
        tags_categ = list(set(y_true))
    if colormap is None:
        colormap = 'gray'
        display_bar = False

    tags_ord = np.arange(len(tags_categ))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=tags_ord, normalize='true')

    thresh = (cm.max() +  cm.min()) / 2.0
    mat = np.ones(cm.shape, dtype='float') # * 0.9 # modulate the color background

    fig, ax = plt.subplots(1,1)
    ax.matshow(mat, cmap=colormap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    if display_bar:
        cax = ax.matshow(cm, cmap=colormap)
        fig.colorbar(cax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if display_bar:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment='center',
                     color='white' if cm[i, j] < thresh else 'black',
                     fontsize=14
                     )
            else:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment='center',
                     fontsize=14
                     )
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(tags_ord)
    ax.set_xticklabels(tags_categ)
    ax.set_yticks(tags_ord)
    ax.set_yticklabels(tags_categ, rotation='vertical')
    fig.tight_layout()
    plt.show()


def matrix_plot_values(mat, title=None, tags_x=None, tags_y=None, colormap=None):

    display_bar = True
    mat_ones =np.ones(mat.shape, dtype='float')

    if title is None:
        title = 'Matrix'

    if colormap is None:
        colormap = 'gray'
        display_bar = False

    range_x = np.arange(mat.shape[1]) if tags_x is None else np.arange(len(tags_x))
    range_y = np.arange(mat.shape[0]) if tags_y is None else np.arange(len(tags_y))
    thresh = mat.max() / 2.0

    fig, ax = plt.subplots(1,1)
    ax.matshow(mat_ones, cmap=colormap, vmin=0, vmax=1)
    ax.set_title(title, pad=20)
    ax.set_xticks(range_x)
    ax.set_yticks(range_y)
    if display_bar:
        cax = ax.matshow(mat, cmap=colormap)
        fig.colorbar(cax)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if display_bar:
                plt.text(j, i, "{:0.2f}".format(mat[i, j]),
                     horizontalalignment='center',
                     color='white' if mat[i, j] < thresh else 'black',
                     fontsize=14 )
            else:
                plt.text(j, i, "{:0.2f}".format(mat[i, j]),
                     horizontalalignment='center',
                     fontsize=14
                     )

    ax.xaxis.set_ticks_position('bottom')
    if tags_x is not None:
        ax.set_xticklabels(tags_x)
    if tags_y is not None:
        ax.set_yticklabels(tags_y, rotation='vertical')
    fig.tight_layout()
    plt.show()

