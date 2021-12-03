import numpy as np
from scipy import signal, stats, cluster
import matplotlib.pyplot as plt


def plot_tsn_01(EXP_TYPE, tsne, y_true, id_class=None):
    all_colors = ['#ffa600', '#a05195', '#ebda96', '#e59359', '#d43d51']

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    label_per_class = EXP_TYPE.split('_')
    code_per_class =  np.arange(len(label_per_class)) # [0, 1]
    color_per_class = all_colors[:len(label_per_class)]

    if id_class is not None and id_class > np.max(code_per_class):
        id_class = None

    if id_class is None:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # for every class, we'll add a scatter plot separately
        for code in code_per_class:

                # find the samples of the current class in the data
                indices = [i for i, l in enumerate(y_true) if l == code]

                # extract the coordinates of the points of this class only
                current_tx = np.take(tx, indices)
                current_ty = np.take(ty, indices)

                # convert the class color to matplotlib format
                color = color_per_class[code]
         	    # add a scatter plot with the corresponding color and label
                ax.scatter(current_tx, current_ty, c=color, label=label_per_class[code])

         	# build a legend using the labels we set previously
        ax.legend(loc='best')
        plt.show()
    else:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for code in [id_class]:

            indices = [i for i, l in enumerate(y_true) if l == code]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format
            color = color_per_class[code]
     	    # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, c=color, label=label_per_class[code])

     	# build a legend using the labels we set previously
        ax.legend(loc='best')
        plt.show()

def scale_to_01_range(x):
    """
    Scale values between zero and one
    """
    x_max = np.max(x)
    x_min = np.min(x)
    return (x - x_min) / (x_max - x_min)

def plot_metrics(avg_cost, msg= None):

    if msg is None:
        msg = ''
    train_loss = avg_cost[:,0]
    valid_loss = avg_cost[:,3]

    train_acc = avg_cost[:,1]
    valid_acc = avg_cost[:,4]

    plt.plot(train_loss,'r-' , label='training loss')
    plt.plot(valid_loss,'b--', label='validation loss')
    plt.legend()
    plt.xlabel(r'epochs')
    plt.title( msg + ' Loss')
    plt.show()

    plt.plot(train_acc, 'r-', label='training')
    plt.plot(valid_acc, 'b--', label='validation')
    plt.legend()
    plt.xlabel(r'epochs')
    plt.title(msg + ' Accuracy')
    plt.show()


def plot_wave(data, node=0, start=30000, length=1024, fs=128.):
    y = data[start : start + length, node]
    x = np.arange(1, y.size + 1) / fs
    fig, ax = plt.subplots(1, 1, figsize=(40, 2))
    ax.plot(x, y)
    ax.set_title('original')
    ax.set_xlabel('secs')
    ax.set_ylabel('uV')
    plt.show()

def plot_correlation(data, features, title='', size=(6, 6)):
    corr = np.abs(stats.spearmanr(data).correlation)
    fig, ax = plt.subplots(1, 1, figsize=size)
    feats_idx = np.arange(0, len(features))
    ax.imshow(corr, cmap='jet')
    ax.set_xticks(feats_idx)
    ax.set_yticks(feats_idx)
    ax.set_xticklabels(features, rotation='vertical', fontsize=16)
    ax.set_yticklabels(features, fontsize=16)
    ax.set_title(title, fontsize=20)
    fig.tight_layout()
    plt.show()

def plot_dendrogram_correlation1(data, features, title='', size=(8,12)):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=size)
    corr = stats.spearmanr(data).correlation
    corr_linkage = cluster.hierarchy.ward(corr)
    dendro = cluster.hierarchy.dendrogram(corr_linkage, labels=features, ax=ax2,
                                  leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax1.imshow(np.abs(corr[dendro['leaves'], :][:, dendro['leaves']]), cmap='jet')

    ax1.set_xticks(dendro_idx)
    ax1.set_yticks(dendro_idx)
    ax1.set_xticklabels(dendro['ivl'], rotation='vertical', fontsize=16)
    ax1.set_yticklabels(dendro['ivl'], fontsize=16)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    plt.show()

def plot_dendrogram_correlation2(data, features, title='', size=(10, 8)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
    corr = stats.spearmanr(data).correlation
    corr_linkage = cluster.hierarchy.ward(corr)
    dendro = cluster.hierarchy.dendrogram(corr_linkage, labels=features, ax=ax1,
                                  leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(np.abs(corr[dendro['leaves'], :][:, dendro['leaves']]), cmap='jet')
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical', fontsize=16)
    ax2.set_yticklabels(dendro['ivl'], fontsize=16)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    plt.show()

def eeg_min_max_values(eeg_df, display=False):
    max_y = max_y = eeg_df.max(0)[:14].values
    max_y = np.floor(max_y).astype('int').tolist()
    min_y = eeg_df.min(0)[:14].values
    min_y = np.floor(min_y).astype('int').tolist()

    if display:
        print(max_y)
        print(min_y)
        col1 = max_y[:7]
        col2 = max_y[7:]
        col2.reverse()
        for i in range(7):
            print(col1[i], '<->', col2[i])

    return min_y, max_y

def plot_wave_by_nodes(sample, color, nodes, msg, SF):
    if len(sample.shape) !=2:
        print('shape of the sample does not match!')
    else:
        fig, ax = plt.subplots(1, 14, figsize=(14,2))
        for idx, node in enumerate(nodes):
            data = sample[idx, :]
            times = np.arange(1, data.size + 1) / SF
            ax[idx].plot(times, data, lw=1.5, color= color)
            ax[idx].axis('off')
            ax[idx].set_title(node, fontsize=20)
        if msg != '':
            plt.suptitle(msg, fontsize=20)
        plt.tight_layout()
        plt.show()


def summarize_result(scores, param=None, title=None, xtitle=None, ytitle=None):
    if param is None:
        param = 'Accuracy'
    print(scores)
    fig, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    plt.boxplot(scores)
    plt.xticks(rotation=90)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.show()
    m, fn = np.mean(scores), np.std(scores)
    print(f'{param}: {m:.2f} (+/- {fn:.2})')


def summarize_results(all_scores, params=None, title=None, xtitle=None, ytitle=None):
    if params is None:
        params = np.arange(len(all_scores)) + 1
    fig, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    for i in range(len(all_scores)):
        m, fn = np.mean(all_scores[i]), np.std(all_scores[i])
        print(f'{params[i]}: {m:.2f} (+/- {fn:.2})')
    plt.boxplot(all_scores) #, labels=params)
    plt.xticks(rotation=90)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.show()

def plot_iqr(iqr, dataset, phase=[1], wave='theta'):
    """
      Plot IQRs
      
    """
    
    import cvc_eeglib.eeg_globals as gbl
    color = ['red', 'green', 'blue']
    id_wave = gbl.dic_wave_id[wave]
    labels = gbl.all_pow_nodes
    ticks = np.arange(0,14)
    pltlab = []
    for label in labels[id_wave * 14 : id_wave * 14 + 14]:    
        pltlab.append(label)
        # pltlab.append(label.split('.')[1])


    plt.title(dataset + ': ' + wave + ' IQR')
    for i in phase:        
        plt.plot(iqr[dataset, i][id_wave * 14 : id_wave * 14 + 14], 
                 marker='o', ls='--', lw=1, c=color[i-1], label= 'phase ' + str(i))    
    plt.xticks(ticks, pltlab, rotation=90)
    plt.legend(loc='upper right')
    plt.show()

