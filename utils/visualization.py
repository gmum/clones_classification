import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


COLOR_MAP = sns.light_palette('dark pink', input='xkcd')


def norm_cm(arr):
    sums = arr.sum(axis=1)
    arr = arr / sums[:, np.newaxis]
    arr = arr.round(2)
    return arr


def show_cm(arrs, n, label_cm, filename):
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(n * 3 + 3, 3), squeeze=False)
    for i in range(n):
        arr = arrs[i]
        sums = arr.sum(axis=1)
        arr = arr / sums[:, np.newaxis]
        arr = arr.round(2)
        curr_ax = ax[0][i]
        labels_plot = label_cm
        sns.heatmap(arr,
                    xticklabels=labels_plot,
                    yticklabels=labels_plot,
                    cmap=COLOR_MAP,
                    square=True,
                    cbar=False,
                    annot=True,
                    ax=curr_ax)
        curr_ax.set_ylabel('True')
        curr_ax.set_xlabel('Predicted')
        curr_ax.set_title(f'Accuracy: {(sum(arr.diagonal()) / sum(sum(arr))).round(2)}')
        # bottom, top = curr_ax.get_ylim()
        # curr_ax.set_ylim(bottom + 0.5, top - 0.5)
    if filename is not None:
        plt.savefig(filename)
    return fig


def show_avg_cm(arrs, n, label_cm, filename=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), squeeze=False)
    arr = np.zeros(arrs[0].shape)
    for i in range(n):
        arr += arrs[i]
    sums = arr.sum(axis=1)
    arr = arr / sums[:, np.newaxis]
    arr = arr.round(2)
    labels_plot = label_cm
    sns.heatmap(arr,
                xticklabels=labels_plot,
                yticklabels=labels_plot,
                cmap=COLOR_MAP,
                square=True,
                cbar=False,
                annot=True)
    ax[0][0].set_ylabel('True')
    ax[0][0].set_xlabel('Predicted')
    ax[0][0].set_title(f'Accuracy: {(sum(arr.diagonal()) / sum(sum(arr))).round(2)}')
    # bottom, top = ax[0][0].get_ylim()
    # ax[0][0].set_ylim(bottom + 0.5, top - 0.5)
    if filename is not None:
        plt.savefig(filename)
    return fig