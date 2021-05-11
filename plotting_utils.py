from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_dataset(X, y, ax=None, c1='mediumblue', c2='crimson'):
    if ax is None:
        _, ax = plt.subplots()

    __sanity_checks(X, y)

    category1 = y == 0
    category2 = y == 1
    ax.scatter(X[category1, 0], X[category1, 1], c=c1)
    ax.scatter(X[category2, 0], X[category2, 1], c=c2)
    return ax


def make_meshgrid(X, padding=0.05, n_x=20, n_y=20):
    __sanity_checks(X)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    padding_x, padding_y = padding * (x_max - x_min), padding * (y_max - y_min)
    x_min -= padding_x
    x_max += padding_x
    y_min -= padding_y
    y_max += padding_y

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_x), np.linspace(y_min, y_max, n_y))
    return xx, yy


def plot_decision_region(model_function, xx, yy, ax=None, cmap=None, levels=100):
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['lightcoral', 'white', 'cornflowerblue'])
    if ax is None:
        _, ax = plt.subplots()

    coords = np.c_[xx.ravel(), yy.ravel()]
    z = model_function(coords)
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, cmap=cmap, levels=levels)
    return ax

def remove_everything(ax, title=''):
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, size=14)


def __sanity_checks(X, y=None):
    if len(np.shape(X)) != 2:
        raise ValueError(f'Dataset X must be array of shape (n_datapoints, 2), was given {np.shape(X)}.')

    n_datapoints, feature_dim = np.shape(X)

    if feature_dim != 2:
        raise ValueError(f'Dataset X must have feature dimension of 2, was given {feature_dim}')

    if y is not None:
        if len(y) != n_datapoints:
            raise ValueError(
                f'Targets y must be of same length as X. Expected length {n_datapoints}, was given {len(y)}')

        data_classes = np.unique(y)

        if len(data_classes) != 2:
            raise ValueError(f'Currently only binary classification problems are supported!')
