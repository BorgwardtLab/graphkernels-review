#!/usr/bin/env python3
#
# visualise_kernel_accuracies.py: visualises the individual kernels by
# means of their accuracies on the individual data sets. Multiple plot
# variants will be provided by this script (based on different metrics
# or dissimilarity measures).

import argparse

from sklearn.manifold import MDS
from sklearn.manifold import TSNE

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def circles(x, y, s, c='b', vmin=None, vmax=None, ax=None, **kwargs):
    '''
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    if not ax:
        ax = plt.gca()

    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection


def embed_distance_matrix(D):
    '''
    Embeds a distance matrix into 2D space for visualisation and
    subsequent analysis.

    :param D: Distance matrix

    :return: Coordinate matrix. Indexing follows the indexing of
    the original matrix; no reshuffling is done.
    '''

    embedding = MDS(
        metric=True,
        dissimilarity='precomputed',
        random_state=42
    )

    return embedding.fit_transform(D)


def resample_data_frame(df, n_samples):
    '''
    Given a data frame whose cells contain a mean and a standard
    deviation, performs `n_samples` sampling operations, storing
    the resulting matrices in a list.
    '''

    def sample(x):
        '''
        Takes a cell of the data frame, converts it into the
        corresponding representation of mean/sdev, and draws
        a random sample from the corresponding distribution.
        '''

        # Short-circuit sampling procedure if there's nothing to sample
        if type(x) != str and np.isnan(x):
            return 0.0

        mean, _, sdev = str(x).split()

        mean = float(mean)
        sdev = float(sdev)

        return mean

    matrices = []

    for i in range(n_samples):
        df_ = df.iloc[:, 1:].applymap(sample)
        matrices.append(df_.iloc[:, 1:].to_numpy())

    return matrices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    args = parser.parse_args()

    df = pd.read_csv(args.INPUT)

    # Check whether we have to perform *sampling* because the cells of
    # the data frame are mean accuracies with standard deviations.
    if 'object' in df.iloc[:, 1:].dtypes.values:
        # TODO: make configurable
        n_samples = 10
        feature_matrices = resample_data_frame(df, n_samples)
    else:
        feature_matrices = [df.iloc[:, 1:].to_numpy()]

    kernel_names = df.iloc[:, 0].values
    data_set_names = df.iloc[0, :].values   # these are not required...

    metrics = [
        {'metric': 'minkowski', 'p': 1},  # $L_1$
        {'metric': 'minkowski', 'p': 2},  # $L_2$
        {'metric': 'chebyshev'},
        {'metric': 'cityblock'}
    ]

    fig, axes = plt.subplots(
        figsize=(16, 4),
        ncols=len(metrics),
        nrows=1,
        squeeze=True
    )

    for index, kwargs in enumerate(metrics):

        # Will contain all coordinate matrices generated by the current
        # embedding method. This makes it possible to calculate a mean,
        # as well as a standard deviation for *positions*.
        coordinate_matrices = []

        for X in feature_matrices:
            D = squareform(pdist(X, **kwargs))
            Y = embed_distance_matrix(D)

            coordinate_matrices.append(Y)

        Y = np.mean(coordinate_matrices, axis=0)
        s = np.std(coordinate_matrices, axis=0)
        s = np.amax(s, axis=1)

        # Prepare plots (this is just for show; we are actually more
        # interested in the output files).
        title = kwargs['metric']

        if 'p' in kwargs:
            title += '_' + str(kwargs['p'])

        axes[index].set_aspect('equal')
        axes[index].scatter(Y[:, 0], Y[:, 1], c='k')

        circles(
            Y[:, 0],
            Y[:, 1],
            s,
            ax=axes[index]
        )

        axes[index].set_title(title)
        axes[index].tick_params(
            bottom=False, top=False,
            left=False, right=False,
            labelbottom=False, labeltop=False,
            labelleft=False, labelright=False,
        )

        for j, text in enumerate(kernel_names):
            x = Y[j, 0]
            y = Y[j, 1]
            axes[index].annotate(text, (x, y))

    plt.tight_layout()
    plt.show()
