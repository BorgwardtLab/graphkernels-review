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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    args = parser.parse_args()

    df = pd.read_csv(args.INPUT)
    X = df.iloc[:, 1:].to_numpy()

    kernel_names = df.iloc[:, 0].values
    data_set_names = df.iloc[0, :].values   # these are not required...

    metrics = [
        {'metric': 'minkowski', 'p': 1},  # $L_1$
        {'metric': 'minkowski', 'p': 2},  # $L_2$
        {'metric': 'cityblock'}
    ]

    fig, axes = plt.subplots(
        figsize=(16, 4),
        ncols=len(metrics),
        nrows=1,
        squeeze=True
    )

    for index, kwargs in enumerate(metrics):
        D = squareform(pdist(X, **kwargs))
        Y = embed_distance_matrix(D)

        # Prepare plots (this is just for show; we are actually more
        # interested in the output files).
        title = kwargs['metric']

        if 'p' in kwargs:
            title += '_' + str(kwargs['p'])

        axes[index].set_aspect('equal')
        axes[index].scatter(Y[:, 0], Y[:, 1])
        axes[index].set_title(title)

        for j, text in enumerate(kernel_names):
            x = Y[j, 0]
            y = Y[j, 1]
            axes[index].annotate(text, (x, y))

    plt.show()
