#!/usr/bin/env python3
#
# visualise_kernel_accuracies.py: visualises the individual kernels by
# means of their accuracies on the individual data sets. Multiple plot
# variants will be provided by this script (based on different metrics
# or dissimilarity measures).

import argparse

from sklearn.manifold import MDS

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

    embedding = MDS(metric=True, dissimilarity='precomputed')
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
        {'metric': 'euclidean'},          # $L_2$
        {'metric': 'minkowski', 'p': 1},  # $L_1$
        {'metric': 'correlation'}
    ]

    fig, axes = plt.subplots(ncols=3, nrows=1, squeeze=True)

    for index, kwargs in enumerate(metrics):
        D = squareform(pdist(X, **kwargs))
        X = embed_distance_matrix(D)

        # Prepare plots (this is just for show; we are actually more
        # interested in the output files).
        title = kwargs['metric']
        axes[index].scatter(X[:, 0], X[:, 1])
        axes[index].set_title(title)

    plt.tight_layout()
