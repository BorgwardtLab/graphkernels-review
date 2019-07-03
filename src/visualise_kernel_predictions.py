#!/usr/bin/env python3
#
# visualise_kernel_predictions.py: visualises the individual kernels by
# means of their actual predictions on all data sets. This script again
# provides different variants of the plot (based on different embedding
# schemes).

import argparse

from sklearn.manifold import MDS

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import numpy as np
import matplotlib.pyplot as plt


# TODO: stole this from another script; maybe there's a way to
# consolidate it?
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

    X = np.loadtxt(args.INPUT, delimiter=',')

    # FIXME: they should be part of the respective file; the current
    # script cannot do this.
    kernel_names = ['EH', 'GL', 'MLG', 'SP', 'VH', 'WL', 'WLOA']

    D = squareform(pdist(X, metric='hamming'))
    Y = embed_distance_matrix(D)

    plt.gca().set_aspect('equal')
    plt.scatter(Y[:, 0], Y[:, 1], c='k')

    for j, text in enumerate(kernel_names):
        x = Y[j, 0]
        y = Y[j, 1]

        plt.annotate(text, (x, y))

    plt.tick_params(
        bottom=False, top=False,
        left=False, right=False,
        labelbottom=False, labeltop=False,
        labelleft=False, labelright=False,
    )

    plt.tight_layout()
    plt.show()
