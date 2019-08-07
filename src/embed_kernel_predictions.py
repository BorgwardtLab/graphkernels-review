#!/usr/bin/env python3
#
# embed_kernel_predictions.py: embeds the individual kernels into 2D by
# using their actual predictions on all data sets.

import argparse

from sklearn.manifold import MDS

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import numpy as np
import pandas as pd


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

    df = pd.read_csv(
        args.INPUT,
        header=0,
        index_col=0,
        na_filter=False,
        engine='c',
        low_memory=False,
    )

    X = df.values
    kernel_names = df.index.values

    D = squareform(pdist(X, metric='hamming'))
    Y = embed_distance_matrix(D)

    for name, coordinate in zip(kernel_names, Y):
        x = coordinate[0]
        y = coordinate[1]
        print(f'{x:2.2f}', f'{y:2.2f}', name)
