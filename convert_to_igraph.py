#!/usr/bin/env python3
#
# convert_to_igraph.py: converts a benchmark data set to an `igraph`
# graph. This code attempts to store all types of information stored
# in the files, including labels and attributes.


import argparse
import logging
import os

import igraph as ig
import numpy as np

def get_data_set_name(directory):

    # The basename of any data set can be read off from its directory;
    # at least, that is what we assume here.
    basename = os.path.basename(directory)

    return basename


def get_adjacency_matrix_path(directory):
    name = get_data_set_name(directory)
    path = os.path.join(directory, f'{name}_A.txt')

    return path, os.path.exists(path)


def get_graph_indicator_path(directory):
    name = get_data_set_name(directory)
    path = os.path.join(directory, f'{name}_graph_indicator.txt')

    return path, os.path.exists(path)


def get_graph_labels_path(directory):
    name = get_data_set_name(directory)
    path = os.path.join(directory, f'{name}_graph_labels.txt')

    return path, os.path.exists(path)


def get_edge_labels_path(directory):
    name = get_data_set_name(directory)
    path = os.path.join(directory, f'{name}_edge_labels.txt')

    return path, os.path.exists(path)


def get_node_labels_path(directory):
    name = get_data_set_name(directory)
    path = os.path.join(directory, f'{name}_node_labels.txt')

    return path, os.path.exists(path)


def load_graphs(directory):
    '''
    Loads a set of graphs from the specified directory. The attributes
    and labels will be loaded automatically, if present.
    '''

    logging.info('Loading adjacency matrix and graph indicator files...')

    A = np.loadtxt(get_adjacency_matrix_path(directory)[0], delimiter=',')
    I = np.loadtxt(get_graph_indicator_path(directory)[0])

    # Total number of edges stored in the full adjacency matrix of *all*
    # graphs; while the total number of vertices is taken from `I`, i.e.
    # the graph indicator matrix.
    n_edges = A.shape[0]
    n_vertices = I.shape[0]

    # Get source indices and target indices; note that we correct the
    # data format because we want *our* indices to start at zero.
    source_indices = (A[:, 0] - 1).astype(int)
    target_indices = (A[:, 1] - 1).astype(int)

    assert np.min(source_indices) >= 0
    assert np.min(target_indices) >= 0

    # Build large adjacency matrix that contains *all* adjacencies
    # between all graphs.
    all_adjacencies = np.zeros((n_vertices, n_vertices), dtype=int)
    for index in range(n_vertices):
        source_index = source_indices[index]
        target_index = target_indices[index]

        all_adjacencies[source_index, target_index] = 1

    # The graph indicator matrix is supposed to start with an index of
    # `1`, as well. This needs to be corrected for.
    assert np.min(I) == 1

    I = I - 1
    n_graphs = len(np.unique(I))

    graphs = []

    # Check whether there are vertex (node) labels. We load them prior
    # to creating the graph because we want to add them directly.
    path, exists = get_node_labels_path(directory)
    if exists:

        logging.info('Loading node labels...')

        # TODO: do we have to support textual node labels as well? I am
        # not aware of these in the benchmark data sets so far.
        node_labels = np.loadtxt(path)
    else:
        node_labels = None

    # Check whether there are edge labels. Again, we load them prior to
    # creating the graph because we want to add them directly.
    path, exists = get_edge_labels_path(directory)
    if exists:

        logging.info('Loading edge labels...')

        # TODO: do we have to support textual edge labels as well? I am
        # not aware of these in the benchmark data sets so far.
        edge_labels = np.loadtxt(path)
    else:
        edge_labels = None

    logging.info('Starting graph creation process...')

    # Create basic graph structure from adjacency matrix. This does
    # *not* yet add any vertices or labels.
    for index in range(n_graphs):
        graph_indices = np.where(I == index)[0]

        local_adjacencies = all_adjacencies[graph_indices, :]
        local_adjacencies = local_adjacencies[:, graph_indices]

        g = ig.Graph.Adjacency((local_adjacencies > 0).tolist())

    # Get graph labels; note that this file *has* to exist because we
    # cannot do any classification otherwise.
    y = np.loadtxt(get_graph_labels_path(directory)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input directory')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=None
    )

    graphs = load_graphs(args.INPUT)
