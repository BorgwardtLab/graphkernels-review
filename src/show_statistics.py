#!/usr/bin/env python3
#
# show_statistics.py: collects statistics of graph kernel benchmark data
# sets and formats them in a table.
#
# Usage: show_statistics.py DIRECTORY
#
# The script assumes that the specified directory contains
# subdirectories with `.pickle` files for processing graph
# data sets.

import argparse
import logging
import glob
import os

import igraph as ig
import numpy as np
import pandas as pd

from tqdm import tqdm


def get_statistics(graphs, name):
    '''
    Calculates statistics for a given data set of graphs and returns a
    `dict` data structure for further processing.

    :param graphs: Graph data set
    :param name: Data set name
    '''

    n_graphs = len(graphs)

    n_classes = len(set([g['label'] for g in graphs]))

    n_nodes = [len(g.vs) for g in graphs]
    n_edges = [len(g.es) for g in graphs]

    V = np.mean(n_nodes)
    E = np.mean(n_edges)

    avg_density = 2 * E  / (V * (V - 1)) 

    has_node_labels = ['label' in g.vs.attributes() for g in graphs]
    has_edge_labels = ['label' in g.es.attributes() for g in graphs]

    dim_node_attributes = None
    dim_edge_attributes = None

    has_node_attributes = ['attribute' in g.vs.attributes() for g in graphs]
    has_edge_attributes = ['attribute' in g.es.attributes() for g in graphs]

    if np.all(has_node_attributes):
        dim_node_attributes = np.mean(
            [len(g.vs['attribute'][0]) for g in graphs]
        )

        # If node attributes exist, their dimensionality *must* be an
        # integer.
        assert dim_node_attributes.is_integer()
        dim_node_attributes = int(dim_node_attributes)

    if np.all(has_edge_attributes):
        dim_edge_attributes = np.mean(
            [len(g.es['attribute'][0]) for g in graphs]
        )

        # If edge attributes exist, their dimensionality *must* be an
        # integer.
        assert dim_edge_attributes.is_integer()
        dim_edge_attributes = int(dim_edge_attributes)

    return {
        'name': name,
        'n_graphs': n_graphs,
        'n_classes': n_classes,
        'avg_n_nodes': f'{np.mean(n_nodes):.2f}',
        'avg_n_edges': f'{np.mean(n_edges):.2f}',
        'avg_density': f'{avg_density:.2f}',
        'has_node_labels': np.all(has_node_labels),
        'has_edge_labels': np.all(has_edge_labels),
        'dim_node_attributes': dim_node_attributes,
        'dim_edge_attributes': dim_edge_attributes,
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('DIRECTORY', help='Input directory', type=str)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=None
    )

    logging.info(f'Traversing directory {args.DIRECTORY}...')

    # Will contain individual rows with information about one data set;
    # together, these will become a nice data frame with statistics for
    # each data set.
    rows = []

    for root, dirs, files in os.walk(args.DIRECTORY, topdown=True):
        for directory in sorted(dirs):
            files = sorted(glob.glob(
                os.path.join(root, directory, '*.pickle'))
            )

            graphs = [
                ig.read(filename, format='picklez') for filename in
                tqdm(files, desc='File')
            ]

            # Use the directory name as the data set name
            name = directory
            rows.append(get_statistics(graphs, name))

            # FIXME: Handle more than one data set :)
            break

        break

    df = pd.DataFrame(rows)
    print(df)
