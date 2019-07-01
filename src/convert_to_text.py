#!/usr/bin/env python3
#
# convert_to_text.py: Converts a set of a graphs to a textual
# representation in terms of their corresponding adjacencies.
# This format is used by the MLG kernel.

import argparse
import logging
import os
import sys
import traceback

import igraph as ig
import numpy as np

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', type=str, help='Input file(s)')
    parser.add_argument(
        '-f', '--force', action='store_true',
        default=False,
        help='If specified, overwrites data'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help='Output directory'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=None
    )

    logging.info('Loading graphs...')

    graphs = [
        ig.read(filename, format='picklez') for filename in
        tqdm(args.FILE, desc='File')
    ]

    os.makedirs(args.output, exist_ok=True)

    name = os.path.basename(args.output)
    adjacency_name = os.path.join(args.output, name + '_A.txt')
    labels_name = os.path.join(args.output, name + '_N.txt')

    if os.path.exists(adjacency_name) or os.path.exists(labels_name):
        if not args.force:
            logging.info('Output path already exists. Exiting.')
            sys.exit(0)

    with open(adjacency_name, 'w') as f, open(labels_name, 'w') as g:

        # Write header: number of graphs in total in the file
        print(len(graphs), file=f)
        print(len(graphs), file=g)

        for graph in graphs:
            A = graph.get_adjacency(attribute=None)
            A = np.array(A.data)

            # Make sure that this matrix is really symmetric
            assert A.shape[0] == A.shape[1]

            # Print adjacency matrix size, followed by the matrix
            # itself, and store it.
            print(A.shape[0], file=f)
            np.savetxt(f, A, delimiter=' ', fmt='%d')

            # Print number of node labels, followed by the label vector
            # it self, and store it. First, we have to check whether an
            # internal label exists. If not, we also use the degree.

            if 'label' in graph.vs.attributes():
                labels = np.array(graph.vs['label'])
            else:
                labels = np.array(graph.degree())

            print(len(labels), file=g)
            np.savetxt(g, labels, delimiter=' ', fmt='%d')
