#!/usr/bin/env python3
#
# count_node_labels.py: debug script to count the number of node labels
# in a graph data set.

import argparse
import collections
import logging

import igraph as ig

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', type=str, help='Input file(s)')

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

    node_labels = collections.Counter()

    for graph in graphs:
        if 'label' in graph.vs.attributes():
            node_labels.update(graph.vs['label'])

    print(f'There are {len(node_labels.keys())} unique node labels')
    print(f'Their counts are {node_labels.most_common()}')
