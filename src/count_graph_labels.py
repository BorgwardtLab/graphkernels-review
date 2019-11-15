#!/usr/bin/env python3
#
# count_graph_labels.py: debug script to count the number of graph labels
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

    graph_labels = collections.Counter()

    for graph in graphs:
        graph_labels[graph['label']] += 1

    print(f'There are {len(graph_labels.keys())} unique graph labels')
    print(f'Their counts are {graph_labels.most_common()}')
