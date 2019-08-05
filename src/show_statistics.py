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
import pandas as pd

from tqdm import tqdm


def get_statistics(graphs, name):
    '''
    Calculates statistics for a given data set of graphs and returns a
    `dict` data structure for further processing.

    :param graphs: Graph data set
    :param name: Name of data set
    '''

    return {}


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

        break
