#!/usr/bin/env python3
#
# collect_runtimes.py: collects runtime information from a directory
# tree of results and collates them *automatically* in a table. This
# table may then be used in subsequent tasks.


import argparse
import glob
import random
import os


import igraph as ig
import pandas as pd
import numpy as np


def calculate_graph_statistics(graphs):
    '''
    Calculates graph statistics, i.e. the average number of nodes and
    the average number of edges, and return them.

    :param graphs: Input graphs

    :return: Tuple containing the average number of nodes and the
    average number of edges.
    '''

    n_nodes = [g.vcount() for g in graphs]
    n_edges = [g.ecount() for g in graphs]

    return np.mean(n_nodes), np.mean(n_edges)


def process_directory(data_path, graph_path, data):
    '''
    Processes a directory and collects all files containing runtime
    information.

    :param data_path: Path to collect data from
    :param graph_path: Path to collect graphs from
    :param data: Dictionary to append data to

    :return: Modified dictionary
    '''

    name = os.path.basename(data_path)

    filenames = sorted(glob.glob(os.path.join(graph_path, name, '*.pickle')))

    # Just ignore all the graphs for which no sampling could be
    # performed anyway.
    if len(filenames) < 100:
        return data

    filenames = random.sample(filenames, 100)
    graphs = [ig.read(filename, format='picklez') for filename in filenames]
    avg_nodes, avg_edges = calculate_graph_statistics(graphs)

    time_filenames = sorted(glob.glob(os.path.join(data_path, 'Time_*.txt')))

    for filename in time_filenames:

        kernel = os.path.splitext(os.path.basename(filename))[0]
        kernel = kernel[kernel.find('_') + 1:]

        # Some of this information is static, but it has to be repeated
        # for every kernel nonetheless.
        data['name'].append(name)
        data['avg_nodes'].append(avg_nodes)
        data['avg_edges'].append(avg_edges)
        data['kernel'].append(kernel)

        with open(filename) as f:
            runtime = f.readlines()[0]
            runtime = float(runtime)

        data['runtime'].append(runtime)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ROOT', help='Root directory for results')
    parser.add_argument('GRAPHS', help='Root directory for graphs')

    args = parser.parse_args()

    # TODO: this is a dependency to another script (i.e. the timing
    # script for creating kernel matrices). I am too lazy to  solve
    # it differently for now.
    random.seed(42)

    # Stores data collected by the subsequent traversal routine. The
    # order of fields/values can change between calls of this script
    # because more data might be available.
    data = {
        'name': [],
        'avg_nodes': [],
        'avg_edges': [],
        'kernel': [],
        'runtime': []
    }

    for root, dirs, files in os.walk(args.ROOT, topdown=True):
        for dirname in sorted(dirs):
            data = process_directory(
                os.path.join(root, dirname),
                args.GRAPHS,
                data
            )

    df = pd.DataFrame.from_dict(data)
    df.to_csv('Runtimes.csv', header=True, index=False)
