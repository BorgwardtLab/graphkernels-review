#!/usr/bin/env python3

# create_kernel_matrices.py: Given a set of graphs, applies a number of
# grakel graph kernels and stores the resulting kernel matrices. Moreover, the
# script also stored the labels inside each set of matrices (under `y`)
# in order to make the output self-contained.


import argparse
import logging
import os
import random
import sys
import traceback

import grakel
import igraph as ig
import numpy as np

from timeit import time
from tqdm import tqdm

from grakel_util import *


def preprocess(graph):
    ''' Relabels nodes in graph with degree '''

    if 'label' not in graph.vs.attributes():
        graph.vs['label'] = graph.degree()

    return(graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'FILE', 
            nargs="+", 
            type=str, 
            help="Input file(s)"
        )

    parser.add_argument(
            "-a", "--algorithm",
            nargs="+",
            default=[],
            type=str,
            help="Indicates which algorithms to run"
        )

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

    parser.add_argument(
            '-t', '--timing', action='store_true',
            default=False,
            help='If specified, only stores timing information'
        )

    args = parser.parse_args()

    logging.basicConfig(
            level=logging.INFO,
            format=None
        )

    logging.info('Loading graphs...')

    if args.timing:
        logging.info("Choosing at most 100 graphs at random for timing")

        random.seed(42)
        args.FILE = random.sample(args.FILE, 100)

    graphs = [
            ig.read(args.FILE[0] + filename, format='picklez') for filename in
            tqdm(sorted(os.listdir(args.FILE[0])), desc='File')
            ]

    graphs = [
            preprocess(graph) for graph in tqdm(graphs, desc="Preprocessing")
            ]
    

    y = [g['label'] for g in graphs]
    graphs = igraph_to_grakel(graphs)
    algorithms = {
            "SP": grakel.kernels.ShortestPath(with_labels=True).fit_transform
            }

    param_grid = {
            "SP": [1]
            }

    # Remove algorithms that have not been specified by the user; this
    # makes it possible to run only a subset of all configurations.
    algorithms = {
            k: v for k, v in algorithms.items() if k in args.algorithm
            }

    os.makedirs(args.output, exist_ok=True)

    for algorithm in sorted(tqdm(algorithms.keys(), desc='Algorithm')):
        
        start_time = time.process_time()
        print(algorithm)

        # Filename for the current algorithm. We create this beforehand
        # in order to check whether we would overwrite something.
        filename = os.path.join(args.output, f'{algorithm}.npz')

        if os.path.exists(filename):
            if not args.force and not args.timing:
                logging.info('Output path already exists. Skipping.')
                continue

        # Function to apply to the list of graphs in order to obtain
        # a kernel matrix.
        f = algorithms[algorithm]

        if algorithm in param_grid.keys():
            print("yes")
            try:
                matrices = {
                        str(param): f(graphs) # I removed par=param
                        for param in param_grid[algorithm]
                        }
            except NotImplementedError:
                logging.warning(f'''Caught exception for {algorithm};
                continuing wiht the next algorithm and its corresponding
                parameter grid.''')

                traceback.print_exc()
                continue

            # Store the label vector of the graph data set along with
            # the set of matrices.
            matrices['y'] = y
            
            # We only save matrices if we are not in timing mode. In
            # somse sense, the calculations will thus be lost but we
            # should not account for the save time anyway.
            if not args.timing:
                np.savez(filename, **matrices)

        else:
            K = f(graphs)

        stop_time = time.process_time()

        # We overwrite this *all* the time because the information can
        # always be replaced easily.
        with open(os.path.join(args.output, f'Time_{algorithm}.txt'), 'w') as f:
            print(stop_time - start_time, file=f)

        # We only save the matrix if we are not in timing mode; see
        # above for the rationale.
        if not args.timing:
            np.savez(filename, K=K, y=y)

