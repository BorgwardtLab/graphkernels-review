#!/usr/bin/env python3
#
# create_kernel_matrices.py: Given a set of graphs, applies a number of
# graph kernels and stores the resulting kernel matrices. Moreover, the
# script also stored the labels inside each set of matrices (under `y`)
# in order to make the output self-contained.

import argparse
import logging
import os
import sys

import graphkernels.kernels as gk
import igraph as ig
import numpy as np

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', type=str, help='Input file(s)')
    parser.add_argument('-a', '--algorithm',
        nargs='+',
        default=[],
        type=str,
        help='Indicates which algorithms to run'
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

    y = np.array([g['label'] for g in graphs])

    algorithms = {
        'EH': gk.CalculateEdgeHistKernel,
        # FIXME: does not yet work
        # 'CGL': gk.CalculateConnectedGraphletKernel,
        'GL': gk.CalculateGraphletKernel,
        'SP': gk.CalculateShortestPathKernel,
        'VEH': gk.CalculateVertexEdgeHistKernel,
        'VH': gk.CalculateVertexHistKernel,
        'WL': gk.CalculateWLKernel,
    }

    param_grid = {
        'WL': [0, 1, 2, 3, 4, 5, 6, 7],  # $h$ = number of iterations
        'GL': [3, 4, 5],                 # $k$ = size of graphlet
    }

    # Remove algorithms that have not been specified by the user; this
    # makes it possible to run only a subset of all configurations.
    algorithms = {
        k: v for k, v in algorithms.items() if k in args.algorithm
    }

    os.makedirs(args.output, exist_ok=True)

    for algorithm in sorted(tqdm(algorithms.keys(), desc='Algorithm')):

        # Function to apply to the list of graphs in order to obtain
        # a kernel matrix.
        f = algorithms[algorithm]

        if algorithm in param_grid.keys():

            try:
                matrices = {
                    str(param): f(graphs, par=param)
                    for param in param_grid[algorithm]
                }
            except NotImplementedError:
                logging.warning(f'''
Caught exception for {algorithm}; continuing with the next algorithm and
its corresponding parameter grid.
                ''')
                continue

            # Store the label vector of the graph data set along with
            # the set of matrices.
            matrices['y'] = y

            filename = os.path.join(args.output, f'{algorithm}.npz')
            np.savez(filename, **matrices)

        else:
            K = f(graphs)

            filename = os.path.join(args.output, f'{algorithm}.npz')
            np.savez(filename, K=K, y=y)
