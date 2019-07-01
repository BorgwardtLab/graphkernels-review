#!/usr/bin/env python3
#
# cat_matrices.py: script that concatenates a set of text-based matrices
# and stores them in `.npz` format with proper lablels. This is required
# in order to provide input for *other* graph kernels.


import argparse
import logging
import os
import sys

import numpy as np

from tqdm import tqdm


def get_parameters(filename):
    '''
    Extracts parameters from a filename. Parameters need to be separated
    by an underline, i.e. `_`. The method cannot assign names to params,
    so it will just report them in the order in which they were found.

    :param filename: Input filename
    :return: List of extracted parameters
    '''

    basename = os.path.basename(filename)
    basename = os.path.splitext(filename)[0]

    return basename.split('_')[1:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', type=str, help='Input file(s)')
    parser.add_argument(
        '-f', '--force', action='store_true',
        default=False,
        help='If specified, overwrites data'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file',
        required=True,
    )
    parser.add_argument(
        '-l', '--labels',
        type=str,
        help='Labels file',
        required=True
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=None
    )

    if os.path.exists(args.output):
        if not args.force:
            logging.info(
'''
Refusing to overwrite output file unless `-f` or `--force`
has been specified.
'''
            )

            sys.exit(0)

    # This array will be filled with the remaining matrices later on.
    # Right now, we can only add the labels.
    matrices = {
        'y': np.loadtxt(args.labels)

    }

    for filename in tqdm(args.FILES, desc='File'):
        parameters = get_parameters(filename)

        matrix = np.loadtxt(filename)
        key = '_'.join(parameters)

        # I know that it is somewhat stupid to use essentially the same
        # key here again as already in the filename, but it makes these
        # scripts a little bit more readable.
        matrices[key] = matrix

    np.savez(args.output, **matrices)
