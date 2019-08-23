#!/usr/bin/env python3
#
# analyse_class_imbalance.py: analyses the class imbalance of data sets.
# The script assumes that data sets are stored as `.pickle` files in the
# following filesystem hierarchy:
#
#   $root/$name/...
#

import argparse
import collections
import glob
import os
import sys

import igraph as ig
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ROOT', type=str, help='Root directory')

    args = parser.parse_args()

    for root, dirs, files in os.walk(args.ROOT, topdown=True):
        for dirname in sorted(dirs):
            filenames = glob.glob(
                os.path.join(root, dirname, '*.pickle')
            )

            if dirname.startswith('Tox'):
                continue

            label_counter = collections.Counter()

            for filename in tqdm(filenames, desc='File'):
                graph = ig.read(filename, format='picklez')
                label = graph['label']
                label_counter[label] += 1

            print(
                f'{dirname}',
                list(label_counter.values()),
                sum(label_counter.values())
            )

            sys.stdout.flush()
