#!/usr/bin/env python3
#
# compare_baselines.py: compares baselines (i.e. the vertex histogram
# kernel) with all other algorithms and plots a relative distribution
# of the accuracy differences.

import pandas as pd

import sys


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], header=0, index_col=0)
    df = df.transpose()

    for index, row in df.iterrows():
        best_accuracy = row.max()
        hist_accuracy = row['VH']

        delta = 100 * (best_accuracy - hist_accuracy) / best_accuracy
        print(f'{delta:2.2f}')
