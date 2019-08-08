#!/usr/bin/env python3
#
# calculate_ranks.py: calculates the ranks of individual graph kernels
# on the benchmark data sets.

import pandas as pd

import sys

from collections import Counter


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], header=0, index_col=0)

    df = df.rank(axis=0, ascending=False, method='average')

    mean = df.mean(axis=1)
    std = df.std(axis=1)

    df['mean'] = mean
    df['std'] = std
    df = df[['mean', 'std']]

    pd.options.display.float_format = '{:,.2f}'.format
    print(df.transpose())
