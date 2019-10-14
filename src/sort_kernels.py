#!/usr/bin/env python3
#
# sort_kernels.py: sorts kernels according to their *average* predictive
# performance over all benchmark data sets.

import pandas as pd

import sys

from collections import Counter


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], header=0, index_col=0)

    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df = df[['mean', 'std']]

    pd.options.display.float_format = '{:,.2f}'.format
    print(df.transpose())
