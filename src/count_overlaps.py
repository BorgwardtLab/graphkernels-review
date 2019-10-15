#!/usr/bin/env python3
#
# count_overlaps.py: collects accuracies and standard deviations of all
# graph kernels from a large CSV file, and calculates how many overlaps
# there are. Here, an overlap indicates that the performance of kernels
# cannot be distinguished on a per-fold basis.

import argparse

import numpy as np
import pandas as pd


def overlaps(m0, s0, m1, s1):
    '''
    Checks whether an interval defined by a mean and a standard
    deviation overlaps with another.
    '''

    a = m0 - s0
    b = m0 + s1
    c = m1 - s1
    d = m1 + s1

    return (b >= c and a <= d) or (d >= a and c <= b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file')

    args = parser.parse_args()

    df = pd.read_csv(args.FILE, header=0, index_col=0)

    print('data_set,n_overlaps,n_pairs')

    for column in df.columns:
        data_set, values = df[column].name, df[column].values
        algorithm = column

        data = []

        for value in values:
            if value is not np.nan:
                m, s = value.split('+-')
                m = float(m.strip())
                s = float(s.strip())

                data.append((m, s))

        n_overlaps = 0
        n_pairs = 0

        for i, (m0, s0) in enumerate(data):
            for j, (m1, s1) in enumerate(data[i+1:]):
                k = j + i + 1
                if overlaps(m0, s0, m1, s1):
                    n_overlaps += 1

                n_pairs += 1

        print(f'{data_set},{n_overlaps},{n_pairs}')
