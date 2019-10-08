#!/usr/bin/env python3
#
# get_best_accuracy.py: collects the *best*, i.e.\ highest, accuracy for
# each benchmark data set and prints it. This is an auxiliary script for
# facilitating other types of analyses.

import pandas as pd

import sys


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], header=0, index_col=0)
    df = df.transpose()

    for index, row in df.iterrows():
        best_accuracy = row.max()

        print(f'{best_accuracy:2.2f} '
              f'{index}'
        )
