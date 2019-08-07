#!/usr/bin/env python3
#
# count_winners.py: counts how often a certain algorithm performs best
# on the benchmark data sets.

import pandas as pd

import sys

from collections import Counter


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], header=0, index_col=0)
    df = df.transpose()

    winners = Counter({name: 0 for name in df.columns.values})

    for index, row in df.iterrows():
        winner = row.idxmax()
        winners[winner] += 1

    print(winners.most_common())
