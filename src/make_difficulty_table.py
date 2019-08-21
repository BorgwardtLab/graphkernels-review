#!/usr/bin/env python3
#
# make_difficulty_table.py: creates a table of difficulties of graph
# kernels for each data set, by calculating the minimum, maximum and
# standard deviation, as well as a delta.

import argparse

import numpy as np
import pandas as pd

import pandas
import tabulate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file')

    args = parser.parse_args()

    df = pd.read_csv(args.FILE, header=0, index_col=0)
    df = df.replace(0.0, np.nan)
    df = df.transpose()

    # Rows now contain a data set, while columns contain the individual
    # graph kernels. This makes it possible to easily create statistics
    # about them.

    min_accuracy = df.min(axis=1)
    max_accuracy = df.max(axis=1)
    avg_accuracy = df.mean(axis=1)
    std_accuracy = df.std(axis=1)
    delta_accuracy = max_accuracy - min_accuracy

    columns_to_drop = df.columns
    df = df.drop(columns_to_drop, axis=1)

    df['min'] = min_accuracy
    df['max'] = max_accuracy
    df['avg'] = avg_accuracy
    df['std'] = std_accuracy
    df['delta'] = delta_accuracy

    print(
        tabulate.tabulate(
            df,
            tablefmt='plain',
            floatfmt='2.2f',
            headers='keys')
    )
