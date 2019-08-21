#!/usr/bin/env python3
#
# collect_accuracies_with_sdev.py: collects accuracies and standard
# deviations of all graph kernels from a large CSV file, and stores
# them in single files (one per data set).

import argparse

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file')

    args = parser.parse_args()

    df = pd.read_csv(args.FILE, header=0, index_col=0)

    for column in df.columns:
        values = df[column].values

        # Will contain the accuracies (first column), followed by the
        # standard deviations (second columns).
        data = []

        for value in values:
            if value is not np.nan:
                x, y = value.split('+-')
                x = float(x.strip())
                y = float(y.strip())

                data.append((x, y))

        data = np.array(data)
        np.savetxt(f'/tmp/{column}.txt', values, fmt='%2.2f')
