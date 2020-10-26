#!/usr/bin/env python3
#
# collect_accuracies.py: collects accuracies of all graph kernels from
# a large CSV file and stores them in single files (one per data set).

import os
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
        values = np.array(values[values > 0])

        # check that ouptut director exists, if not create it
        if not os.path.exists('../output/Boxplots/'):
            os.makedirs('../output/Boxplots/')

        np.savetxt(f'../output/Boxplots/{column}.txt', values, fmt='%2.2f')
