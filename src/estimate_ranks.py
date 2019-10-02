#!/usr/bin/env python3
#
# estimate_ranks.py: estimates the ranks of individual graph kernels on
# the benchmark data sets by employing a sampling technique. Everything
# is based on the assumption that accuracy and standard deviation are a
# well-described by a normal distribution.
#
# Sampling from this distribution multiple times and *then* calculating
# a rank is much more stable.

import numpy as np
import pandas as pd

import sys


def sample(x):

    # Check if we have a valid sample or not
    if type(x) == float and np.isnan(x):
        return 0.0

    mu, sigma = x.split('+-')
    mu = float(mu)
    sigma = float(sigma)

    return np.random.normal(loc=mu, scale=sigma)


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], header=0, index_col=0)
    n_samples = 10000

    df_ranks_all = None

    for i in range(n_samples):
        df_sample = df.applymap(sample)
        df_ranks = df_sample.rank(axis=0, ascending=False, method='average')

        if df_ranks_all is None:
            df_ranks_all = df_ranks
        else:
            df_ranks_all = df_ranks_all + df_ranks

    df_ranks_all /= n_samples

    mean = df_ranks_all.mean(axis=1)
    std = df_ranks_all.std(axis=1)

    df = pd.DataFrame(columns=['mean', 'std'])
    df['mean'] = mean
    df['std'] = std

    pd.options.display.float_format = '{:,.2f}'.format
    print(df)
