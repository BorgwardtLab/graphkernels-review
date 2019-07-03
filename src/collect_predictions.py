#!/usr/bin/env python3
#
# collect_predictions.py: collects predictions of all graph kernels from
# a set of JSON files and concatenates them into a large vector. This is
# a precursor to a more detailed analysis.

import argparse
import collections
import itertools
import json
import tabulate

import numpy as np
import pandas as pd

from tqdm import tqdm


def concatenate_predictions(prediction, data):
    '''
    Concatenates the predictions of a classifier on a given data set and
    returns them, following original fold order.

    :param prediction: Name of the prediction to extract. For example,
    if `y_pred` is used, the prediction of the kernel on the fold will
    be used . If `y_test` is used, the original labels are returned.

    :param data: Nested dictionary of which to extract data. The
    function will throw an error if the required keys do not exist.

    :return: Dictionary whose keys represent kernels on some data set,
    and whose values constitute vectors of the desired predictions.
    '''

    # Needs to contain a list by default because each performance
    # measure is reported for every iteration and *every* kernel.
    results = collections.defaultdict(list)

    # Need to ensure that we iterate in a sorted manner over this data
    # frame because the order *needs* to be the same for each data set
    # that we encounter.
    for iteration in sorted(data['iterations']):
        data_per_iteration = data['iterations'][iteration]

        results_per_fold = collections.defaultdict(list)

        for fold in data_per_iteration['folds']:
            data_per_fold = data_per_iteration['folds'][fold]

            # Store the desired measure per kernel and per fold; the
            # result is a list of values for each iteration.
            for kernel in data_per_fold['kernels']:
                data_per_kernel = data_per_fold['kernels'][kernel]
                predictions = data_per_kernel[prediction]
                results_per_fold[kernel].append(predictions)

        # Flatten the list and report a *single* vector per kernel,
        # which we can subsequently reuse. This is for an iteration
        # only, and we need to flatten it some more later on.
        for kernel, values in sorted(results_per_fold.items()):
            results[kernel].append(list(itertools.chain(*values)))

    results = {
        k: list(itertools.chain(*v)) for k, v in sorted(results.items())
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', help='Input file')

    args = parser.parse_args()

    df = None

    # We collate *all* of the files instead of using single one. This
    # gives us more flexibility. In theory, this should also work for
    # files in which multiple measurements are present.
    for filename in tqdm(args.FILE, desc='File'):

        with open(filename) as f:
            data = json.load(f)

        assert data

        name = data['name']
        predictions = concatenate_predictions('y_pred', data)

        #df_local = pd.DataFrame.from_dict(accuracies, orient='index')
        #df_local = df_local * 100

        #mean = df_local.mean(axis=1)
        #std = df_local.std(axis=1)

        ## Replace everything that is not a mean
        #df_local[name] = f'{mean.values[0]:2.2f} +- {std.values[0]:2.2f}'
        #df_local = df_local[[name]]

        #if df is None:
        #    df = df_local
        #else:
        #    df = df.combine_first(df_local)

    print(
      tabulate.tabulate(
        df.transpose(),
        tablefmt='github',
        headers='keys',
      )
    )

    # Stores original data frame containing the mean accuracy values as
    # well as the standard deviations.
    #df.to_csv('../results/Accuracies_with_sdev.csv')
