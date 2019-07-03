#!/usr/bin/env python3
#
# collect_predictions.py: collects predictions of all graph kernels from
# a set of JSON files and concatenates them into a large vector. This is
# a precursor to a more detailed analysis.

import argparse
import collections
import itertools
import json

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

        for fold in sorted(data_per_iteration['folds']):
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


def concatenate_labels(label, data):
    '''
    Concatenates the labels of a classifier on a given data set and
    returns them, following original fold order.

    :param label: Name of the label to extract.

    :param data: Nested dictionary of which to extract data. The
    function will throw an error if the required keys do not exist.

    :return: A list of all labels per folds encountered for the given
    data set. The list follows the original ordering of folds.
    '''

    results = []

    # Need to ensure that we iterate in a sorted manner over this data
    # frame because the order *needs* to be the same for each data set
    # that we encounter.
    for iteration in sorted(data['iterations']):
        data_per_iteration = data['iterations'][iteration]

        results_per_fold = collections.defaultdict(list)

        for fold in sorted(data_per_iteration['folds']):
            data_per_fold = data_per_iteration['folds'][fold]
            labels = data_per_fold[label]

            results.extend(labels)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', help='Input file')

    args = parser.parse_args()

    # Stores the *complete* data set, i.e. all predictions from all
    # kernels. The first level is a kernel, while the second level,
    # i.e. the values of the first level, contain predictions *per*
    # data set.
    all_predictions = collections.defaultdict(dict)

    # Stores all data set names and all kernel names, respectively.
    # Moreover, stores the number of predictions for a data set. We
    # assume that *if* a kernel runs, it makes predictions for each
    # sample in the data set.
    data_set_to_size = dict()
    kernel_names = set()

    # We collate *all* of the files instead of using single one. This
    # gives us more flexibility. In theory, this should also work for
    # files in which multiple measurements are present.
    for filename in tqdm(args.FILE, desc='File'):

        with open(filename) as f:
            data = json.load(f)

        assert data

        name = data['name']
        predictions = concatenate_predictions('y_pred', data)
        labels = concatenate_labels('y_test', data)

        # Insert values into the global data dictionary, while making
        # sure that no re-ordering happens.
        for kernel, values in sorted(predictions.items()):
            all_predictions[kernel][name] = values

            # Data set has been seen; ensure that the size is correct
            if name in data_set_to_size:
                assert len(values) == data_set_to_size[name]
            else:
                data_set_to_size[name] = len(values)

            # Store kernel name so that we can unroll everything
            kernel_names.add(kernel)

        # Use this to indicate the original labels
        if 'XX' not in all_predictions:
            all_predictions['XX'][name] = labels
            kernel_names.add('XX')

    # Unroll the prediction scores and create a new matrix that can be
    # stored. First, we need to collect all data set, though; it *may*
    # be possible that we have missing values for some data sets.

    n_rows = len(kernel_names)
    n_cols = sum([v for k, v in data_set_to_size.items()])

    X = np.zeros((n_rows, n_cols))

    # Kernels go into the rows, predictions go into the columns and are
    # unrolled as *one* big list.
    for row_index, kernel in enumerate(sorted(all_predictions.keys())):

        # Stores columns, indexed by data sets.
        columns = {}

        # Fill the column vector with NaNs. This ensures that everything
        # can be calculated correctly even if no predictions are present
        # for one of the kernels.
        for data_set, size in sorted(data_set_to_size.items()):
            x = np.empty((1, size))
            x[:] = np.nan

            columns[data_set] = x

        for data_set, values in sorted(all_predictions[kernel].items()):

            # Check that we are not doing something stupid with the
            # predictions
            assert len(values) == columns[data_set].shape[1]
            columns[data_set][0, :] = values

        x = np.concatenate([v for k, v in sorted(columns.items())],
                axis=1)

        X[row_index, :] = x

    # That's the professional way ;)
    X[np.isnan(X)] = -9999
    X = X.astype(np.int)

    np.savetxt(
        '../results/Predictions_raw.csv',
        X,
        delimiter=',',
        fmt='%d'
    )
