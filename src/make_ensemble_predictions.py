#!/usr/bin/env python3
#
# make_ensemble_predictions.py: collects predictions of all graph
# kernels from a set of JSON files and treats them as an ensemble
# based on majority votes.

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
    and whose values constitute lists of lists vectors of the desired
    predictions. Each entry of the list corresponds to a certain fold.
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

        for kernel, values in sorted(results_per_fold.items()):
            results[kernel].append(values)

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

        results_per_fold = []

        for fold in sorted(data_per_iteration['folds']):
            data_per_fold = data_per_iteration['folds'][fold]
            labels = data_per_fold[label]

            # Ensures that the data types are correct later on
            labels = [str(label) for label in labels]

            results_per_fold.append(labels)

        results.append(results_per_fold)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', help='Input file')

    args = parser.parse_args()

    # Stores the *complete* data set, i.e. all predictions from all
    # kernels. The first level is a data set, while the second one,
    # i.e. the values of the first level, contain predictions *per*
    # graph kernel.
    all_predictions = collections.defaultdict(dict)

    # Stores all data set names and all kernel names, respectively.
    # Moreover, stores the number of predictions for a data set. We
    # assume that *if* a kernel runs, it makes predictions for each
    # sample in the data set.
    data_set_to_size = dict()
    kernel_names = set()

    # We collate *all* of the files instead of using a single one,
    # as this is more flexible.
    for filename in tqdm(args.FILE, desc='File'):

        with open(filename) as f:
            data = json.load(f)

        assert data

        # FIXME: this assumes that the number of iterations and the
        # number of folds is the same for all data sets. It is, but
        # only for our use case.
        n_iterations = len(data['iterations'])
        n_folds = len(data['iterations']['0']['folds'])

        name = data['name']
        predictions = concatenate_predictions('y_pred', data)
        labels = concatenate_labels('y_test', data)

        # Insert values into the global data dictionary, while making
        # sure that no re-ordering happens.
        for kernel, values in sorted(predictions.items()):
            all_predictions[name][kernel] = values

            # Data set has been seen; ensure that the size is correct
            if name in data_set_to_size:
                assert len(values) == data_set_to_size[name]
            else:
                data_set_to_size[name] = len(values)

            # Store kernel name so that we can unroll everything
            kernel_names.add(kernel)

        # Use this to indicate the original labels; we add it all the
        # time for every data set because this is easier than doing a
        # separate existence query.
        all_predictions[name]['XX'] = labels
        kernel_names.add('XX')

    # Go through *all* the data sets and their respective folds while
    # collecting predictions.
    for data_set in sorted(all_predictions.keys()):
        predictions_per_data_set = all_predictions[data_set]

        # This is indexed over each repetition (first axis) and each
        # fold (second axis).
        labels_per_fold = None

        predictions_array = [
            [[] for x in range(n_folds)] for y in
                range(n_iterations)
        ]

        for kernel in sorted(predictions_per_data_set.keys()):
            predictions = predictions_per_data_set[kernel]

            if kernel == 'XX':
                labels_per_fold = list(predictions)
            else:
                for i, preds_per_iteration in enumerate(predictions):
                    for j, preds_per_fold in enumerate(preds_per_iteration):

                        # If the array is empty, initialise it with a
                        # counter based on the labels of each object.
                        if len(predictions_array[i][j]) == 0:
                            predictions_array[i][j] = [
                                collections.Counter() for _ in preds_per_fold
                            ]

                        for k, label in enumerate(preds_per_fold):
                            predictions_array[i][j][k][str(label)] += 1

        # Perform a majority vote for this data set and only extract the
        # most common suggestion.
        for i in range(n_iterations):
            for j in range(n_folds):
                for k, c in enumerate(predictions_array[i][j]):
                    label, _ = c.most_common(1)[0]
                    predictions_array[i][j][k] = label

        accuracies = []

        # Pretend to re-run the prediction analysis for the data set.
        # This involves comparing the original labels with the votes,
        # and calculating a new accuracy and standard deviation.
        for i in range(n_iterations):

            accuracies_per_fold = []

            for j in range(n_folds):
                result = [
                    x == y for x, y in zip(predictions_array[i][j], \
                        labels_per_fold[i][j])
                ]

                n = len(result)
                accuracy = sum(result) / n

                accuracies_per_fold.append(accuracy)
            
            # This is the mean accuracy for the current repetition of
            # the splitting process.
            accuracies.append(np.mean(accuracies_per_fold))

        print(data_set,
              f'{np.mean(accuracies) * 100:2.2f}',
              '+-',
              f'{np.std(accuracies) * 100:2.2f}')
