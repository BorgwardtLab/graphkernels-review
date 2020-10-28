#!/usr/bin/env python3


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

    # Stores predictions for each data set in a nested fashion. Each
    # data set will consist of a list of all its objects, which will
    # be repeated along all folds. Each list entry is a *set*, whose
    # entries correspond to those kernels that are able to predict a
    # label correctly.
    predictions_per_data_set = collections.defaultdict(list)

    # We collate *all* of the files instead of using single one. This
    # gives us more flexibility. In theory, this should also work for
    # files in which multiple measurements are present.
    
    files_to_ignore = [
            "/cluster/work/borgw/graphkernels-review-results/COIL-RAG_EH_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/COIL-RAG_SP_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/COIL-RAG_RW_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/COIL-RAG_CSM_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-low_EH_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-low_SP_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-low_RW_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-low_CSM_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-med_EH_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-med_SP_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-med_RW_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-med_CSM_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-high_EH_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-high_SP_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-high_RW_gkl.json",
            "/cluster/work/borgw/graphkernels-review-results/Letter-high_CSM_gkl.json",
            ]
    
    for filename in tqdm(args.FILE, desc='File'):
        if filename in files_to_ignore:
            continue
        else:
            with open(filename) as f:
                data = json.load(f)

            assert data

            name = data['name']

            predictions = concatenate_predictions('y_pred', data)
            labels = concatenate_labels('y_test', data)

            # Check whether the data set has to be set up first; this
            # involves creating a list that can contain the sets of a
            # fold.
            if name not in predictions_per_data_set.keys():

                # This is somewhat inelegant, because we pretend that we are
                # looping when in reality, we are *not*.
                for kernel, values in sorted(predictions.items()):
                    predictions_per_data_set[name] = [
                        set() for index in range(len(values))
                    ]
                    break

            # Check which labels coincide so that we can update the vector
            # of predictions accordingly.
            for kernel, values in sorted(predictions.items()):
                correct_labels = np.equal(values, labels)
                correct_indices = np.where(correct_labels == True)

                # Insert kernel name into the set of kernels that are able
                # to perform proper predictions.
                for index in correct_indices[0].ravel():
                    predictions_per_data_set[name][index].add(
                        kernel
                    )

    # Header for the output file; we do this manually because we are
    # mavericks.
    print('data_set,best_agreement,n_kernels')

    # Analyse the data sets now
    for name in sorted(predictions_per_data_set.keys()):

        counter = collections.Counter()

        for kernels in predictions_per_data_set[name]:
            frozen = frozenset(kernels)
            counter[frozen] += 1

        k, v = counter.most_common(1)[0]
        n = sum(counter.values())

        print(f'{name},{v/n},{len(k)}')

    print('')

    # Header for the output file; we do this manually because we are
    # mavericks.
    print('data_set,total_accuracy,n_kernels')

    # Analyse curves for each data set individually now. This only
    # involves counting the number of kernels in terms of their
    # cardinality.
    for name in sorted(predictions_per_data_set.keys()):

        counter = collections.defaultdict(lambda: 0)

        for kernels in predictions_per_data_set[name]:
            counter[len(kernels)] += 1

        n = sum(counter.values())
        total_accuracy = 0.0

        for k in sorted(counter.keys(), reverse=True):
            v = counter[k]
            total_accuracy += v/n
            
            print(f'{name},{total_accuracy},{k}')

        print('')

    # Header for the output file; we do this manually because we just
    # don't care.
    print('data_set,unclassifiable')

    # Check how many graphs fail to be classified by *all* of the
    # kernels, i.e. there is *no* kernel capable of classifying a
    # graph of that class correctly.
    for name in sorted(predictions_per_data_set.keys()):

        k = 0
        n = 0

        for kernels in predictions_per_data_set[name]:
            n += 1
            if len(kernels) == 0:
                k += 1

        print(f'{name},{k/n*100}')
