#!/usr/bin/env python3
#
# extract_average_depth.py: extracts the average depth of a graph kernel
# by treating the "K" parameter of the kernel matrix as the depth that
# was used to analyse the graph.

import argparse
import collections
import json
import tabulate

import numpy as np
import pandas as pd

from tqdm import tqdm


def collate_performance_measure(measure, data, aggregate='mean'):
    '''
    Collates a performance measure, such as accuracy, over a given data
    set, returning it for each kernel. This will permit subsequent data
    analysis steps.

    :param measure: Name of the performance measure; if no such measure
    can be found, the function will return an empty dictionary.

    :param data: Nested dictionary of which to extract data. The
    function will throw an error if the required keys do not exist.

    :param aggregate: Specifies an aggregation strategy for the
    performance measures of the individual folds. By default, a
    mean is calculated.

    :return: Dictionary whose keys represent a kernel and whose values
    represent lists of the desired performance measure.
    '''

    aggregate_fn = {
        'mean': np.mean
    }

    aggregate_fn = aggregate_fn[aggregate]

    # Needs to contain a list by default because each performance
    # measure is reported for every iteration and *every* kernel.
    results = collections.defaultdict(list)

    for iteration in data['iterations']:
        data_per_iteration = data['iterations'][iteration]

        results_per_fold = collections.defaultdict(list)

        for fold in data_per_iteration['folds']:
            data_per_fold = data_per_iteration['folds'][fold]

            # Store the desired measure per kernel and per fold; the
            # result is a list of values for each iteration.
            for kernel in data_per_fold['kernels']:
                data_per_kernel = data_per_fold['kernels'][kernel]
                value = data_per_kernel[measure]
                results_per_fold[kernel].append(value)

        # Aggregate values over all folds to obtain *one* value per
        # iteration because that is what we need to properly report
        # everything.
        for kernel, values in results_per_fold.items():
            aggregated_value = aggregate_fn(values)
            results[kernel].append(aggregated_value)

    return results


def collate_model_information(measure, data, aggregate='mean'):

    aggregate_fn = {
        'mean': np.mean
    }

    aggregate_fn = aggregate_fn[aggregate]

    # Needs to contain a list by default because each performance
    # measure is reported for every iteration and *every* kernel.
    results = collections.defaultdict(list)

    for iteration in data['iterations']:
        data_per_iteration = data['iterations'][iteration]

        results_per_fold = collections.defaultdict(list)

        for fold in data_per_iteration['folds']:
            data_per_fold = data_per_iteration['folds'][fold]

            # Store the desired measure per kernel and per fold; the
            # result is a list of values for each iteration.
            for kernel in data_per_fold['kernels']:
                data_per_kernel = data_per_fold['kernels'][kernel]
                value = data_per_kernel['best_model'][measure]

                try:
                    value = int(value)
                except ValueError:
                    return None

                results_per_fold[kernel].append(value)

        # Aggregate values over all folds to obtain *one* value per
        # iteration because that is what we need to properly report
        # everything.
        for kernel, values in results_per_fold.items():
            aggregated_value = aggregate_fn(values)
            results[kernel].append(aggregated_value)

    return results


def vectorise(df):
    '''
    Vectorises a data frame containing accuracy values. Each *row* in
    the resulting matrix will correspond to a given graph kernel, and
    each *column* will correspond to a specific task, i.e. a data set
    for classification.

    This function performs some normalisation: NaNs are replaced by a
    zero, which indicates failure on a task.

    :param df: Input data frame
    :return: Data frame in `float` format, containing only kernels and
    data sets.
    '''

    def get_accuracy(x):
        tokens = str(x).split()
        accuracy = tokens[0]
        accuracy = accuracy.replace('nan', '0.0')

        return float(accuracy)

    return df.applymap(get_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', help='Input file')

    args = parser.parse_args()

    df = pd.DataFrame(
        columns=['data', 'kernel', 'mean_accuracy', 'mean_depth']
    )

    # We collate *all* of the files instead of using single one. This
    # gives us more flexibility. In theory, this should also work for
    # files in which multiple measurements are present.
    for filename in tqdm(args.FILE, desc='File'):

        with open(filename) as f:
            data = json.load(f)

        assert data

        name = data['name']
        accuracies = collate_performance_measure('accuracy', data)
        depths = collate_model_information('K', data)

        # Skip file; it contains a kernel that does not support
        # different depths.
        if depths is None:
            continue

        df_mean = pd.DataFrame.from_dict(accuracies, orient='index')
        df_mean = df_mean * 100

        mean = df_mean.mean(axis=1)
        mean = mean.values[0]

        df_depths = pd.DataFrame.from_dict(depths, orient='index')

        mean_depth = df_depths.mean(axis=1)
        mean_depth = mean_depth.values[0]

        df = df.append(
            {
                'data': name,
                'kernel': df_mean.index.values[0],
                'mean_accuracy': mean,
                'mean_depth': mean_depth,
            },
            ignore_index=True
        )

    print(df.to_csv(index=False))
