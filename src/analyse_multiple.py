#!/usr/bin/env python3
#
# analyse_multiple.py: analyses multiple JSON result files and creates
# a simple output table. In contrast to the other script, this one can
# handle more than one JSON file.

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


def format_cell(x):
    '''
    Formats a cell for easier inclusion in a printed table.

    :param x: cell
    :return: Formatted cell; depending on the input, LaTeX characters
    will be added.
    '''

    if type(x) is float:
        return x

    # Replace underscores in any case in order to make it easier to
    # include the table somewhere.
    x = x.replace('_', '\\_')

    if '+-' in x:
        x = x.replace('+-', '\\pm')

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', help='Input file')
    parser.add_argument(
        '-m', '--measure',
        default='accuracy',
        type=str,
        help='Performance measure to collate for the analysis'
    )

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
        accuracies = collate_performance_measure(args.measure, data)

        df_local = pd.DataFrame.from_dict(accuracies, orient='index')
        df_local = df_local * 100

        mean = df_local.mean(axis=1)
        std = df_local.std(axis=1)

        # Replace everything that is not a mean
        df_local[name] = f'{mean.values[0]:2.2f} +- {std.values[0]:2.2f}'
        df_local = df_local[[name]]

        if df is None:
            df = df_local
        else:
            df = df.combine_first(df_local)

    print(
      tabulate.tabulate(
        df.transpose(),
        tablefmt='github',
        headers='keys',
      )
    )

    # Stores original data frame containing the mean accuracy values as
    # well as the standard deviations.
    df.to_csv(f'../results/{args.measure}_with_sdev.csv')

    # Store data frame containing nothing but the accuracies in order to
    # make it possible to *compare* graph kernels more meaningfully than
    # based on single tasks.
    df_vectorised = vectorise(df)
    df_vectorised.to_csv(f'../results/{args.measure}.csv')

    df = df.applymap(format_cell)

    print(
      tabulate.tabulate(
        df.transpose(),
        tablefmt='plain',
        headers='keys',
      )
    )
