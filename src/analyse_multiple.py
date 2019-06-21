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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', help='Input file')

    args = parser.parse_args()

    df = None

    # We collate *all* of the files instead of using single one. This
    # gives us more flexibility. In theory, this should also work for
    # files in which multiple measurements are present.
    for filename in args.FILE:

        with open(filename) as f:
            data = json.load(f)

        assert data

        name = data['name']
        accuracies = collate_performance_measure('accuracy', data)

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
