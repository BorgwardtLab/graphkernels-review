#!/usr/bin/env python3
#
# analyse.py: analyses a given JSON result file and creates a simple
# output table.


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
    parser.add_argument('FILE', help='Input file')

    args = parser.parse_args()

    with open(args.FILE) as f:
        data = json.load(f)

    assert data

    name = data['name']
    accuracies = collate_performance_measure('accuracy', data)

    df = pd.DataFrame.from_dict(accuracies, orient='index')
    df = df * 100
    mean = df.mean(axis=1)
    std = df.std(axis=1)

    df['mean'] = mean
    df['std'] = std

    print(
      tabulate.tabulate(
        df,
        tablefmt='github',
        floatfmt='2.2f',
        headers='keys',
      )
    )
