#!/usr/bin/env python3
#
# analyse.py: analyses a given JSON result file and creates a simple
# output table.


import argparse
import collections
import json


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

    # Needs to contain a list by default because each performance
    # measure is reported for every iteration and *every* kernel.
    results = collections.defaultdict(list)

    for iteration in data['iterations']:
        data_per_iteration = data['iterations'][iteration]
        kernels = data_per_iteration['kernels']

        for kernel in kernels:
            data_per_kernel = data_per_iteration['kernels'][kernel]
            values = data_per_kernel[measure]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', help='Input file')

    args = parser.parse_args()

    with open(args.FILE) as f:
        data = json.load(f)

    assert data

    name = data['name']
    accuracies = collate_performance_measure('accuracy', data)
