#!/usr/bin/env python3
#
# count_winners_per_category.py: counts how often a certain algorithm
# performs best on the benchmark data sets, broken down by the data set
# type.

import pandas as pd

import sys

from collections import Counter

# Please refer to the survey paper for more information about this
# description/categorisation.
name_to_type = {
    'BZR'               : 'v',
    'BZR_MD'            : 'vi',
    'COIL-DEL'          : 'iv',
    'COIL-RAG'          : 'iv',
    'COLLAB'            : 'i',
    'COX2'              : 'v',
    'COX2_MD'           : 'vi',
    'DD'                : 'ii',
    'DHFR'              : 'v',
    'DHFR_MD'           : 'vi',
    'ENZYMES'           : 'v',
    'ER_MD'             : 'vi',
    'FRANKENSTEIN'      : 'iv',
    'IMDB-BINARY'       : 'i',
    'IMDB-MULTI'        : 'i',
    'KKI'               : 'ii',
    'Letter-high'       : 'iv',
    'Letter-low'        : 'iv',
    'Letter-med'        : 'iv',
    'MSRC_21'           : 'ii',
    'MSRC_21C'          : 'ii',
    'MSRC_9'            : 'ii',
    'MUTAG'             : 'iii',
    'Mutagenicity'      : 'iii',
    'NCI1'              : 'ii',
    'NCI109'            : 'ii',
    'OHSU'              : 'ii',
    'PROTEINS'          : 'ii',
    'PROTEINS_full'     : 'v',
    'PTC_FM'            : 'iii',
    'PTC_FR'            : 'iii',
    'PTC_MM'            : 'iii',
    'PTC_MR'            : 'iii',
    'Peking_1'          : 'ii',
    'REDDIT-BINARY'     : 'i',
    'REDDIT-MULTI-12K'  : 'i',
    'REDDIT-MULTI-5K'   : 'i',
    'SYNTHETIC'         : 'ii',
    'SYNTHETICnew'      : 'i',
    'Synthie'           : 'iv'
}


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], header=0, index_col=0)
    df = df.transpose()

    winners_per_type = {
        c: Counter() for c in sorted(name_to_type.values())
    }

    for index, row in df.iterrows():
        winner = row.idxmax()

        # There's probably a reason for us removing the data set from
        # the list of all data sets!
        if index not in name_to_type:
            print('Skipping data set', index, 'because its type is unknown.')
            continue

        data_set_type = name_to_type[index]
        winners_per_type[data_set_type][winner] += 1

    n = 0

    for c in sorted(set(name_to_type.values())):
        print(f'Class {c}:')
        winners = winners_per_type[c]
        print(winners.most_common(3))

        n += sum(winners.values())

    # Check that we are not missing anything
    assert n == len(name_to_type)
