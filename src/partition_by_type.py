#!/usr/bin/env python3
#
# partition_by_type.py: partitions kernel values by data set type,
# making it possible to create point plots or box plots.

import pandas as pd

import sys
import os

from collections import Counter

# Please refer to the survey paper for more information about this
# description/categorisation.
name_to_type = {
    'AIDS'              : 'iii',  
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

    kernels = [name for name in df.columns.values]

    accuracies_per_type = {
        c: pd.DataFrame(columns=kernels) for c in sorted(name_to_type.values())
    }

    for index, row in df.iterrows():

        # There's probably a reason for us removing the data set from
        # the list of all data sets!
        if index not in name_to_type:
            print('Skipping data set', index, 'because its type is unknown.')
            continue

        data_set_type = name_to_type[index]

        accuracies_per_type[data_set_type] = \
            accuracies_per_type[data_set_type].append(row)

    # Rank based on mean accuracies
    for c in sorted(set(name_to_type.values())):
        df = accuracies_per_type[c]
        df = df.transpose()

        print('-' * 72)
        print(f'Class {c}:')
        print('-' * 72, '\n')
        print('name,data_set,accuracy')
        
        # check that ouptut director exists, if not create it
        if not os.path.exists('../output/aurocs_per_class/'):
            os.makedirs('../output/aurocs_per_class/')

        for index, row in df.iterrows():
            name = index

            with open(f'../output/aurocs_per_class/{c}_{name}.csv', 'w') as f:
                print('data_set,accuracy', file=f)
                for i, value in enumerate(row.values):
                    data_set = row.index[i]
                    print(f'{data_set},{value}', file=f)

        print('')
