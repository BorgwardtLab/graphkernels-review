#!/usr/bin/env python
#
# filter_data_sets.py: filters data sets from the list of all results if
# there is a graph kernel that was unable to classify them. This ensures
# that only a coreset of data sets that are classifiable is kept.


import pandas as pd

df = pd.read_csv('../results/auroc.csv', header=0, index_col=0)

columns_to_keep = []
bad_columns = [
    # Solved data sets
    'AIDS',
    #'Letter-low',
    #'Synthie',
    #'SYNTHETICnew'
    # Simple data sets
    'BZR_MD',
    'COX2_MD',
    'KKI',
    'PTC_MM',
    ## Node-attributed data sets
    #'Letter-med',
    #'Letter-high',
    ##
    #'COIL-DEL',
    #'FRANKENSTEIN',
]

for column in df.columns:
    if 1000 not in df[column].values and column not in bad_columns:
        columns_to_keep.append(column)

df = df[columns_to_keep]

print(df.to_csv(index=True))
