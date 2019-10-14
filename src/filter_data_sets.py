#!/usr/bin/env python
#
# filter_data_sets.py: filters data sets from the list of all results if
# there is a graph kernel that was unable to classify them. This ensures
# that only a coreset of data sets that are classifiable is kept.


import pandas as pd

df = pd.read_csv('../results/Accuracies.csv', header=0, index_col=0)

columns_to_keep = []

for column in df.columns:
    if 0.0 not in df[column].values:
        columns_to_keep.append(column)

df = df[columns_to_keep]

print(df.to_csv(index=True))
