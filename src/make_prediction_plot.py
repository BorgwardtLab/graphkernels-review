#!/usr/bin/env python3

import numpy as np
import pandas as pd


df = pd.read_csv('../results/Accuracies_with_sdev.csv', header=0,
        index_col=0)

for index, row in df.iterrows():
    with open(f'/tmp/{index}_accuracies.csv', 'w') as f:
        for i, value in enumerate(row):
            if type(value) == float and np.isnan(value):
                x, y = 0, 0
            else:
                x, y = value.split('+-')
                x, y = float(x), float(y)
            
            print(i, x, y, file=f)
