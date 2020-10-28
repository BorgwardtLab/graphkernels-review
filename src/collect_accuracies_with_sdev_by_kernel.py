import os
import argparse

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file')

    args = parser.parse_args()

    df = pd.read_csv(args.FILE, header=0, index_col=0)
    df = df.transpose()

    for column in df.columns:
        values = df[column].values

        # Will contain the accuracies (first column), followed by the
        # standard deviations (second columns).
        data = []

        for value in values:
            if value is not np.nan:
                x, y = value.split('+-')
                x = float(x.strip())
                y = float(y.strip())

                data.append((round(x,2), round(y,2)))

        data = pd.DataFrame(data)
        print(column)
        
        # check that ouptut director exists, if not create it
        if not os.path.exists('../output/accuracies/'):
            os.makedirs('../output/accuracies/')


        data.to_csv(f'../output/accuracies/{column}_accuracies.csv',
                header=False)#, data, fmt='%2.2f')
