# `graphkernels-review`: Code and data sets for the review on graph kernels

## Generating kernel matrices

    ./src/create_kernel_matrices.py -o ./matrices/MUTAG ./data/MUTAG/*.pickle

## Training a classifier on a set of kernel matrices

    ./src/train.py ./matrices/MUTAG/*.npz -n MUTAG -o ./results/MUTAG.json

## Example output

|     |     0 |     1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |   mean |   std |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-------|
| EH  | 85.51 | 85.01 | 84.54 | 85.17 | 84.48 | 85.73 | 84.59 | 85.78 | 84.62 | 83.95 |  84.94 |  0.61 |
| GL  | 67.63 | 68.10 | 66.52 | 67.57 | 66.52 | 68.74 | 67.07 | 67.54 | 69.82 | 67.07 |  67.66 |  1.02 |
| SP  | 83.12 | 82.98 | 79.82 | 84.48 | 84.65 | 83.18 | 83.46 | 82.09 | 84.56 | 85.50 |  83.38 |  1.61 |
| VEH | 84.57 | 82.42 | 81.84 | 82.12 | 85.15 | 86.29 | 84.81 | 84.68 | 82.46 | 83.48 |  83.78 |  1.52 |
| VH  | 86.07 | 84.54 | 86.12 | 85.73 | 85.70 | 85.82 | 85.15 | 86.37 | 86.18 | 86.59 |  85.83 |  0.60 |
| WL  | 83.63 | 80.70 | 86.12 | 82.87 | 86.26 | 87.29 | 86.65 | 83.68 | 87.89 | 85.95 |  85.10 |  2.28 |

## Using the repository

Create symbolic links:

- `data`: symbolic link to `/cluster/work/borgw/graphkernels-review/data`
- `matrices`: symbolic link to `/cluster/work/borgw/graphkernels-review/matrices`
