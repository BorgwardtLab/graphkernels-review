#!/usr/bin/env python3
#
# train.py: given a set of kernel matrices, which are assumed to belong
# to the *same* data set, fits and trains a classifier, while reporting
# the best results.

import argparse
import os

import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm


# Parameter grid for the classifier, but also for the 'pre-processing'
# of a kernel matrix.
param_grid = {
    'C': 10. ** np.arange(-3, 4),  # 10^{-3}..10^{3}
    'normalize': [False, True]
}


def grid_search_cv(
    clf,
    train_indices,
    n_folds,
    param_grid,
    kernel_matrices,
    y
):
    '''
    Internal grid search routine for a set of kernel matrices. The
    routine will use a pre-defined set of train indices to use for
    the grid search. Other indices will *not* be considered. Thus,
    information leakage is prevented.


    :param clf: Classifier to fit
    :param train_indices: Indices permitted to be used for cross-validation
    :param n_folds: Number of folds for the cross-validation
    :param param_grid: Parameters for the grid search
    :param kernel_matrices: Kernel matrices to check; each one of them
    is assumed to represent a different choice of parameter. They will
    *all* be checked iteratively by the routine.
    :param y: Labels (used for scoring)

    :return: Best classifier, fitted using the best parameters and ready
    for further predictions.
    '''

    # Custom model for an array of precomputed kernels
    # 1. Stratified K-fold
    #cv = StratifiedKFold(n_splits=cv, shuffle=False)
    #results = []
    #for train_index, test_index in cv.split(precomputed_kernels[0], y):
    #    split_results = []
    #    params = [] # list of dict, its the same for every split
    #    # run over the kernels first
    #    for K_idx, K in enumerate(precomputed_kernels):
    #        # Run over parameters
    #        for p in list(ParameterGrid(param_grid)):
    #            sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), 
    #                    train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
    #            split_results.append(sc)
    #            params.append({'K_idx': K_idx, 'params': p})
    #    results.append(split_results)
    ## Collect results and average
    #results = np.array(results)
    #fin_results = results.mean(axis=0)
    ## select the best results
    #best_idx = np.argmax(fin_results)
    ## Return the fitted model and the best_parameters
    #ret_model = clone(model).set_params(**params[best_idx]['params'])
    #return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'MATRIX', nargs='+',
        type=str,
        help='Input kernel matrix'
    )

    args = parser.parse_args()

    # Load *all* matrices; each one of the is assumed to represent
    # a certain input data set.
    matrices = {
        os.path.splitext(os.path.basename(filename))[0]:
            np.load(filename) for filename in tqdm(args.MATRIX, desc='File')
    }
