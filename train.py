#!/usr/bin/env python3
#
# train.py: given a set of kernel matrices, which are assumed to belong
# to the *same* data set, fits and trains a classifier, while reporting
# the best results.

import argparse
import logging
import os

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm


def grid_search_cv(
    clf,
    train_indices,
    n_folds,
    param_grid,
    kernel_matrices,
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

    :return: Best classifier, i.e. the classifier with the best
    parameters. Needs to be refit prior to predicting labels on
    the test data set. Moreover, the best-performing matrix, in
    terms of the grid search, is returned. It has to be used in
    all subsequent prediction tasks.
    '''

    y = kernel_matrices['y'][train_indices]

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42  # TODO: make configurable
    )

    # From this point on, `train_index` and `test_index` are supposed to
    # be understood *relative* to the input training indices.
    for train_index, test_index in cv.split(train_indices, y):
        for parameter, K in kernel_matrices.items():

            # Skip labels; we could also remove them from the set of
            # matrices but this would make the function inconsistent
            # because it should *not* fiddle with the input data set
            # if it can be avoided.
            if parameter == 'y':
                continue

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

def train_and_test(train_indices, test_indices, matrices):
    '''
    Trains the classifier on a set of kernel matrices (that are all
    assumed to come from the same algorithm). This uses pre-defined
    splits to prevent information leakage.

    :param train_indices: Indices to be used for training
    :param test_indices: Indices to be used for testing
    :param matrices: Kernel matrices belonging to some algorithm
    '''

    # Parameter grid for the classifier, but also for the 'pre-processing'
    # of a kernel matrix.
    param_grid = {
        'C': 10. ** np.arange(-3, 4),  # 10^{-3}..10^{3}
        'normalize': [False, True]
    }

    clf, K = grid_search_cv(
        SVC(kernel='precomputed'),
        train_indices,
        n_folds=5,
        param_grid=param_grid,
        kernel_matrices=matrices
    )

    # Refit the classifier on the test data set; using the kernel matrix
    # that performed best in the hyperparameter search.
    K_train = K[train_indices][:, train_indices]
    clf.fit(K_train, y[train_indices])

    y_test = y[test_indices]
    K_test = K[test_indices][:, train_indices]
    y_pred = clf.predict(K_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'{accuracy * 100:2.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'MATRIX', nargs='+',
        type=str,
        help='Input kernel matrix'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=None
    )

    logging.info('Loading input data...')

    # Load *all* matrices; each one of the is assumed to represent
    # a certain input data set.
    matrices = {
        os.path.splitext(os.path.basename(filename))[0]:
            np.load(filename) for filename in tqdm(args.MATRIX, desc='File')
    }

    logging.info('Checking input data and preparing splits...')

    n_graphs = None
    y = None

    for name, matrix in tqdm(matrices.items(), 'File'):
        for parameter in matrix:

            M = matrix[parameter]

            if parameter != 'y':
                # A kernel matrix needs to be square
                assert M.shape[0] == M.shape[1]
            else:
                if y is None:
                    y = M
                else:
                    assert y.shape == M.shape

            # Either set the number of graphs, or check that each matrix
            # contains the same number of them.
            if n_graphs is None:
                n_graphs = M.shape[0]
            else:
                assert n_graphs == M.shape[0]

    clf = SVC(kernel='precomputed')

    # Prepare cross-validated indices for the training data set.
    # Ideally, they should be loaded from *outside*.
    all_indices = np.arange(n_graphs)
    n_iterations = 10
    n_folds = 10

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42 # TODO: make configurable?
    )

    for iteration in range(n_iterations):
        for train_index, test_index in cv.split(all_indices, y):
            train_indices = all_indices[train_index]
            test_indices = all_indices[test_index]

            for name, matrix in matrices.items():
                # Main function for training and testing a certain kernel
                # matrix on the data set.
                train_and_test(
                    train_indices,
                    test_indices,
                    matrix
                )
