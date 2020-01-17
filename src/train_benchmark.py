#!/usr/bin/env python3
#
# train_benchmark.py: given a set of kernel matrices, which are assumed to belong
# to the *same* data set, fits and trains a classifier, while reporting
# the best results.
# The classified is fitted to the training data in a 10-fold cross validation
# regime. The validation data performance (on the last chunk of the fold) is
# reported as well as the performance on the (benchmarked) test data is.
#
#   NOTE: In the output json all test data performances can be found by
#   appending `_bm` to the dictionary key, e.g. precision_bm.

import argparse
import collections
import logging
import json
import os
import sys
import warnings

import numpy as np

from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

# This is guaranteed to produce the *best* results in terms of time
# precision.
from timeit import default_timer as timer

from tqdm import tqdm


def normalize(matrix):
    '''
    Normalizes a kernel matrix by dividing through the square root
    product of the corresponding diagonal entries. This is *not* a
    linear operation, so it should be treated as a hyperparameter.

    :param matrix: Matrix to normalize
    :return: Normalized matrix
    '''

    # Ensures that only non-zero entries will be subjected to the
    # normalisation procedure. The remaining entries will be kept
    # at zero. This prevents 'NaN' values from cropping up.
    mask = np.diagonal(matrix) != 0
    n = len(np.diagonal(matrix))
    k = np.zeros((n, ))
    k[mask] = 1.0 / np.sqrt(np.diagonal(matrix)[mask])

    return np.multiply(matrix, np.outer(k, k))


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
    all subsequent prediction tasks. Additionally, the function
    also returns a dictionary of the best parameters.
    '''

    y = kernel_matrices['y'][train_indices]

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42  # TODO: make configurable
    )

    best_clf = None
    best_accuracy = 0.0
    best_parameters = {}

    # From this point on, `train_index` and `test_index` are supposed to
    # be understood *relative* to the input training indices.
    for train_index, test_index in cv.split(train_indices, y):
        for K_param, K in kernel_matrices.items():

            # Skip labels and benchmark indices;
            # we could also remove them from the set of
            # matrices but this would make the function inconsistent
            # because it should *not* fiddle with the input data set
            # if it can be avoided.
            if K_param == 'y' or K_param == 'benchmark_indices':
                continue

            # This ensures that we *cannot* access the test indices,
            # even if we try :)
            K = K[train_indices][:, train_indices]

            # Starts enumerating all 'inner' parameters, i.e. the ones
            # that pertain to the classifier (and potentially how this
            # matrix should be treated)
            for parameters in list(param_grid):
                if parameters['normalize']:
                    K = normalize(K)

                # Remove the parameter because it does not pertain to
                # the classifier below.
                clf_parameters = {
                    key: value
                    for key, value in parameters.items()
                    if key not in ['normalize']
                }

                accuracy, params = _fit_and_score(
                    clone(clf),
                    K,
                    y,
                    scorer=make_scorer(accuracy_score),
                    train=train_index,
                    test=test_index,
                    verbose=0,
                    parameters=clf_parameters,
                    fit_params=None,  # No additional parameters for `fit()`
                    return_parameters=True,
                )

                # Note that when storing the parameters, we can re-use
                # the original grid because we want to know about this
                # normalization.
                if accuracy > best_accuracy:
                    best_clf = clone(clf).set_params(**params)
                    best_accuracy = accuracy
                    best_parameters = parameters

                    # Update kernel matrix parameter to indicate which
                    # matrix was used to obtain these results. The key
                    # will also be returned later on.
                    best_parameters['K'] = K_param

    return clf, kernel_matrices[best_parameters['K']], best_parameters


def train_and_test(
    train_indices, test_indices, matrices, n_classes, max_iterations,
    benchmark_indices
):
    '''
    Trains the classifier on a set of kernel matrices (that are all
    assumed to come from the same algorithm). This uses pre-defined
    splits to prevent information leakage.

    :param train_indices: Indices to be used for training
    :param test_indices: Indices to be used for testing
    :param matrices: Kernel matrices belonging to some algorithm
    :param classes: Class labels
    :param max_iterations: Maximum number of iterations for SVM
    :param benchmark_indices: The performance of the classified on the
        benchmark dataset is also reported.

    :return: Dictionary containing information about the training
    process and the trained model.

    '''

    # Parameter grid for the classifier, but also for the 'pre-processing'
    # of a kernel matrix.
    param_grid = {
        'C': 10.**np.arange(-3, 4),  # 10^{-3}..10^{3}
        'normalize': [False, True]
    }

    clf, K, best_parameters = grid_search_cv(
        SVC(
            class_weight='balanced',
            kernel='precomputed',
            probability=True,
            max_iter=max_iterations
        ),
        train_indices,
        n_folds=5,
        param_grid=ParameterGrid(param_grid),
        kernel_matrices=matrices
    )

    # Refit the classifier on the test data set; using the kernel matrix
    # that performed best in the hyperparameter search.
    K_train = K[train_indices][:, train_indices]
    clf.fit(K_train, y[train_indices])

    y_test = y[test_indices]

    K_test = K[test_indices][:, train_indices]
    y_pred = clf.predict(K_test)
    y_score = clf.predict_proba(K_test)

    # Prediction-based measures
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    n_classes = len(classes)
    x = 12

    # Score-based measures; in the multi-class setting, they are not
    # available so we have to solve this manually.
    if n_classes == 2:
        auroc = roc_auc_score(y_test, y_score[:, 1])
        auprc = average_precision_score(y_test, y_score[:, 1])
    else:

        # This ensures that column $i$ can be used to access all labels
        # of the same class.
        y_test_binarized = label_binarize(y_test, classes=classes)

        aurocs = []
        auprcs = []

        # In some cases, scores cannot be returned for each of the
        # classes because the number of instances is insufficient.
        n_scores = min(n_classes, y_score.shape[1])

        for i in range(n_scores):
            try:
                auroc = roc_auc_score(
                    y_test_binarized[:, i],
                    y_score[:, i],
                )
                auprc = average_precision_score(
                    y_test_binarized[:, i], y_score[:, i]
                )

                aurocs.append(auroc)
                auprcs.append(auprc)

            # Ignore errors in the calculation and do *not* include the
            # results in the subsequent mean calculation.
            except ValueError:
                pass

        auroc = np.mean(aurocs)
        auprc = np.mean(auprcs)

    # Repeat the evaluation procedure for the benchmark
    y_benchmark = y[benchmark_indices]
    K_benchmark = K[benchmark_indices][:, train_indices]
    y_pred_bm = clf.predict(K_benchmark)
    y_score_bm = clf.predict_proba(K_benchmark)

    # Prediction-based measures
    accuracy_bm = accuracy_score(y_benchmark, y_pred_bm)
    precision_bm = precision_score(y_benchmark, y_pred_bm, average='macro')
    recall_bm = recall_score(y_benchmark, y_pred_bm, average='macro')

    if n_classes == 2:
        auroc_bm = roc_auc_score(y_benchmark, y_score_bm[:, 1])
        auprc_bm = average_precision_score(y_benchmark, y_score_bm[:, 1])

    results = {
        'best_model': best_parameters,
        'accuracy': accuracy,
        'accuracy_bm': accuracy_bm,
        'precision': precision,
        'precision_bm': precision_bm,
        'recall': recall,
        'recall_bm': recall_bm,
        'auroc': auroc,
        'auroc_bm': auroc_bm,
        'auprc': auprc,
        'auprc_bm': auprc_bm,
        'y_pred': y_pred.tolist(),
        'y_pred_bm': y_pred_bm.tolist(),
        'y_score': y_score.tolist(),
        'y_score_bm': y_score_bm.tolist(),
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'MATRIX', nargs='+', type=str, help='Input kernel matrix'
    )
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        default=False,
        help='If specified, overwrites data'
    )
    parser.add_argument(
        '-i',
        '--with-indices',
        action='store_true',
        default=False,
        help='If specified, stores indices (LARGE!) in JSON file'
    )
    parser.add_argument(
        '-n', '--name', type=str, help='Data set name', required=True
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output file',
        required=True,
    )
    parser.add_argument(
        '-I',
        '--max-iterations',
        type=int,
        help='Maximum number of iterations to use for training',
        default=1e5
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=None)

    logging.info(f'Using at most {args.max_iterations} iterations')

    if os.path.exists(args.output):
        if not args.force:
            logging.info(
                '''
Refusing to overwrite output file unless `-f` or `--force`
has been specified.
'''
            )

            sys.exit(0)

    logging.info('Loading input data...')

    # Load *all* matrices; each one of the is assumed to represent
    # a certain input data set.
    logging.info(args.MATRIX)
    matrices = {
        os.path.splitext(os.path.basename(filename))[0]: np.load(filename)
        for filename in tqdm(args.MATRIX, desc='File')
    }

    logging.info('Checking input data and preparing splits...')

    n_graphs = None
    y = None

    # I am aware of the fact that not all metrics are defined for all
    # data sets; there is no need to clutter up the output.
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    # Check input data by looking at the shape of matrices and their
    # corresponding label vectors.
    for name, matrix in tqdm(matrices.items(), 'File'):
        for parameter in matrix:

            M = matrix[parameter]

            if parameter != 'y' and parameter != 'benchmark_indices':
                # A kernel matrix needs to be square
                assert M.shape[0] == M.shape[1]
            elif parameter == 'y':
                if y is None:
                    y = M
                else:
                    assert y.shape == M.shape
            elif parameter == 'benchmark_indices':
                benchmark_indices = M

            # Either set the number of graphs, or check that each matrix
            # contains the same number of them.
            if n_graphs is None:
                n_graphs = M.shape[0]
            elif parameter != 'y' and parameter != 'benchmark_indices':
                assert n_graphs == M.shape[0]

    # Determine classes; this will be required later on when calculating
    # AUROC and AUPRC.
    classes = list(sorted(set(y)))

    # Infer the indices of the test samples

    # Prepare cross-validated indices for the training data set.
    # Ideally, they should be loaded from *outside*.

    # The assumption here is that the test/benchmark graphs are added last to
    # kernel_matrices. This is being ensured in create_kernel_matrices_benchmark
    all_indices = np.arange(n_graphs - len(benchmark_indices))
    y_opt = y[:-len(benchmark_indices)]
    n_iterations = 1
    n_folds = 10

    # Stores the results of the complete training process, i.e. over
    # *all* matrices, *all* folds, and so on.
    all_results = dict()
    all_results['name'] = args.name
    all_results['iterations'] = dict()

    # Prepare time measurement
    start_time = timer()

    for name, matrix in matrices.items():

        logging.info(f'Kernel name: {name}')

        for iteration in range(n_iterations):

            logging.info(f'Iteration {iteration + 1}/{n_iterations}')

            # Every matrix, i.e. every kernel, gets the *same* folds so that
            # these results do not have to be stored multiple times. Yet, we
            # have to re-initialize this based on the iteration, for we want
            # different splits there.
            cv = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=42 + iteration  # TODO: make configurable?
            )

            for fold_index, (train_index, test_index) in \
                    enumerate(cv.split(all_indices, y_opt)):

                logging.info(f'Processing fold {fold_index + 1}/{n_folds}...')

                # These indices do *not* change for individual kernels,
                # but we will overwrite them nonetheless.
                train_indices = all_indices[train_index]
                test_indices = all_indices[test_index]

                # Main function for training and testing a certain kernel
                # matrix on the data set.
                results = train_and_test(
                    train_indices, test_indices, matrix, classes,
                    args.max_iterations, benchmark_indices
                )

                # We already have information about the folds for this
                # particular iteration. This works because each kernel
                # is shown the *same* folds.
                if iteration in all_results['iterations'].keys():
                    pass

                # Store information about indices and labels. This has
                # to be done only for the first kernel matrix.
                else:
                    all_results['iterations'][iteration] = {
                        'folds': collections.defaultdict(dict)
                    }

                # Add fold information; this might overwrite one that is
                # already stored, but since all folds are the same, this
                # is not a problem.
                per_fold = all_results['iterations'][iteration]['folds']

                if args.with_indices:
                    per_fold[fold_index]['train_indices'
                                         ] = train_indices.tolist()
                    per_fold[fold_index]['test_indices'] = test_indices.tolist(
                    )

                per_fold[fold_index]['y_test'] = y[test_indices].tolist()

                # Prepare results for the current fold of the current
                # iteration. This will collect individual values, and
                # thus make it necessary to sum/collate over axes.
                #
                # We take whatever information has been supplied by the
                # function above.
                fold_results = {key: value for key, value in results.items()}

                # Check whether we are already storing information about
                # kernels.
                if 'kernels' not in per_fold[fold_index].keys():
                    per_fold[fold_index]['kernels'] = {}

                kernel_results = per_fold[fold_index]['kernels']

                # The results for this kernel on this particular fold
                # must not have been reported anywhere else.
                assert name not in kernel_results.keys()

                kernel_results[name] = {
                    key: value
                    for key, value in fold_results.items()
                }

    all_results['runtime'] = timer() - start_time
    all_results['max_iterations'] = args.max_iterations

    # The check for overwriting this data is only done once. If we have
    # arrived here, we might just as well write out our results.
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=4)
