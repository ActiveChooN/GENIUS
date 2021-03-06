from sklearn.tree.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import numpy as np
from multiprocessing import Pool
import time
from importlib.util import find_spec
from .vim import VIM

# Trying import some optional libs
catboost_spec = find_spec('catboost')
if catboost_spec is not None:
    catboost = catboost_spec.loader.load_module()

xgboost_spec = find_spec('xgboost')
if xgboost_spec is not None:
    xgboost = xgboost_spec.loader.load_module()


def load_expr_from_file(filename, sep='\t'):
    """Load expression experiments data from file

    Parameters
    ----------

    filename: str
        path to file with expression data.

    sep: str, optional
        separator used for in file.

    Returns
    -------

        Two arrays, first is array with gene names, second is two-dimensional matrix with expression experiments data.
    """

    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")
    if not isinstance(sep, str):
        raise ValueError("Separator must be a string.")

    with open(filename) as f:
        gene_names = np.array(f.readline().rstrip('\n').split(sep))
        gene_data = np.array([list(map(float, x.rstrip('\n').split(sep))) for x in f.readlines()])
    
    return gene_names, gene_data


def GENIE3(gene_expr_data, gene_names=None, regulators=None, n_trees=1000, k='auto', tree_method='RF', n_jobs=1,
           task_type='CPU', devices=None, verbose=False):
    """'Computation of tree-based scores for all putative regulatory links.

    Parameters
    ----------

    gene_expr_data: numpy array
        Array containing gene expression values. Each row corresponds to a condition and each column corresponds to
        a gene.

    gene_names: list of strings, optional
        List of length p, where p is the number of columns in expr_data, containing the names of the genes. The i-th
        item of gene_names must correspond to the i-th column of expr_data.
        default: None

    regulators: list of strings, optional
        List containing the names of the candidate regulators. When a list of regulators is provided, the names of all
        the genes must be provided (in gene_names). When regulators is set to 'all', any gene can be a candidate
        regulator.
        default: 'all'

    tree_method: 'RF', 'ET', 'GB', 'XGB' or 'CB', optional
        Specifies which tree-based procedure is used: either Random Forest ('RF'), Extra-Trees ('ET'),
        Gradient-Boosting('GB'), XGBoost('XGB') or CatBoost('CB')
        default: 'RF'

    k: 'sqrt', 'auto', 'log2' or a positive integer, optional
        Specifies the number of selected attributes at each node of one tree: either the square root of the number
        of candidate regulators ('sqrt'), the total number of candidate regulators ('all'), or any positive integer.
        default: 'sqrt'

    n_trees: positive integer, optional
        Specifies the number of trees grown in an ensemble.
        default: 1000

    n_jobs: Positive integer
        Number of threads that will be started for tree learning

    task_type: 'CPU' or 'GPU'
        Select computing device. Used only with CatBoost estimator.

    devices: str of positive integer separated with ':' or '-'
        IDs of computing devices. Used only with CatBoost estimator.

    verbose: Bool
        If True it's log some information in standard output
        default: False

    Returns
    -------

    An array in which the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene.
    All diagonal elements are set to zero (auto-regulations are not considered). When a list of candidate regulators
    is provided, the scores of all the edges directed from a gene that is not a candidate regulator are set to zero.

    """

    start_time = time.time()

    # Check input parameters
    if not isinstance(gene_expr_data, np.ndarray):
        raise ValueError('gene_expr_data must be an array in which each row corresponds to a condition/sample '
                         'and each column corresponds to a gene')

    n_genes = gene_expr_data.shape[1]

    if gene_names is not None:
        if not isinstance(gene_names, (list, tuple)):
            raise ValueError('input argument gene_names must be a list of gene names')
        elif len(gene_names) != n_genes:
            raise ValueError('input argument gene_names must be a list of length p, where p is the number '
                             'of columns/genes in the expr_data')

    if regulators is not None:
        if not isinstance(regulators, (list, tuple)):
            raise ValueError('input argument regulators must be a list of gene names')

        if gene_names is None:
            raise ValueError('the gene names must be specified (in input argument gene_names)')
        else:
            s_intersection = set(gene_names).intersection(set(regulators))
            if not s_intersection:
                raise ValueError('the genes must contain at least one candidate regulator')

    if tree_method is not 'RF' and tree_method is not 'ET' and tree_method is not 'GB' and tree_method is not 'XGB' \
            and tree_method is not 'CB':
        raise ValueError('input argument tree_method must be "RF" (Random Forests) or "ET" (Extra-Trees) '
                         'or "GB" (GradientBoosting)')

    if tree_method is 'XGB' and xgboost_spec is None:
        raise ImportError('failed to import XGBoost, check if it is installed')

    if tree_method is 'CB' and catboost_spec is None:
        raise ImportError('failed to import CatBoost, check if it is installed')

    if k is not 'auto' and k is not 'sqrt' and k is not 'log2' and \
            (not isinstance(k, int) or isinstance(k, int) and k <= 0):
        raise ValueError('input argument k must be "auto", "sqrt", "log2"  or a strictly positive integer')

    if not isinstance(n_trees, int) or isinstance(n_trees, int) and n_trees <= 0:
        raise ValueError('input argument n_trees must be a strictly positive integer')

    if not isinstance(n_jobs, int) or isinstance(n_jobs, int) and n_jobs <= 0:
        raise ValueError('input argument n_jobs must be a strictly positive integer')

    if verbose:
        print("\nStarting to train model ({}) with {} trees and {} jobs".format(tree_method, n_trees, n_jobs))

    if regulators is None:
        input_genes_idx = list(range(n_genes))
    else:
        input_genes_idx = [i for i, name in enumerate(gene_names) if name in regulators]

    vim = VIM(n_genes, gene_names)

    pool_input_data = [[gene_expr_data, i, input_genes_idx, tree_method, k, n_trees, task_type, devices, verbose]
                       for i in range(n_genes)]
    if n_jobs > 1:
        pool = Pool(n_jobs)
        pool_output = pool.map(wr_GENIE3_single, pool_input_data)
        for (i, vi) in pool_output:
            vim._mat[i, :] = vi
    else:
        for i in range(n_genes):
            vi = GENIE3_single(gene_expr_data, i, input_genes_idx, tree_method, k, n_trees, task_type, devices, verbose)
            vim._mat[i, :] = vi

    vim._mat = np.transpose(vim._mat)

    if verbose:
        print("Elapsed time: {:.2f} s.".format(time.time() - start_time))

    return vim


def wr_GENIE3_single(args):
    return [args[1], GENIE3_single(*args)]


def GENIE3_single(gene_expr_data, output_idx, input_idx, tree_method, k, n_trees, task_type, devices, verbose):
    if verbose:
        print("Computing gene {}/{}...".format(output_idx + 1, gene_expr_data.shape[1]))

    # Expression of target gene
    gene_expr_data_output = gene_expr_data[:, output_idx]

    # Normalize output data
    gene_expr_data_output = (gene_expr_data_output / np.std(gene_expr_data_output)) if np.std(gene_expr_data_output) \
        else gene_expr_data_output

    # Remove target gene from candidate regulators
    input_idx = input_idx[:]
    if output_idx in input_idx:
        input_idx.remove(output_idx)

    # Take slice for input data
    gene_expr_data_input = gene_expr_data[:, input_idx]

    if tree_method == 'RF':
        estimator = RandomForestRegressor(n_estimators=n_trees, max_features=k)
    elif tree_method == 'ET':
        estimator = ExtraTreesRegressor(n_estimators=n_trees, max_features=k)
    elif tree_method == 'GB':
        estimator = GradientBoostingRegressor(n_estimators=n_trees, max_features=k)
    elif tree_method == 'CB':
        estimator = catboost.CatBoostRegressor(n_estimators=n_trees, silent=True, task_type=task_type, devices=devices)
    elif tree_method == 'XGB':
        estimator = xgboost.XGBRegressor(n_estimators=n_trees, silent=True)

    # Train tree ensemble
    estimator.fit(gene_expr_data_input, gene_expr_data_output)

    # Compute importance scores
    feature_importances = compute_feature_importances(estimator)
    vi = np.zeros(gene_expr_data.shape[1])
    vi[input_idx] = feature_importances

    return vi


def compute_feature_importances(estimator):
    if isinstance(estimator, BaseDecisionTree):
        return estimator.tree_.compute_feature_importances(normalize=False)
    elif isinstance(estimator, GradientBoostingRegressor):
        return estimator.feature_importances_
    elif catboost_spec is not None and isinstance(estimator, catboost.CatBoostRegressor):
        return estimator.feature_importances_ / np.sum(estimator.feature_importances_)
    elif xgboost_spec is not None and isinstance(estimator, xgboost.XGBRegressor):
        return estimator.feature_importances_
    else:
        importance_arr = np.asarray([e.tree_.compute_feature_importances(normalize=False)
                       for e in estimator.estimators_])
        return np.sum(importance_arr, axis=0) / len(estimator)
