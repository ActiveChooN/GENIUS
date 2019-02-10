import numpy as np
from sklearn.cluster import KMeans
from multiprocessing import Pool
import time
from ._models import GENIE3E, GLasso, GENIE3E_single, wr_GENIE3E_single
from .vim import VIM


def combine_methods_avg(gene_expr_data, gene_names=None, regulators=None, parameters=None):
    """Computation of stack of algorithms  average score for all putative regulatory links.

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

    parameters: list of dict
        List containing dict with parameters that will be applied to models

    Returns
    -------

        Average matrix with connections between genes

    """

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

    if not isinstance(parameters, list):
        raise ValueError("parameters must be a list")
    if not len(parameters):
        raise ValueError("parameters must contain one dict with parameters at least")

    mat_sum = np.zeros([n_genes, n_genes])

    for param in parameters:
        if param['method'] is 'GL':
            param.__delattr__('method')
            vim = GLasso(gene_expr_data, gene_names, **param)
        else:
            vim = GENIE3E(gene_expr_data, gene_names, **param)
        mat_sum = np.sum([mat_sum, vim.get_probabilities_matrix()])

    vim = VIM(n_genes, gene_names)
    vim._mat = mat_sum / len(parameters)
    return vim


def combine_methods_weighted(gene_expr_data, gene_names=None, regulators=None, parameters=None, weights=None):
    """Computation of stack of algorithms weighted score for all putative regulatory links.

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

        parameters: list of dict
            List containing dict with parameters that will be applied to models

        weights: np.ndarray
            Array with float weights of liner combination of methods.

        Returns
        -------

            Weighted matrix with connections between genes

        """

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

    if not isinstance(parameters, list):
        raise ValueError("parameters must be a list")
    if not len(parameters):
        raise ValueError("parameters must contain one dict with parameters at least")

    if not isinstance(weights, np.ndarray):
        raise ValueError("Weights must by np.ndarray")
    if not len(parameters) == len(weights):
        raise ValueError("List of parameters and list of weights must have same size")

    mat_sum = np.zeros([n_genes, n_genes])

    weights = weights / np.sum(weights)

    for param, weight in zip(parameters, weights):
        if param['method'] is 'GL':
            param.__delattr__('method')
            vim = GLasso(gene_expr_data, gene_names, **param)
        else:
            vim = GENIE3E(gene_expr_data, gene_names, **param)
        mat_sum = np.sum([mat_sum, vim.get_probabilities_matrix() * weight])

    vim = VIM(n_genes, gene_names)
    vim._mat = mat_sum
    return vim


def compute_with_subsampling(gene_expr_data, gene_names=None, regulators=None, recomputing_strategy='threshold',
                             first_method='RF', second_method='ET', n_trees=1000, k='auto',  alpha=0.0001,
                             threshold=0.005, n_jobs=1, task_type='CPU', devices=None, verbose=False):
    """Computation of tree-based scores and linear regression score with regularization  and recomputing for target
    selected with specified threshold for all putative regulatory links.

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

        recomputing_strategy: 'threshold' or 'kmeans'
            Select the strategy of recomputing. If threshold is selected it will recompute connection between  genes
            only if they are stronger than threshold. Otherwise it will select strong connection with clustering
            algorithm.
            default: 'threshold'

        first_method: 'RF', 'ET', 'GB', 'XGB', 'CB', 'LS' or 'BR' optional
            Specifies which tree-based procedure is used: either Random Forest ('RF'), Extra-Trees ('ET'),
            Gradient-Boosting('GB'), XGBoost('XGB'), CatBoost('CB'), Linear Regression with Lasso regularization('LS') or
            BayesianRidge('BR'). It's used for the first step in computing process.
            default: 'RF'

        second_method: 'RF', 'ET', 'GB', 'XGB', 'CB', 'LS' or 'BR' optional
            Method is used for the second step in computing process. Same as first one.
            default: 'ET'

        threshold: positive float
            Threshold used for selecting regulators, which will be used for second computation step.

        k: 'sqrt', 'auto', 'log2' or a positive integer, optional
            Specifies the number of selected attributes at each node of one tree: either the square root of the number
            of candidate regulators ('sqrt'), the total number of candidate regulators ('all'), or any positive integer.
            default: 'sqrt'

        n_trees: positive integer, optional
            Specifies the number of trees grown in an ensemble.
            default: 1000

        alpha: positive float between 0. and 1.
            Alpha value for lasso regularization. Default set for best AUC_ROC
            default: 0.000001

        n_jobs: Positive integer
            Number of threads that will be started for tree learning. Using not with CatBoost.

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

    if first_method is not 'RF' and first_method is not 'ET' and first_method is not 'GB' and first_method is not 'XGB' \
            and first_method is not 'CB' and first_method is not 'LS' and first_method is not 'BR':
        raise ValueError('input argument method must be "RF" (Random Forests) or "ET" (Extra-Trees) '
                         'or "GB" (GradientBoosting)')

    if second_method is not 'RF' and second_method is not 'ET' and second_method is not 'GB' and second_method is \
            not 'XGB' and second_method is not 'CB' and second_method is not 'LS' and second_method is not 'BR':
        raise ValueError('input argument method must be "RF" (Random Forests) or "ET" (Extra-Trees) '
                         'or "GB" (GradientBoosting)')

    if k is not 'auto' and k is not 'sqrt' and k is not 'log2' and \
            (not isinstance(k, int) or isinstance(k, int) and k <= 0):
        raise ValueError('input argument k must be "auto", "sqrt", "log2"  or a strictly positive integer')

    if not isinstance(n_trees, int) or isinstance(n_trees, int) and n_trees <= 0:
        raise ValueError('input argument n_trees must be a strictly positive integer')

    if not isinstance(n_jobs, int) or isinstance(n_jobs, int) and n_jobs <= 0:
        raise ValueError('input argument n_jobs must be a strictly positive integer')

    if not isinstance(recomputing_strategy, str) and (recomputing_strategy is not 'threshold' or recomputing_strategy
                                                      is not 'kmeans'):
        raise ValueError('recomputing strategy must be "threshold" or "kmeans"')

    if verbose:
        print("Starting first step...")

    # compute vim for the first step
    vim = GENIE3E(gene_expr_data, gene_names, regulators, n_trees, k, first_method, alpha, n_jobs, task_type, devices,
                  verbose)

    if verbose:
        print("Starting second step...")

    input_params = []

    # recompute vim column for each target
    for idx, col in enumerate(vim.get_probabilities_matrix().T):
        if recomputing_strategy is 'threshold':
            strong_prob = [(i, col[i]) for i in range(col.shape[0]) if col[i] > threshold]
        else:
            km = KMeans(2, init=np.array([[0.], [1.]]), n_init=1)
            km.fit(col.reshape(-1, 1))
            predicted_classes = km.predict(col.reshape(-1, 1))
            strong_idx = [x for x, cl in enumerate(predicted_classes) if cl == 1]
            strong_prob = [(i, col[i]) for i in strong_idx]
        if strong_prob:
            input_params.append([gene_expr_data, idx, [x[0] for x in strong_prob], second_method, k, n_trees, alpha,
                                 task_type, devices, verbose])

    if n_jobs > 1:
        pool = Pool(n_jobs)
        pool_output = pool.map(wr_GENIE3E_single, input_params)
        pool.close()
        for (i, vi) in pool_output:
            vim._mat[:, i] = vi
    else:
        for param in input_params:
            vim._mat[:, param[1]] = GENIE3E_single(*param)

    if verbose:
        print("Total elapsed time: {:.2f} s.".format(time.time() - start_time))

    return vim


def compute_simple_aggregated_vim(vim_list, coefs=None):
    """Compute rank aggregated score of different VIM.

    Parameters
    ----------

    vim_list: list iof VIM
        list containing few vims to aggregate

    coefs: np.ndarray
        list with float coefficients of aggregation
        default: np.ones(len(vim_list))
    Returns
    -------
        aggregated vim
    """

    if not isinstance(vim_list, list):
        raise ValueError('vim_list must be a list of VIMs')

    if not vim_list:
        raise ValueError('vim_list must contain at least one element')

    for vim in vim_list:
        if not isinstance(vim, VIM):
            raise ValueError('All elements of vim_list must be a VIM object')

    if not (isinstance(coefs, np.ndarray) or coefs is None):
        raise ValueError('coefs must be list object')

    if coefs is not None and len(coefs) != len(vim_list):
        raise ValueError('vim_list and coefs must be arrays of the same size')

    # filling coefs array if none
    if coefs is None:
        coefs = np.ones(shape=len(vim_list))

    # computing vertex rank
    for vim in vim_list:
        for row_idx in range(vim._mat.shape[0]):
            temp = vim._mat[row_idx].argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(vim._mat.shape[1])
            vim._mat[row_idx] = ranks

    result_vim = VIM(vim_list[0]._mat.shape[0], vim_list[0].get_names())

    # filling output matrix
    for idx in range(vim_list[0]._mat.shape[0]):
        for vim_idx, vim in enumerate(vim_list):
            result_vim._mat[idx] += vim._mat[idx] * coefs[vim_idx   ]
        result_vim._mat[idx] /= len(vim_list)

    # normalizing output matrix
    for row in result_vim._mat:
        row /= np.sum(row)

    return result_vim
