from sklearn.linear_model import Lasso
from .vim import VIM
import numpy as np
from multiprocessing import Pool
import time


def GENLasso(gene_expr_data, gene_names, regulators=None, alpha=0.0001, n_jobs=1, verbose=False):
    """Computation of scores for all putative regulatory links based on linear regression with Lasso regularization.

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

        alpha: float
            hyperparameter for linear regression with Lasso regularization
            default: 1.0

        n_jobs: Positive integer
            Number of threads that will be started for tree learning

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

    if verbose:
        print("\nStarting to train Lasso model with alpha={} and {} jobs".format(alpha, n_jobs))

    if regulators is None:
        input_genes_idx = list(range(n_genes))
    else:
        input_genes_idx = [i for i, name in enumerate(gene_names) if name in regulators]

    vim = VIM(n_genes, gene_names)

    if n_jobs > 1:
        pool_input_data = [[gene_expr_data, i, input_genes_idx, verbose]
                           for i in range(n_genes)]
        pool = Pool(n_jobs)
        pool_output = pool.map(wr_GENLasso_single, pool_input_data)
        for (i, vi) in pool_output:
            vim._mat[i, :] = vi
    else:
        for i in range(n_genes):
            vi = GENLasso_single(gene_expr_data, i, input_genes_idx, alpha, verbose)
            vim._mat[i, :] = vi

    vim._mat = np.transpose(vim._mat)

    if verbose:
        print("Elapsed time: {:.2f} s.".format(time.time() - start_time))

    return vim


def wr_GENLasso_single(args):
    return [args[1], GENLasso_single(*args)]


def GENLasso_single(gene_expr_data, output_idx, input_idx, alpha, verbose):
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

    # Create and train model
    estimator = Lasso(alpha)
    estimator.fit(gene_expr_data_input, gene_expr_data_output)

    # Compute normalized coefs
    vi = np.zeros(gene_expr_data.shape[1])
    vi[input_idx] = estimator.coef_ / np.sum(np.abs(estimator.coef_)) \
        if np.sum(np.abs(estimator.coef_)) else estimator.coef_

    return vi
