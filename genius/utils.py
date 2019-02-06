import numpy as np
import xml.dom.minidom as md


def xml_graph_to_adjacency_matrix(filename):
    dom = md.parse(filename)

    nodes = dom.getElementsByTagName("Node")
    ids = [int(a.getAttribute('id')) for a in nodes]
    adjacency_matrix = np.zeros(shape=(len(ids), len(ids)))
    edges = dom.getElementsByTagName("Edge")
    for e in edges:
        source = int(e.getElementsByTagName('from')[0].firstChild.nodeValue)
        target = int(e.getElementsByTagName('to')[0].firstChild.nodeValue)
        adjacency_matrix[source][target] = 1
        adjacency_matrix[target][source] = 1
    return adjacency_matrix


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
