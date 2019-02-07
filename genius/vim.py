import os
import csv
import warnings
import numpy as np
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import networkx as nx
from .utils import matrix_stringify


def _make_label_dict(labels, idx):
    l = {}
    for i, label in zip(idx, labels):
        l[i] = label
    return l


class VIM(object):
    def __init__(self, size, names):
        # TODO: add doc
        self._size = size
        self._mat = np.zeros([self._size, self._size])
        self._mat_stand = np.zeros([self._size, self._size])
        self._mat_bin = np.zeros([self._size, self._size])
        self._mat_deg = ([[] for _ in range(self._size)])
        self._names = names

    def save_csv(self, filename):
        with open(os.path.join(os.getcwd(), filename), 'w') as f:
            wr = csv.writer(f)
            wr.writerow([""] + self._names)
            for i in range(self._size):
                wr.writerow([self._names[i]] + self._mat[i].tolist())

    def compute_vertexes_from_probabilities(self, sigma=1.67):
        self._mat_stand = (self._mat - np.average(self._mat, axis=0)) / np.std(self._mat)
        self._mat_bin = np.vectorize(lambda x: 1. if x > sigma else 0.)(self._mat_stand)
        return self._mat_bin

    def plot(self):
        if not np.any(self._mat_bin):
            self.compute_vertexes_from_probabilities()
        rows, cols = np.where(self._mat_bin == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.draw(gr, labels=_make_label_dict(self._names, rows), with_labels=True)
        plt.show()

    def get_auc_roc_score(self, true_data):
        if isinstance(true_data, VIM):
            true_data = true_data._mat
        fpr, tpr, thresholds = roc_curve(matrix_stringify(true_data), matrix_stringify(self._mat))
        return auc(fpr, tpr)

    def compute_vertexes_degree(self):
        if not np.any(self._mat_bin):
            self.compute_vertexes_from_probabilities()
        for i in range(self._size):
            deg = 0
            for j in range(self._size):
                if self._mat_bin[i][j] == 1:
                    deg += 1
            self._mat_deg[deg] += [i]
        return self._mat_deg

    def get_probabilities_matrix(self):
        return self._mat

    @staticmethod
    def load_from_file(filename, sep=','):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            gene_names = f.readline().rstrip('\n').split(sep)
            gene_data = np.array([list(map(float, x.rstrip('\n').split(sep))) for x in f.readlines()])
        vim = VIM(gene_data.shape[0], gene_names)
        vim._mat = gene_data
        return vim

    def get_auc_score_by_vertex_degree(self, true_data):
        if isinstance(true_data, VIM):
            true_data = true_data._mat

        if not np.any(self._mat_deg):
            self.compute_vertexes_degree()
        aucs_d = np.zeros(self._size)
        for i, deg in enumerate(self._mat_deg):
            if len(deg):
                for j, idx in enumerate(deg):
                    true_arr = true_data[idx]
                    arr = self._mat[idx]
                    try:
                        fpr, tpr, thresholds = roc_curve(true_arr, arr)
                        roc_auc = auc(fpr, tpr)
                    except:
                        roc_auc = 0.
                        warnings.warn("Failed to compute roc_auc")
                    aucs_d[j] = roc_auc if not np.isnan(roc_auc) else 0.
        return aucs_d