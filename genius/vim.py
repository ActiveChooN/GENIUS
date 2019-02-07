import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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
