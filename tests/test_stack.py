"""The scientific stack the project builds on is installed and works together."""

import networkx as nx
import numpy as np
from scipy import sparse

import imigraph


def test_package_imports():
    assert imigraph.__doc__


def test_graph_laplacian_rows_sum_to_zero():
    laplacian = nx.laplacian_matrix(nx.cycle_graph(5))
    assert sparse.issparse(laplacian)
    np.testing.assert_allclose(laplacian.sum(axis=1), 0)
