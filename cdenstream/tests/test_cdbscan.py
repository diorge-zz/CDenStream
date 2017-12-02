import numpy as np
from ..cdbscan import cdbscan


def test_easy_clusters_no_constraints():
    points = np.array([[1, 1],
                       [52, 3],
                       [1, 2],
                       [2, 3],
                       [50, 4],
                       [51, 2]])
    epsilon = 5
    minpts = 2
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts)
    assert sorted(sorted(clusters)[0]) == [0, 2, 3]
    assert sorted(sorted(clusters)[1]) == [1, 4, 5]


def test_fully_constrained():
    points = np.array([[1, 1],
                       [52, 3],
                       [1, 2],
                       [2, 3],
                       [50, 4],
                       [51, 2]])
    epsilon = 5
    minpts = 2
    mustlink = np.array([[0, 1], [2, 3], [4, 5]])
    cannotlink = np.array([[0, 2], [1, 4], [3, 5]])
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts)
    assert sorted(sorted(clusters[0])) == [0, 1]
    assert sorted(sorted(clusters[1])) == [2, 3]
    assert sorted(sorted(clusters[1])) == [4, 5]
