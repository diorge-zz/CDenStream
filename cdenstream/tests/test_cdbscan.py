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
    assert sorted(clusters) == [(0, 2, 3), (1, 4, 5)]


def test_fully_constrained():
    points = np.array([[1, 1],
                       [52, 3],
                       [1, 2],
                       [2, 3],
                       [50, 4],
                       [51, 2]])
    epsilon = 5
    minpts = 2
    mustlink = set([(0, 1), (2, 3), (4, 5)])
    cannotlink = set([(0, 2), (1, 4), (3, 5)])
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts,
                       mustlink=mustlink, cannotlink=cannotlink)
    assert sorted(clusters) == [(0, 1), (2, 3), (4, 5)]


def test_fully_must_constrained():
    points = np.array([[1, 1],
                       [52, 3],
                       [1, 2],
                       [50, 4],
                       [2, 3],
                       [51, 2]])
    epsilon = 5
    minpts = 2
    mustlink = set([(0, 1), (2, 3), (3, 4)])
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts,
                       mustlink=mustlink)
    assert sorted(clusters) == [(0, 1, 2, 3, 4, 5)]


def test_fully_cannot_constrained():
    points = np.array([[1, 1],
                       [52, 3],
                       [1, 2],
                       [50, 4],
                       [2, 3],
                       [51, 2]])
    epsilon = 5
    minpts = 2
    cannotlink = set([(0, 2), (2, 4), (1, 5), (3, 5)])
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts,
                       cannotlink=cannotlink)
    assert sorted(clusters) == [(0,), (1,), (2,), (3,), (4,), (5,)]


def test_mustlink_merging():
    points = np.array([[1, 1],
                       [52, 3],
                       [1, 2],
                       [50, 4],
                       [2, 3],
                       [51, 2]])
    epsilon = 5
    minpts = 2
    mustlink = set([(0, 1)])
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts,
                       mustlink=mustlink)
    assert sorted(clusters) == [(0, 1, 2, 3, 4, 5)]


def test_singleton_outlier():
    points = np.array([[1, 1],
                       [52, 3],
                       [1, 2],
                       [50, 4],
                       [2, 3],
                       [51, 2],
                       [100, 200]])
    epsilon = 5
    minpts = 2
    mustlink = set([(0, 1)])
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts,
                       mustlink=mustlink)
    assert sorted(clusters) == [(0, 1, 2, 3, 4, 5)]


def test_outlier_cluster():
    points = np.array([[1, 1],
                       [52, 3],
                       [1, 2],
                       [50, 4],
                       [2, 3],
                       [51, 2],
                       [100, 200],
                       [100, 201]])
    epsilon = 5
    minpts = 3
    mustlink = set([(0, 1)])
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts,
                       mustlink=mustlink)
    assert sorted(clusters) == [(0, 1, 2, 3, 4, 5)]


def test_connection_through_alpha():
    points = np.array([[1, 1], [2, 2], [2, 1], # cluster 1
                       [4, 5], [5, 4], [5, 5], # cluster 2
                       [9, 1], [8, 0], [13, 2], # cluster 3
                       [3, 3], [7, 3]]) # bridges between 1-2 and 2-3
    epsilon = 4
    minpts = 2
    mustlink = set([(7, 8)])
    clusters = cdbscan(points, epsilon=epsilon, minpts=minpts,
                       mustlink=mustlink)
    # assert is becomes a single cluster because the bridges merge the clusters
    assert sorted(clusters) == [tuple(range(len(points)))]
