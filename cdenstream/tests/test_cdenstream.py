"""Module for C-DenStream unit tests
"""


from collections import defaultdict
import numpy as np
from ..cdbscan import cdbscan
from ..cdenstream import CDenStream


def test_simple_stream():
    """Basic test for a short stream (1 update),
    no outliers and no concept drift
    """
    train_dataset = np.array([[0, 0],
                              [10, 11],
                              [12, 10],
                              [1, 0],
                              [2, -1],
                              [11, 12]])
    precluster = cdbscan(train_dataset, epsilon=3, minpts=2)
    points = defaultdict(list)
    for index, cluster in enumerate(precluster):
        points[cluster].append(train_dataset[index, :])

    denstream = CDenStream(ndim=2, mindist=3, minpts=2, outlier_radius=1)
    denstream.initialize(points.values())

    denstream.point_arrival([3, 0], 1)
    denstream.point_arrival([2, 1], 2)
    denstream.point_arrival([1, 1], 3)
    denstream.point_arrival([9, 10], 4)
    denstream.update(4)

    print(denstream.query(0.5))
