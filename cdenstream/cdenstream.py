"""CDenStream algiorithm
Based on the work "C-DenStream: Using Domain Knowledge on a Data Stream"
by Ruiz, Menasalvas and Spiliopoulou (2009)
"""


import numpy as np


class MicroCluster:
    """A potential core-micro-cluster or outlier-micro-cluster,
    according to the definition (w, CF1, CF2, t0).
    In the case of core-micro-cluster, the value of t0 is simply not used
    """

    def __init__(self, ndim=2, timestamp=0):
        self.weight = 0
        self.linear_dimensions = np.zeros((ndim,))
        self.squared_dimensions = np.zeros((ndim,))
        self.timestamp = timestamp
        self.buffer = []

    @property
    def center(self):
        """Returns the weighted centroid of the micro-cluster
        """
        return self.linear_dimensions / self.weight

    @property
    def radius(self):
        """Returns the weighted radius of the micro-cluster
        """
        mcf1 = np.linalg.norm(self.linear_dimensions)
        mcf2 = np.linalg.norm(self.squared_dimensions)
        return np.sqrt((mcf2 / self.weight) - (mcf1 / self.weight) ** 2)

    def merge(self, point):
        """Adds a point to the micro-cluster
        (but does not update yet)

        :param point: must be a numpy array with the same
        dimension passed into the constructor
        """
        self.buffer.append(point)

    def update(self, timeinterval, decay):
        """Updates the micro-cluster statistics
        using the merged points since the last update

        :param timeinterval: the time elapsed since last update
        :param decay: lambda value for decay function
        """
        if len(self.buffer) == 0:
            decayment = 2 ** (-decay * timeinterval)
            self.weight *= decayment
            self.linear_dimensions *= decayment
            self.squared_dimensions *= decayment
        else:
            for point in self.buffer:
                self.weight += 1
                self.linear_dimensions += point
                self.squared_dimensions += point ** 2
            self.buffer.clear()
