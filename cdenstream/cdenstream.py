"""CDenStream algiorithm
Based on the work "C-DenStream: Using Domain Knowledge on a Data Stream"
by Ruiz, Menasalvas and Spiliopoulou (2009)
"""


from collections import namedtuple
import numpy as np


class MicroCluster:
    """A potential core-micro-cluster or outlier-micro-cluster,
    according to the definition (w, CF1, CF2, t0).
    In the case of core-micro-cluster, the value of t0 is simply not used
    """

    def __init__(self, kind, ndim=2, timestamp=0):
        self.kind = kind
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

    def copy(self):
        """Makes a deep copy of the object
        """
        new = MicroCluster(self.kind, self.ndim, self.timestamp)
        new.weight = self.weight
        new.linear_dimensions = np.copy(self.linear_dimensions)
        new.squared_dimensions = np.copy(self.squared_dimensions)
        new.buffer = [np.copy(pt) for pt in self.buffer]
        


Constraint = namedtuple('Constraint', ['kind', 'weight'])

class ConstraintMap:
    """Represents the CO-MC matrix
    """
    def __init__(self):
        self._constraints = {}

    def __getitem__(self, microclusterpair):
        pair = tuple(sorted(microclusterpair))
        return self._constraints[pair]

    def __contains__(self, microclusterpair):
        pair = tuple(sorted(microclusterpair))
        return pair in self._constraints

    def _set(self, microclusterpair, constraint):
        pair = tuple(sorted(microclusterpair))
        self._constraints[pair] = constraint

    def merge_constraint(self, mc1, mc2, kind, timestamp):
        """Merges a constraint from the stream into the current mapping
        The micro-clusters must be immutable, hashable, equatable IDs
        Kind must be either 'mustlink' or 'cannotlink'
        """

        if (mc1, mc2) not in self:
            constraint = Constraint(kind=kind, weight=timestamp)
            self._set((mc1, mc2), constraint)
        else:
            constraint = self[mc1, mc2]
            if constraint.kind == kind:
                constraint.weight += 1
            else:
                constraint.kind = kind
                constraint.weight = timestamp

    def update(self, timeinterval, decay):
        """Updates the micro-cluster statistics,
        similar to MicroCluster.update
        """
        decayment = 2 ** (-decay * timeinterval)
        for constraint in self._constraints.values():
            constraint.weight *= decayment


class CDenStream:
    """State and operations for the C-DenStream algorithm
    """
    def __init__(self, ndim=2, mindist=0.5, minpts=5,
                 outlier_radius=7, decay_rate=0.01):
        self.constraints = ConstraintMap()
        self.microclusters = {}
        self.nextmicrocluster = 0
        self.ndim = ndim
        self.mindist = mindist
        self.minpts = minpts
        self.outlier_radius = outlier_radius
        self.decay_rate = decay_rate
        self.timestamp = 0

    def initialize(self, microclusters, mustlink=None, cannotlink=None):
        """Initializes with the result of a clustering algorithm,
        assuming timestamp zero
        """
        for cluster in microclusters:
            newmc = MicroCluster('unknown', self.ndim)
            for point in cluster:
                newmc.merge(point)
            newmc.update(0, 0)
            if newmc.weight > self.minpts * self.outlier_radius:
                newmc.kind = 'core'
            else:
                newmc.kind = 'outlier'
            self.microclusters[self.nextmicrocluster] = newmc
            self.nextmicrocluster += 1

        if mustlink is not None:
            for pt1, pt2 in mustlink:
                mc1 = self._get_closest_microcluster(pt1)
                mc2 = self._get_closest_microcluster(pt2)
                self.constraints.merge_constraint(mc1, mc2, 'mustlink', 0)

        if cannotlink is not None:
            for pt1, pt2 in cannotlink:
                mc1 = self._get_closest_microcluster(pt1)
                mc2 = self._get_closest_microcluster(pt2)
                self.constraints.merge_constraint(mc1, mc2, 'cannotlink', 0)

    def _get_closest_microcluster(self, point, kind='any'):
        """Finds the micro-cluster that is closest to the point parameter
        """
        distances = [(k, np.linalg.norm(point - v.center))
                     for k, v in self.microclusters.items()]
        sorted_ids = sorted(distances, key=(lambda x: x[1]))
        for mc_id in sorted_ids:
            if 'kind' == 'any' or self.microclusters[mc_id].kind == kind:
                return mc_id

        raise ValueError('No clusters of wanted kind')

    def point_arrival(self, point, timestamp):
        """Merges a new point from the stream into the micro-clusters
        """
        timeinterval = timestamp - self.timestamp

        microcluster = self._get_closest_microcluster(point, 'core')
        clone = microcluster.copy()
        clone.merge(point)
        clone.update(timeinterval, self.decay_rate)
        if clone.radius <= self.mindist:
            microcluster.merge(point)
        else:
            microcluster = self._get_closest_microcluster(point, 'outlier')
            clone = microcluster.copy()
            clone.merge(point)
            clone.update(timeinterval, self.decay_rate)
            if clone.radius <= self.mindist:
                microcluster.merge(point)
                microcluster.update()
                if microcluster.weight > self.outlier_radius * self.minpts:
                    microcluster.kind = 'core'
            else:
                newmc = MicroCluster('outlier', self.ndim, timestamp)
                newmc.merge(point)
                self.microclusters[self.nextmicrocluster] = newmc
                self.nextmicrocluster += 1

    def constraint_arrival(self, kind, point1, point2, timestamp):
        """Merges a new constraint from the stream into the constraint map
        """
        mc1 = self._get_closest_microcluster(point1)
        mc2 = self._get_closest_microcluster(point2)
        self.constraints.merge_constraint(mc1, mc2, kind, timestamp)

    def update(self, timestamp):
        """Updates the weights of micro-clusters and constraints
        to keep the stored information fresh
        """
        timeinterval = timestamp - self.timestamp
        for microcluster in self.microclusters.values():
            microcluster.update(timeinterval, self.decay_rate)
        self.constraints.update(timeinterval, self.decay_rate)
        self.timestamp = timestamp
