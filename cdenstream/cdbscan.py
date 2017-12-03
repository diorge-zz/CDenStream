"""CDBScan algorithm
Based on the work "Density-based semi-supervised clustering"
by Ruiz, Spiliopoulou and Menasalvas (2009)
"""
import numpy as np
from sklearn.neighbors import KDTree
from .constraint import cluster_respect_cannot_link_constraints


def find_density_reachable_points(dataset, maximum_distance):
    """Creates the density-reachable matrix of the dataset
    The return is a dict that maps each point index (zero-based)
    to a tuple of indices for the points in its neighborhood
    """
    element_count = dataset.shape[0]
    kdtree = KDTree(dataset, metric="euclidean")
    neighborhoods = kdtree.query_radius(X=dataset, r=maximum_distance)
    density_reachable = dict()
    for element_index in range(element_count):
        density_reachable[element_index] = tuple(neighborhoods[element_index])

    return density_reachable


class Cluster:
    """Represents a cluster object
    Clusters have a kind (noise, alpha or local)
    and a collection of points
    """
    def __init__(self, kind, points=None):
        self.kind = kind
        if points is None:
            self.points = tuple()
        else:
            self.points = points

    def __iter__(self):
        return iter(self.points)

    def __repr__(self):
        return repr(self.points)


def cdbscan(dataset, epsilon=0.01, minpts=5, mustlink=None, cannotlink=None):
    """Implementation of the CDBScan algorithm
    """
    allclusters = {}

    if mustlink is None:
        mustlink = set()

    if cannotlink is None:
        cannotlink = set()

    # clusters: -1 for unclustered, otherwise the id of the cluster
    # use clusters[point_id] to find out which cluster a point belongs to
    clusters = np.empty((dataset.shape[0],), dtype=np.int)
    clusters.fill(-1)
    nextcluster = 0

    densityreachable = find_density_reachable_points(dataset, epsilon)

    for index, point in enumerate(dataset):
        # if point is yet unlabeled
        if clusters[index] == -1:
            pointdr = densityreachable[index]
            if len(pointdr) < minpts:
                # noise point
                pass
            elif not cluster_respect_cannot_link_constraints(pointdr, cannotlink):
                for node in pointdr:
                    clusterpoints = (node,)
                    allclusters[nextcluster] = Cluster('local', clusterpoints)
                    clusters[node] = nextcluster
                    nextcluster += 1
            else:
                # core point
                ldr = list(pointdr)
                clusterpoints = tuple(ldr)
                allclusters[nextcluster] = Cluster('local', clusterpoints)
                clusters[ldr] = nextcluster
                nextcluster += 1

    # Step 3a: merge must-link constraints
    for ml1, ml2 in mustlink:
        cluster1 = clusters[ml1]
        cluster2 = clusters[ml2]
        if cluster1 == cluster2:
            continue
        if cluster1 != -1:
            points_of_c1 = allclusters[cluster1]
            del allclusters[cluster1]
        else:
            points_of_c1 = Cluster('noise', [ml1])
        if cluster2 != -1:
            points_of_c2 = allclusters[cluster2]
            del allclusters[cluster2]
        else:
            points_of_c2 = Cluster('noise', [ml2])
        merged = Cluster('alpha', tuple(set(points_of_c1).union(set(points_of_c2))))
        allclusters[nextcluster] = merged
        for point in merged:
            clusters[point] = nextcluster
        nextcluster += 1


    # Step 3b - Build the final clusters
    def compute_cluster_centroid(cluster):
        points = allclusters[cluster].points
        return np.mean(points)

    def compute_reachable_clusters(target_cluster, clusterkind='all'):
        reachable_points = {densityreachable[p] for p in allclusters[target_cluster]}
        reachable_points = {x for sublist in reachable_points for x in sublist}

        reachable_clusters_indexes = list()
        for cluster_index, cluster in allclusters.items():
            if clusterkind == 'all' or clusterkind == cluster.kind:
                for point in cluster.points:
                    if point in reachable_points:
                        reachable_clusters_indexes.append(cluster_index)
                        break

        return reachable_clusters_indexes

    clusters_changed = True
    while clusters_changed:
        clusters_changed = False
        localclusters = {idx: cluster
                         for idx, cluster in allclusters.items()
                         if cluster.kind == 'local'}

        for index_of_lc, localcluster in localclusters.items():
            elements_of_lc = localcluster.points
            reachable_alpha = compute_reachable_clusters(index_of_lc, 'alpha')
            if len(reachable_alpha) > 0:
                centroids_of_reachable_alpha = [compute_cluster_centroid(i)
                                                for i in reachable_alpha]

                lc_centroid = compute_cluster_centroid(index_of_lc)
                dist_to_reachable_alpha = [np.linalg.norm(lc_centroid - alpha_centroid)
                                           for alpha_centroid in centroids_of_reachable_alpha]

                closest_alpha = reachable_alpha[np.argmin(dist_to_reachable_alpha)]

                merged_cluster = set(elements_of_lc)
                merged_cluster.update(allclusters[closest_alpha])
                if cluster_respect_cannot_link_constraints(merged_cluster, cannotlink):
                    del allclusters[index_of_lc]
                    del allclusters[closest_alpha]
                    allclusters[nextcluster] = Cluster('alpha', tuple(merged_cluster))
                    for point in merged_cluster:
                        clusters[point] = nextcluster
                    nextcluster += 1
                    clusters_changed = True
                    break

    return [x.points for x in allclusters.values()]
