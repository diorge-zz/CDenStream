"""CDBScan algorithm
Based on the work "Density-based semi-supervised clustering"
by Ruiz, Spiliopoulou and Menasalvas (2009)
"""
import numpy as np
from sklearn.neighbors import KDTree
from .constraint import cluster_respect_cannot_link_constraints, sanitize_constraints


def centroid(cluster_points):
    """Finds the centroid of the cluster
    (i.e. the mean of each dimension)
    """
    return np.mean(cluster_points)


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


def find_reachable_clusters(cluster_to_point, densityreachable, target_cluster, clusterkind='all'):
    """Finds the clusters that can be reached from another cluster
    """
    reachable_points = {densityreachable[p] for p in cluster_to_point[target_cluster]}
    reachable_points = {x for sublist in reachable_points for x in sublist}

    reachable_clusters_indexes = list()
    for cluster_index, cluster in cluster_to_point.items():
        if clusterkind == 'all' or clusterkind == cluster.kind:
            for point in cluster:
                if point in reachable_points:
                    reachable_clusters_indexes.append(cluster_index)
                    break

    return reachable_clusters_indexes


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


def create_local_clusters(hyperparam, state):
    """Step 2 of CDBScan
    For each data point, it is labeled as either noise or a local cluster.
    Local clusters may have a single or multiple points,
    depending or cannot-link constraints
    """
    point_to_cluster = state['point_to_cluster']
    cluster_to_point = state['cluster_to_point']
    next_cluster = state['next_cluster']

    for index, _ in enumerate(hyperparam['dataset']):
        # if point is yet unlabeled
        if point_to_cluster[index] == -1:
            pointdr = state['densityreachable'][index]
            if len(pointdr) < hyperparam['minpts']:
                # noise point
                pass
            elif not cluster_respect_cannot_link_constraints(pointdr, hyperparam['cannotlink']):
                for node in pointdr:
                    clusterpoints = (node,)
                    cluster_to_point[next_cluster] = Cluster('local', clusterpoints)
                    point_to_cluster[node] = next_cluster
                    next_cluster += 1
            else:
                # core point
                ldr = list(pointdr)
                clusterpoints = tuple(ldr)
                cluster_to_point[next_cluster] = Cluster('local', clusterpoints)
                point_to_cluster[ldr] = next_cluster
                next_cluster += 1

    state['next_cluster'] = next_cluster


def merge_clusters(cluster_to_point, point_to_cluster, cannotlink,
                   new_cluster_id, new_cluster_kind,
                   *clusters_to_merge):
    """Merges two or more clusters into a single new cluster
    containing all the points.
    Returns if the merge is allowed by cannot-link constraints
    (if it is not, the merge is not performed)

    :param clusters_to_merge: list of cluster points (not IDs)
    """
    clusters_to_merge = list(clusters_to_merge)
    all_points = set.union(*map(set, clusters_to_merge))

    if cluster_respect_cannot_link_constraints(all_points, cannotlink):
        new_cluster = Cluster(new_cluster_kind, tuple(all_points))
        cluster_to_point[new_cluster_id] = new_cluster

        for point in new_cluster:
            point_to_cluster[point] = new_cluster_id

        return True
    return False


def merge_mustlink_constraints(hyperparam, state):
    """Step 3a of CDBScan
    Enforces every must-link constraint that is not yet respected
    The clusters created by merging two of these local clusters
    are labeled as alpha clusters
    """
    point_to_cluster = state['point_to_cluster']
    cluster_to_point = state['cluster_to_point']
    next_cluster = state['next_cluster']

    for ml1, ml2 in hyperparam['mustlink']:
        cluster1 = point_to_cluster[ml1]
        cluster2 = point_to_cluster[ml2]

        if cluster1 == cluster2:
            continue

        if cluster1 != -1:
            points_of_c1 = cluster_to_point[cluster1]
            del cluster_to_point[cluster1]
        else:
            points_of_c1 = Cluster('noise', [ml1])

        if cluster2 != -1:
            points_of_c2 = cluster_to_point[cluster2]
            del cluster_to_point[cluster2]
        else:
            points_of_c2 = Cluster('noise', [ml2])

        merge_clusters(cluster_to_point, point_to_cluster,
                       hyperparam['cannotlink'], next_cluster,
                       'alpha', points_of_c1, points_of_c2)
        next_cluster += 1

    state['next_cluster'] = next_cluster


def merge_local_into_alpha(hyperparam, state):
    """Step 3b of CDBScan
    Find local clusters that can be merged into
    newly formed alpha clusters,
    then merge then into a new alpha cluster, iteratively
    """
    point_to_cluster = state['point_to_cluster']
    cluster_to_point = state['cluster_to_point']
    next_cluster = state['next_cluster']

    clusters_changed = True
    while clusters_changed:
        clusters_changed = False
        localclusters = {idx: cluster
                         for idx, cluster in cluster_to_point.items()
                         if cluster.kind == 'local'}

        for index_of_lc, localcluster in localclusters.items():
            reachable_alpha = find_reachable_clusters(
                cluster_to_point, state['densityreachable'], index_of_lc, 'alpha')
            if len(reachable_alpha) > 0:
                centroids_of_reachable_alpha = [centroid(cluster_to_point[i].points)
                                                for i in reachable_alpha]

                lc_centroid = centroid(cluster_to_point[index_of_lc].points)
                dist_to_reachable_alpha = [np.linalg.norm(lc_centroid - alpha_centroid)
                                           for alpha_centroid in centroids_of_reachable_alpha]

                closest_alpha = reachable_alpha[np.argmin(dist_to_reachable_alpha)]

                canmerge = merge_clusters(cluster_to_point, point_to_cluster,
                                          hyperparam['cannotlink'],
                                          next_cluster, 'alpha',
                                          localcluster,
                                          cluster_to_point[closest_alpha])
                if canmerge:
                    del cluster_to_point[index_of_lc]
                    del cluster_to_point[closest_alpha]
                    next_cluster += 1
                    clusters_changed = True
                    break


def cdbscan(dataset, epsilon=0.01, minpts=5, mustlink=None, cannotlink=None):
    """Implementation of the CDBScan algorithm
    """
    cluster_to_point = {}

    mustlink, cannotlink = sanitize_constraints(X=dataset,
                                                must_link=mustlink,
                                                cannot_link=cannotlink)

    # clusters: -1 for unclustered, otherwise the id of the cluster
    # use clusters[point_id] to find out which cluster a point belongs to
    point_to_cluster = np.empty((dataset.shape[0],), dtype=np.int)
    point_to_cluster.fill(-1)

    hyperparams = {'dataset': dataset,
                   'epsilon': epsilon,
                   'minpts': minpts,
                   'mustlink': mustlink,
                   'cannotlink': cannotlink}
    state = {'point_to_cluster': point_to_cluster,
             'cluster_to_point': cluster_to_point,
             'next_cluster': 0}

    # effectively the Step 1 of the algorithm's pseudocode
    state['densityreachable'] = find_density_reachable_points(dataset, epsilon)
    # Step 2 of the algorithm
    create_local_clusters(hyperparams, state)
    # Step 3a
    merge_mustlink_constraints(hyperparams, state)
    # Step 3b - Build the final clusters
    merge_local_into_alpha(hyperparams, state)

    return point_to_cluster