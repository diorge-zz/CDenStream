from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KDTree
from .constraint import *
import numpy as np

NEIGHBORHOOD_MIN_POINTS = 50
NEIGHBORHOOD_RADIUS = 2


def compute_density_reachable_points(dataset, maximum_distance):
    element_count = dataset.shape[0]
    kdtree = KDTree(dataset, metric="euclidean")
    neighborhoods = kdtree.query_radius(X=dataset, r=maximum_distance)
    density_reachable = dict()
    for element_index in range(element_count):
        density_reachable[element_index] = tuple(neighborhoods[element_index])

    return density_reachable


def compute_density_connectable_points(distances, point_index, maximum_distance):
    """
    distances should be a sklearn.metrics.pairwise.pairwise_distances matrix
    """

    def compute_neighbors(element_index):
        return tuple(i for i in range(element_count)
                     if distances[element_index, i] <= maximum_distance)

    element_count = distances.shape[0]

    # Storing points already visited to prevent infinite loops
    points_already_visited = set()

    # The initial set of density_reachable points is the neighborhood of
    # the point
    density_connectable = set(compute_neighbors(point_index))

    new_points_to_explore = True
    while new_points_to_explore:
        old_reachable_point_count = len(density_connectable)
        reachable_neighborhoods = [compute_neighbors(i) for i in density_connectable
                                   if i not in points_already_visited]
        points_already_visited.update(density_connectable)
        for neighborhood in reachable_neighborhoods:
            density_connectable.update(neighborhood)
        new_points_to_explore = old_reachable_point_count < len(density_connectable)

    return density_connectable


class Payload:
    """Tuple emulation that carry the index of the point
    """

    def __init__(self, point, index):
        self.point = point
        self.index = index

    def __len__(self):
        return len(self.point)

    def __getitem__(self, i):
        return self.point[i]

    def __repr__(self):
        return f'{self.index}:{self.point}'


def cdbscan(dataset, epsilon=0.01, minpts=5, mustlink=None, cannotlink=None):
    """
    Step 1 -> Partition the data space with a KD-Tree
    kdtree := BuildKDTree(D)

    Step 2 -> Create local clustersin the KD-Tree
    for each leaf node v in kdtree and each unlabeled point pi in v do
        DR(pi) := all points density-reachable from pi within eps
        if |DR(pi)| < MinPts then
            Label pi as NOISE_POINT
        else if exists a constraint CL among points in DR(pi) then
            Each point in DR(pi) and pi becomes one LOCAL_CLUSTER
        else
            Label pi as CORE_POINT
            All of {pi} U DR(pi) becomes one LOCAL_CLUSTER
        end
    end

    Step 3a -> Merge clusters and enforce the Must-Link constraints
    for each constraint in ML do
        Let p, p' be the data points in the constraint
        Find the clusters C, C' with p in C and p' in C'
        Merge C, C' into cluster Cnew := C U C' and label it as ALPHA_CLUSTER
    end

    Step 3b -> Build the final clusters
    while number of local clusters decreases do
        for each local cluster C do
            Let C' be the closest ALPHA_CLUSTER that is density-reachable from C
            if exists no constraint in CL between points of C, C' then
                Incorporate C into C', i.e. C' := C U C'
            end
        end
    end
    return each ALPHA_CLUSTER and each remaining LOCAL_CLUSTER
    """
    localclusters = []
    alphaclusters = []

    # 0 = unlabeled, 1 = core, -1 = noise
    labels = np.zeros((dataset.shape[0],), dtype=np.int)

    # clusters: -1 for unclustered, otherwise the id of the cluster
    clusters = np.empty((dataset.shape[0],), dtype=np.int).fill(-1)
    nextcluster = 0

    densityreachable = compute_density_reachable_points(dataset, epsilon)

    for index, point in enumerate(dataset):
        # if point is yet unlabeled
        if labels[index] == 0:
            dr = densityreachable[index]
            if len(dr) < minpts:
                # noise point
                labels[index] = -1
            elif not cluster_respect_cannot_link_constraints(dr, cannotlink):
                for node in dr:
                    localclusters.append([node])
            else:
                # core point
                ldr = list(dr)
                labels[ldr] = 1
                localclusters.append(ldr)

    # Step 3b - Build the final clusters
    def compute_cluster_centroid(dataset, cluster):
        """
        Clusters são representados como coleções de indices.
        :param dataset: é uma matriz.
        :param cluster: é uma coleção de índices.
        :return: um vetor.
        """
        points = [dataset[p] for p in cluster]
        return np.mean(points)

    def compute_reachable_clusters(target_cluster, list_of_clusters, reachability_dictionary):
        """
        Clusters são representados como coleções de indices de elementos em um dataset.
        :param target_cluster: é um cluster, logo, é uma coleção de índices de elementos de um dataset.
        :param list_of_clusters: é uma __LISTA__, de clusters. Isto é, um tipo que associa a cada indices (de clusters)
        a coleções de indices (de elementos de um dataset).
        :param reachability_dictionary: é um dicionario que associa a cada indice (de elemento de um dataset) uma
        coleção de indices (de elementos de um dataset).
        :return: uma coleção de coleção de índices dos clusters em list_of_clusters que são reachable pelo
        target_cluster.
        """
        assert isinstance(o=list_of_clusters, t=list)

        reachable_sets_of_points = [reachability_dictionary(p) for p in target_cluster]
        reachable_points = set()
        for s in reachable_sets_of_points:
            reachable_points.update(s)

        reachable_clusters_indexes = list()
        for cluster_index, cluster_points in enumerate(list_of_clusters):
            for p in cluster_points:
                if p in reachable_points:
                    reachable_clusters_indexes.add(cluster_index)
                    break

        return reachable_clusters_indexes

    list_of_alpha_clusters = list(alphaclusters)
    list_of_local_clusters = list(localclusters)
    clusters_changed = True
    while clusters_changed:
        clusters_changed = False
        old_list_of_local_clusters = list(list_of_local_clusters)

        for index_of_lc, elements_of_lc in enumerate(old_list_of_local_clusters):
            indexes_of_reachable_alpha = compute_reachable_clusters(target_cluster=elements_of_lc,
                                                                    list_of_clusters=list_of_alpha_clusters,
                                                                    reachability_dictionary=densityreachable)
            centroids_of_reachable_alpha = [compute_cluster_centroid(dataset=dataset, cluster=list_of_alpha_clusters[i])
                                            for i in indexes_of_reachable_alpha]

            lc_centroid = compute_cluster_centroid(dataset=dataset, cluster=elements_of_lc)
            dist_to_reachable_alpha = [np.linalg.norm(lc_centroid - alpha_centroid)
                                       for alpha_centroid in centroids_of_reachable_alpha]

            index_of_closest_alpha_cluster = np.argmin(dist_to_reachable_alpha)[0]

            merged_cluster = set(elements_of_lc)
            merged_cluster.update(list_of_alpha_clusters[index_of_closest_alpha_cluster])
            if cluster_respect_cannot_link_constraints(cluster=merged_cluster, cl_constraints=cannotlink):
                del list_of_local_clusters[elements_of_lc]
                del list_of_alpha_clusters[list_of_alpha_clusters[index_of_closest_alpha_cluster]]
                list_of_alpha_clusters.append(merged_cluster)
                break

    # return each ALPHA_CLUSTER and each remaining LOCAL_CLUSTER
    # ? Como prefere?
    return None


def run_test():
    X = [
        (0, 0),
        (0, 1),
        (1, 0),
        (10, 0),
        (11, 1),
        (9, 0)]
    X = np.array(X)

    dr = compute_density_reachable_points(dataset=X,
                                          maximum_distance=NEIGHBORHOOD_RADIUS)
    print(dr)


if __name__ == "__main__":
    run_test()
