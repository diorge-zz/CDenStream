from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KDTree
from .constraint import cluster_respect_cannot_link_constraints
import numpy as np


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


class Cluster:
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

    densityreachable = compute_density_reachable_points(dataset, epsilon)

    for index, point in enumerate(dataset):
        # if point is yet unlabeled
        if clusters[index] == -1:
            dr = densityreachable[index]
            if len(dr) < minpts:
                # noise point
                pass
            elif not cluster_respect_cannot_link_constraints(dr, cannotlink):
                for node in dr:
                    clusterpoints = (node,)
                    allclusters[nextcluster] = Cluster('local', clusterpoints)
                    clusters[node] = nextcluster
                    nextcluster += 1
            else:
                # core point
                ldr = list(dr)
                clusterpoints = tuple(ldr)
                allclusters[nextcluster] = Cluster('local', clusterpoints)
                clusters[ldr] = nextcluster
                nextcluster += 1

    # Step 3a: merge must-link constraints
    for ml1, ml2 in mustlink:
        c1 = clusters[ml1]
        c2 = clusters[ml2]
        if c1 == c2:
            continue
        if c1 != -1:
            points_of_c1 = allclusters[c1]
            del allclusters[c1]
        else:
            points_of_c1 = Cluster('noise', [ml1])
        if c2 != -1:
            points_of_c2 = allclusters[c2]
            del allclusters[c2]
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
                for p in cluster.points:
                    if p in reachable_points:
                        reachable_clusters_indexes.append(cluster_index)
                        break

        return reachable_clusters_indexes

    clusters_changed = True
    while clusters_changed:
        clusters_changed = False

        for index_of_lc, lc in allclusters.items():
            if lc.kind == 'local':
                elements_of_lc = lc.points
                indexes_of_reachable_alpha = compute_reachable_clusters(index_of_lc, 'alpha')
                if len(indexes_of_reachable_alpha) > 0:
                    centroids_of_reachable_alpha = [compute_cluster_centroid(i)
                                                    for i in indexes_of_reachable_alpha]

                    lc_centroid = compute_cluster_centroid(index_of_lc)
                    dist_to_reachable_alpha = [np.linalg.norm(lc_centroid - alpha_centroid)
                                            for alpha_centroid in centroids_of_reachable_alpha]

                    index_of_closest_alpha_cluster = indexes_of_reachable_alpha[np.argmin(dist_to_reachable_alpha)]

                    merged_cluster = set(elements_of_lc)
                    merged_cluster.update(allclusters[index_of_closest_alpha_cluster])
                    if cluster_respect_cannot_link_constraints(cluster=merged_cluster, cl_constraints=cannotlink):
                        del allclusters[index_of_lc]
                        del allclusters[index_of_closest_alpha_cluster]
                        allclusters[nextcluster] = Cluster('alpha', tuple(merged_cluster))
                        for point in merged_cluster:
                            clusters[point] = nextcluster
                        nextcluster += 1
                        clusters_changed = True
                        break

    return [x.points for x in allclusters.values()]
