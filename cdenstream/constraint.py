def add_pair_to_set(pair, set):
    if pair not in set and (pair[1], pair[0]) not in set:
        set.add(pair)


def expand_constraints_transitively(constraints):
    old_constraints = set()
    updated_constraints = set(constraints)
    while old_constraints != updated_constraints:
        old_constraints = set(updated_constraints)
        for a, b in old_constraints:
            for c, d in old_constraints:
                if a in (c, d) or b in (c, d):
                    add_pair_to_set(pair=(a, c), set=updated_constraints)
                    add_pair_to_set(pair=(a, d), set=updated_constraints)
                    add_pair_to_set(pair=(b, c), set=updated_constraints)
                    add_pair_to_set(pair=(b, d), set=updated_constraints)
    return updated_constraints


def sanitize_and_expand_constraints(X, ml_constraints, cl_constraints):
    """
    This function checks if the must_link constraints and cannot_link constraints may be respected.
    To do so it generates all constraints that can be reached transitively, removes constraints that
    link a element to itself or prohibits such links and, finally, compares the must_link constraints
    to the cannot_link constraints.
    It also checks if all contraints point to valid elements of X.
    :param X: tabular dataset.
    :param ml_constraints: set of pairs of indexes of elements that must be linked.
    :param cl_constraints: set of pairs of indexes of elements that cannot be linked.
    :return: This function returns a pair containing the expanded must_link_constraints and
    expanded cannot_link_constraints.
    """
    # Checking if constraints are made of pairs
    if any((c for c in ml_constraints if len(c) != 2)):
        raise Exception('must_link must contain only pairs')
    if any((c for c in cl_constraints if len(c) != 2)):
        raise Exception('cannot_link must contain only pairs')

    # Checking if constraints point to elements outside of the dataset
    valid_range = range(0, X.shape[0])
    if any((c for c in ml_constraints if c[0] not in valid_range or c[1] not in valid_range)):
        raise Exception('must_link contains references to elements outside the dataset.')
    if any((c for c in cl_constraints if c[0] not in valid_range or c[1] not in valid_range)):
        raise Exception('cannot_link contains references to elements outside the dataset.')

    # Expanding, removing identities and sorting constraints
    ml_constraints = expand_constraints_transitively(ml_constraints)
    ml_constraints = [c for c in ml_constraints if c[0] != c[1]]
    ml_constraints = sorted(ml_constraints, key=lambda p: p)

    cl_constraints = expand_constraints_transitively(cl_constraints)
    cl_constraints = [c for c in cl_constraints if c[0] != c[1]]
    cl_constraints = sorted(cl_constraints, key=lambda p: p)

    # Checking if after merging the must_link constraints, it is still possible to respect the cannot_link constraints
    for c in ml_constraints:
        if c in cl_constraints:
            raise Exception(f"Constraint {c} is produced both by must_link and cannot_link.")

    return ml_constraints, cl_constraints


def cluster_respect_cannot_link_constraints(cluster, cl_constraints):
    if not cl_constraints:
        return True

    broken_constraint = any((pair for pair in cl_constraints if pair[0] in cluster and pair[1] in cluster))
    return not broken_constraint


def partition_respect_cannot_link_constraints(partition, cl_constraints):
    return all((cluster_respect_cannot_link_constraints(cluster=c, cl_constraints=cl_constraints) for c in partition))


def compute_respected_must_link_constraints(cluster, ml_constraints):
    respected_constraints = [pair for pair in ml_constraints if pair[0] in cluster and pair[1] in cluster]
    return respected_constraints


def partition_respect_must_link_constraints(partition, ml_constraints):
    respected_constraints = set()
    for cluster in partition:
        respected_constraints.update(
            compute_respected_must_link_constraints(cluster=cluster, ml_constraints=ml_constraints))
    return len(respected_constraints) == len(ml_constraints)
