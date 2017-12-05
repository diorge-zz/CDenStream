import numpy as np


def expand_constraints_transitively(constraints):
    def add_pair_to_set(pair, set):
        if pair not in set and (pair[1], pair[0]) not in set:
            set.add(pair)

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


def sanitize_constraints(X, must_link, cannot_link):
    """
    This function checks if the must_link constraints and cannot_link constraints may be respected.
    To do so it generates all constraints that can be reached transitively, removes constraints that
    link a element to itself or prohibits such links and, finally, compares the must_link constraints
    to the cannot_link constraints.
    It also checks if all contraints point to valid elements of X.
    :param X: tabular dataset.
    :param must_link: set of pairs of indexes of elements that must be linked.
    :param cannot_link: set of pairs of indexes of elements that cannot be linked.
    :return: This function returns a pair containing the expanded must_link_constraints and
    expanded cannot_link_constraints.
    """
    if must_link is None:
        must_link = set()
    if cannot_link is None:
        cannot_link = set()

    # Checking if constraints are made of pairs
    if any((c for c in must_link if len(c) != 2)):
        raise Exception('must_link must contain only pairs.')
    if any((c for c in cannot_link if len(c) != 2)):
        raise Exception('cannot_link must contain only pairs.')

    # Checking if constraints point to elements outside of the dataset
    valid_range = range(0, X.shape[0])
    if any((c for c in must_link if c[0] not in valid_range or c[1] not in valid_range)):
        raise Exception('must_link contains references to elements outside the dataset.')
    if any((c for c in cannot_link if c[0] not in valid_range or c[1] not in valid_range)):
        raise Exception('cannot_link contains references to elements outside the dataset.')

    # Expanding constraints transitively, removing identities and sorting tuples to prevent
    must_link = expand_constraints_transitively(must_link)
    must_link = [(c[0], c[1]) if c[0] < c[1] else (c[1], c[0])
                 for c in must_link
                 if c[0] != c[1]]

    cannot_link = expand_constraints_transitively(cannot_link)
    cannot_link = [(c[0], c[1]) if c[0] < c[1] else (c[1], c[0])
                   for c in cannot_link
                   if c[0] != c[1]]

    # Checking if after merging the must_link constraints, it is still possible to respect the cannot_link constraints
    for c in must_link:
        if c in cannot_link:
            raise Exception(f"Constraint {c} is produced both by must_link and cannot_link.")

    return must_link, cannot_link


def cluster_respect_cannot_link_constraints(cluster, cannot_link):
    if not cannot_link:
        return True

    broken_constraint = any((pair for pair in cannot_link if pair[0] in cluster and pair[1] in cluster))
    return not broken_constraint


def generate_constraints_from_labels(constraint_count, labels):
    if constraint_count < 0 or constraint_count > len(labels):
        raise Exception('constraint_count must be between [0,len(labels].')

    must_link = set()
    cannot_link = set()
    while len(must_link) + len(cannot_link) < constraint_count:
        lhs = np.random.randint(len(labels))
        rhs = np.random.randint(len(labels))
        if labels[lhs] == labels[rhs]:
            must_link.add((lhs, rhs))
        else:
            cannot_link.add((lhs, rhs))

    return must_link, cannot_link
