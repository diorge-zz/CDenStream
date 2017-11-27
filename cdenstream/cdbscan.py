from sklearn.datasets import load_iris
from ckdtree import ckdtree


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


def sanitize_and_expand_constraints(X, must_link, cannot_link):
    # Checking if constraints are made of pairs
    if any((c for c in must_link if len(c) != 2)):
        raise Exception('must_link must contain only pairs')
    if any((c for c in cannot_link if len(c) != 2)):
        raise Exception('cannot_link must contain only pairs')

    # Checking if constraints point to elements outside of the dataset
    valid_range = range(0, X.shape[0])
    if any((c for c in must_link if c[0] not in valid_range or c[1] not in valid_range)):
        raise Exception('must_link contains references to elements outside the dataset.')
    if any((c for c in cannot_link if c[0] not in valid_range or c[1] not in valid_range)):
        raise Exception('cannot_link contains references to elements outside the dataset.')

    # Expanding, removing identities and sorting constraints
    must_link = expand_constraints_transitively(must_link)
    must_link = [c for c in must_link if c[0] != c[1]]
    must_link = sorted(must_link, key=lambda p: p)

    cannot_link = expand_constraints_transitively(cannot_link)
    cannot_link = [c for c in cannot_link if c[0] != c[1]]
    cannot_link = sorted(cannot_link, key=lambda p: p)

    # Checking if after merging the must_link constraints, it is still possible to respect the cannot_link constraints
    for c in must_link:
        if c in cannot_link:
            raise Exception(f"Constraint {c} is produced both by must_link and cannot_link.")

    return must_link, cannot_link


def run_test():
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(X.shape)

    must_link = [(0, 1),
                 (1, 2),
                 (2, 3),
                 (3, 4),
                 (4, 5)]

    cannot_link = [(5, 10),
                   (13, 12),
                   (149, 12)]

    must_link, cannot_link = sanitize_and_expand_constraints(X=X, must_link=must_link, cannot_link=cannot_link)
    print(f"must_link: {must_link}")
    print(f"cannot_link: {cannot_link}")


if __name__ == "__main__":
    run_test()
