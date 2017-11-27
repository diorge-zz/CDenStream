from sklearn.datasets import load_iris
from ckdtree import ckdtree


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
