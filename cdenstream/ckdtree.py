import numpy as np


def ckdtree(X, cannot_link):
    clusters = []
    shape = X.shape
    for d in range(shape[1]):
        median = np.median(X[:, d])
        print(median)
