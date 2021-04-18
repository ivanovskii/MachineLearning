from sklearn.datasets import load_boston
from itertools import combinations
import numpy as np

def extended_boston():
    
    boston = load_boston()
    X, y = boston['data'], boston['target']

    pairwise_product = np.array(list(map(lambda Xy: Xy[0]*Xy[1], combinations(X.T, 2)))).T
    X = np.append(np.append(X, X**2, axis=1), pairwise_product, axis=1)

    return [X, y]
