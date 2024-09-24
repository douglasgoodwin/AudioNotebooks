# def normalize(X):
# 	X -= X.min(axis=0)
# 	X /= X.max(axis=0)
# 	return X
# 
import numpy as np

def normalize(X):
    X = X - np.min(X, axis=0)
    X = X / np.ptp(X, axis=0)
    return X
