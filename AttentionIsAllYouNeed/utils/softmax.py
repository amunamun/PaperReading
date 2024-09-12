import numpy as np

def softmax(x):
    return  (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T