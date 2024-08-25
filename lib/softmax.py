import numpy as np

def Softmax(inputs):
    A = np.exp(inputs) / sum(np.exp(inputs))
    return A