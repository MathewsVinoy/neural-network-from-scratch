import numpy as np


def get_accuraacy(out , y):
    pred = np.argmax(out, 0)
    return (np.sum(pred == y)/y.size)*100