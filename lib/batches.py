import numpy as np
from numpy.random import randint

def get_batch(data, batch_size=32):
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data.T
    Y = data_dev[0]
    X_dev = data_dev[1:n]
    X = X_dev / 255.

    # Create batches
    num_batches = int(m / batch_size)
    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_X = X[:, start:end]
        batch_Y = Y[start:end]
        batches.append((batch_X, batch_Y))

    return batches, m