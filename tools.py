import numpy as np

def reshape(data):
    num_data = data.shape[0]
    reshaped_data = np.empty((num_data, 32, 32, 3,))
    for i in range(0, num_data):
        reshaped_data[i, :, :, 0] = data[i, 0:1024].reshape(32, 32)
        reshaped_data[i, :, :, 1] = data[i, 1024:2048].reshape(32, 32)
        reshaped_data[i, :, :, 2] = data[i, 2048:3072].reshape(32, 32)

    return reshaped_data