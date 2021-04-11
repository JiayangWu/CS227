import numpy as np
from kshape import _sbd as SBD
import math
def min_max(data, feature_range=(0, 1)):
    """
    implements min-max scaler
    """
    min_v = feature_range[0]
    max_v = feature_range[1]
    max_vals = data.max(axis=1)[:, None, :]
    min_vals = data.min(axis=1)[:, None, :]
    X_std = (data - min_vals) / (max_vals - min_vals)
    return X_std * (max_v - min_v) + min_v

def normalize(data):
    """
    Z-normalize data with shape (x, y)
    x = # of timeseries
    y = len of each timeseries
    
    s.t. each array in [., :, .] (i.e. each timeseries variable)
    is zero-mean and unit stddev
    """
    # sz, l = data.shape
    means = np.mean(data)
    stddev = np.std(data)
    return (data - means)/stddev

def augment_data(X_train, K = 1000, alpha = 0.1, enable_same_noise = False):
    # K is the target size, alpha is the fluctuated rate of fabricated data.
    repeat_times = K // len(X_train) - 1
    new_x_data = None
    X_train_mean = np.mean(X_train)
    for _ in range(repeat_times):
        #for x_series, y_series in zip(X_train, y_train):
        for x_series in X_train:
            # print(enable_same_noise)
            if enable_same_noise:
                # same noise 0.1 * mean: 
                random = np.random.random(x_series.shape) # [0 - 1]
                x_noise = 0#((random + 1)/10) * X_train_mean # [0.1, 0.2] * mean
                new_x_series = np.reshape(x_series * x_noise, (1, len(x_series), 1))
            else:
                x_noise = 1 - 0.1 * (np.random.random(x_series.shape) - 0.5)
                new_x_series = np.reshape(x_series * x_noise, (1, len(x_series), 1))

            if new_x_data is None:
                new_x_data = new_x_series
            else:
                new_x_data = np.append(new_x_data, new_x_series, axis = 0)

    print("The original data shape is:", np.shape(X_train))
    X_train = np.append(X_train, new_x_data, axis = 0)
    print("The augmented data shape is:", np.shape(X_train))

    return X_train


def generateRandomPairs(K, X_train):
    num_train = len(X_train)
    # indices1 = np.random.choice(num_train, int(num_train * math.log2(num_train)))
    # indices2 = np.random.choice(num_train, int(num_train * math.log2(num_train)))
    indices1 = np.random.choice(num_train, K)# * math.log2(num_train)))
    indices2 = np.random.choice(num_train, K)# * math.log2(num_train)))

    X = X_train[indices1]
    Y = X_train[indices2]
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)

    return X, Y
    

def calculatePreSBD(X, Y):
    normalized_X = normalize(X)
    normalized_Y = normalize(Y)

    normalized_X_2d = normalize(X).squeeze()
    normalized_Y_2d = normalize(Y).squeeze()

    distance = []
    for x, y in zip(normalized_X_2d, normalized_Y_2d):
        distance.append(SBD(np.array(x), np.array(y)))
    distance = np.array(distance)
    print('distance shape: ', distance.shape)
    return normalized_X, normalized_Y, distance