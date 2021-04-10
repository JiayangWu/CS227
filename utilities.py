import numpy as np
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

def augment_data(X_train, K = 1000, alpha = 0.1):
    # K is the target size, alpha is the fluctuated rate of fabricated data.
    repeat_times = K // len(X_train) - 1
    new_x_data = None
    X_train_mean = np.mean(X_train)
    for _ in range(repeat_times):
        #for x_series, y_series in zip(X_train, y_train):
        for x_series in X_train:
            #same noise 0.1 * mean: x_noise = 0.2 * (np.random.random(x_series.shape) - 0.5) * X_train_mean

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

    