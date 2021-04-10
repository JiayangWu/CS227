import py_ts_data
import time
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import os
from kshape import _sbd as SBD
from auto_encoder import AutoEncoder, train_step
from utilities import min_max, normalize, augment_data

dataset = "GunPoint"
EPOCHS = 500
data_augmentation = False

X_train, y_train, X_test, y_test, info = py_ts_data.load_data(dataset, variables_as_channels=True)
print("Dataset shape: Train: {}, Test: {}".format(X_train.shape, X_test.shape))
print(np.shape(y_train))



num_train = len(X_train)
if num_train < 1000 and data_augmentation:
    X_train= augment_data(X_train)
    num_train = len(X_train)

# NlogN is too large, for N = 1000, NlogN would be 10K
# indices1 = np.random.choice(num_train, int(num_train * math.log2(num_train)))
# indices2 = np.random.choice(num_train, int(num_train * math.log2(num_train)))
indices1 = np.random.choice(num_train, int(num_train))
indices2 = np.random.choice(num_train, int(num_train))

print(len(indices1), num_train, math.log2(num_train))
X = X_train[indices1]
Y = X_train[indices2]
print('X shape: ', X.shape)
print('Y shape: ', Y.shape)

normalized_X = normalize(X)
normalized_Y = normalize(Y)

normalized_X_2d = normalize(X).squeeze()
normalized_Y_2d = normalize(Y).squeeze()

distance = []
for x, y in zip(normalized_X_2d, normalized_Y_2d):
    distance.append(SBD(np.array(x), np.array(y)))
distance = np.array(distance)
print('distance shape: ', distance.shape)
#distance = SBD(normalized_X, normalized_Y)


#%%
# fig, axs = plt.subplots(1, 2, figsize=(10, 3))
# axs[0].plot(X_train[0])
# X_train = min_max(X_train, feature_range=(-1, 1))
# axs[1].plot(X_train[0])
# X_test = min_max(X_test, feature_range=(-1, 1))
# plt.show()

#%% [markdown]
# # Encode and Decode

#%%
kwargs = {
    "input_shape": (X_train.shape[1], X_train.shape[2]),
    "filters": [32, 64, 128],
    "kernel_sizes": [5, 5, 5],
    "code_size": 16,
}

ae = AutoEncoder(**kwargs)

#%% [markdown]
# # Training

#%%

# SHUFFLE_BUFFER = 100
# K = len(set(y_train))

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH)

loss_history = []
t1 = time.time()
for epoch in range(EPOCHS):
    #total_loss = train_step(X_train, ae)
    total_loss = train_step(normalized_X, normalized_Y, distance, ae, alpha = 0.5)
    loss_history.append(total_loss)
    print("Epoch {}: {}".format(epoch, total_loss), end="\r")
    
print("The training time is:", (time.time() - t1) / 60)
# plt.plot(loss_history)


#%%
# plt.xlim(left = 5, right = len(loss_history))
plt.xlabel("epoch starting from 5")
plt.ylabel("loss")
plt.title("Loss vs epoch")
# print(loss_history[5:])
plt.plot(loss_history[5:])
# plt.show()
if not os.path.isdir("./images/" + dataset):
    os.mkdir("./images/" + dataset)

plt.savefig("./images/" + dataset + "/loss.png")

#%%
code_test = ae.encode(X_test)
decoded_test = ae.decode(code_test)
plt.clf()
plt.plot(X_test[0], label = "Original TS")
plt.plot(decoded_test[0], label = "reconstructed TS")

plt.savefig("./images/" + dataset + "/reconstruction.png")
# plt.show()

losses = []
for ground, predict in zip(X_test, decoded_test):
    losses.append(np.linalg.norm(ground - predict))

L2_distance = np.array(losses).mean()
print("Mean L2 distance: {}".format(L2_distance))


#%%
from sklearn.neighbors import NearestNeighbors

def nn_dist(x, y):
    """
    Sample distance metric, here, using only Euclidean distance
    """
    # x = x.reshape((50, 2))
    # y = y.reshape((50, 2))
    return np.linalg.norm(x - y)

nn_x_test = X_test.reshape((-1, np.shape(X_test)[1]))
baseline_nn = NearestNeighbors(n_neighbors=10, metric=nn_dist).fit(nn_x_test)
code_nn = NearestNeighbors(n_neighbors=10).fit(code_test)

# For each item in the test data, find its 11 nearest neighbors in that dataset (the nn is itself)
baseline_11nn = baseline_nn.kneighbors(nn_x_test, 11, return_distance=False)
code_11nn     = code_nn.kneighbors(code_test, 11, return_distance=False)

# On average, how many common items are in the 10nn?
result = []
for b, c in zip(baseline_11nn, code_11nn):
    # remove the first nn (itself)
    b = set(b[1:])
    c = set(c[1:])
    result.append(len(b.intersection(c)))

ten_nn_score = np.array(result).mean()
print("10-nn score is:", ten_nn_score)
with open("./images/" + dataset + "/record.txt", "w") as f:
    f.write(" ".join([str(round(L2_distance,2)), str(round(ten_nn_score,2))]))



