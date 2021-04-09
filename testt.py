# import sys
# sys.path.append("/Users/fsolleza/Documents/Projects/timeseries-data") # path to this repository
import py_ts_data
import time
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
from kshape import _sbd as SBD
from auto_encoder import AutoEncoder, train_step
from utilities import min_max, normalize, augment_data

X_train, y_train, X_test, y_test, info = py_ts_data.load_data("Plane", variables_as_channels=True)
print("Dataset shape: Train: {}, Test: {}".format(X_train.shape, X_test.shape))
print(np.shape(y_train))