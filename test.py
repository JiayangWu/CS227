#### Settings ######
import py_ts_data
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from kshape import _sbd as SBD
from auto_encoder import AutoEncoder, train_step
from utilities import min_max, normalize, augment_data, generateRandomPairs, calculatePreSBD

##### settings below
dataset = "UMD"
EPOCHS = 500
enable_data_augmentation = False
percentage_similarity_loss = 0
LSTM = False
##### settings above

from train_and_test import trainAndTest