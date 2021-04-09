import tensorflow as tf
import numpy as np
import random
from kshape import _sbd as SBD
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
# from  import mean_absolute_error as MAE

class Encoder(tf.keras.Model):

    def __init__(self, input_shape, code_size, filters, kernel_sizes):
        super(Encoder, self).__init__()
        assert len(filters) == len(kernel_sizes)
        assert len(input_shape) == 2 # (x, y), x = # of samples, y = # of vars
        #self.input_shape = input_shape
        self.code_size = code_size

        self.convs = []
        self.norms = []
        output_len = input_shape[0]
        output_channels = input_shape[1]

        for f, k in zip(filters, kernel_sizes):
            l = tf.keras.layers.Conv1D(f, k, activation="relu")
            b = tf.keras.layers.BatchNormalization()
            # d = tf.keras.layers.Dropout(0.2)
            self.convs.append(l)
            self.norms.append(b)
            output_len = output_len - (k-1)
            output_channels = f

        self.last_kernel_shape = (output_len, output_channels)
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(code_size)

    def call(self, inputs, training=False):

        x = self.convs[0](inputs)
        x = self.norms[0](x)
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            x = conv(x)
            x = norm(x, training=training)
        assert x.shape[1:] == self.last_kernel_shape
        #print(x.shape)
        x = self.flatten(x)

        x = self.out(x)
        return x

class Decoder(tf.keras.Model):

    def __init__(self, code_size, last_kernel_shape, output_shape, filters, kernel_sizes):
        super(Decoder, self).__init__()

        assert len(last_kernel_shape) == 2
        assert len(output_shape) == 2 # (x, y) x = # of samples, y = samples n variables

        self.code_size = code_size
        self.last_kernel_shape = last_kernel_shape
        self.expected_output_shape = output_shape

        flat_len = last_kernel_shape[0] * last_kernel_shape[1]

        self.expand = tf.keras.layers.Dense(flat_len)
        self.reshape = tf.keras.layers.Reshape(last_kernel_shape)

        self.convs = []
        self.norms = []

        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            l = tf.keras.layers.Conv1DTranspose(f, k)
            b = tf.keras.layers.BatchNormalization()
            self.convs.append(l)
            self.norms.append(b)

    def call(self, inputs, training=False):
        x = self.expand(inputs)
        x = self.reshape(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x, training=training)
            x = conv(x)
        assert self.expected_output_shape == x.shape[1:]
        return x

_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00015)
_mse_loss = tf.keras.losses.MeanSquaredError()

class AutoEncoder:
    def __init__(self, **kwargs):

        input_shape     = kwargs["input_shape"]
        code_size       = kwargs["code_size"]
        filters         = kwargs["filters"]
        kernel_sizes    = kwargs["kernel_sizes"]

        if "loss" in kwargs:
            loss        = kwargs["loss"]
        else:
            loss        = _mse_loss

        if "optimizer" in kwargs:
            optimizer   = kwargs["optimizer"]
        else:
            optimizer   = _optimizer

        self.encode = Encoder(input_shape, code_size, filters, kernel_sizes)

        decoder_filters = list(filters[:len(filters)-1])
        decoder_filters.append(input_shape[1])
        last_kernel_shape = self.encode.last_kernel_shape

        self.decode = Decoder(code_size, last_kernel_shape, input_shape, decoder_filters,
                kernel_sizes)

        self.loss = loss
        self.optimizer = optimizer
    
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

def flatten_and_normalize(tensor):
    # print(tf.reshape(tensor,[-1]))
    return normalize(tf.reshape(tensor,[-1]))

def ED(X):
    return tf.math.reduce_euclidean_norm(X, 1)

# @tf.function
def train_step(X, Y, distance, auto_encoder, optimizer=_optimizer, loss = _mse_loss, alpha = 1):
    with tf.GradientTape() as tape:
        # print(np.shape(X))
        X_codes = auto_encoder.encode(X, training=True)
        Y_codes = auto_encoder.encode(Y, training=True)

        X_decodes = auto_encoder.decode(X_codes, training=True)
        Y_decodes = auto_encoder.decode(Y_codes, training=True)
        
        # all_decodes =  tf.concat([X_decodes, Y_decodes], axis = 0)
        # all_inputs =  tf.concat([X, Y], axis = 0)
        # print(ED(X_codes - Y_codes))
        # print(distance)
        # print(X_codes, Y_codes)
        # return
        similarity_loss = 0
        if alpha > 0:
            subtraction = ED(X_codes - Y_codes) - distance
            similarity_loss = tf.math.reduce_sum(subtraction) / np.shape(X)[0]
        reconstruction_loss = loss(X, X_decodes) + loss(Y, Y_decodes)
        # reconstruction_loss = loss(all_decodes, all_inputs)
        print("\nrec_loss:", reconstruction_loss, "simi_loss:", similarity_loss)

        loss = reconstruction_loss + alpha * similarity_loss

        trainables = auto_encoder.encode.trainable_variables + auto_encoder.decode.trainable_variables

    gradients = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(gradients, trainables))
    return loss

# def train_step(input, auto_encoder, optimizer=_optimizer, loss = _mse_loss):
#     with tf.GradientTape() as tape:
#         codes = auto_encoder.encode(input, training=True)
#         decodes = auto_encoder.decode(codes, training=True)
        
#         similarity_distance = 0
#         # distance, path = fastdtw(codes, input, dist=euclidean)
#         for i in range(len(codes) - 1):
#             # print(flatten_and_normalize(input[i]))
#             # print(flatten_and_normalize(codes[i]))
#             input_a, input_b = flatten_and_normalize(input[i]), flatten_and_normalize(input[i + 1])
#             codes_a, codes_b = flatten_and_normalize(codes[i]), flatten_and_normalize(codes[i + 1])
#             similarity_distance += abs(euclidean(codes_a, codes_b) - SBD(input_a, input_b))

#         similarity_distance /= len(codes) - 1

#         # print(loss(input, decodes), similarity_distance)
#         loss = loss(input, decodes) + similarity_distance

#         trainables = auto_encoder.encode.trainable_variables + auto_encoder.decode.trainable_variables

#     gradients = tape.gradient(loss, trainables)
#     optimizer.apply_gradients(zip(gradients, trainables))
#     return loss