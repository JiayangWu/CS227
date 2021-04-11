import tensorflow as tf
import numpy as np
import random
from kshape import _sbd as SBD
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
# from  import mean_absolute_error as MAE

class Encoder(tf.keras.Model):
    def __init__(self, input_shape, code_size, filters, kernel_sizes, LSTM = False):
        super(Encoder, self).__init__()
        assert len(filters) == len(kernel_sizes)
        assert len(input_shape) == 2 # (x, y), x = # of samples, y = # of vars
        #self.input_shape = input_shape
        self.code_size = code_size
        self.LSTM_layer = LSTM
        self.convs = []
        self.norms = []
        output_len = input_shape[0]
        output_channels = input_shape[1]

        for f, k in zip(filters, kernel_sizes):
            l = tf.keras.layers.Conv1D(f, k, activation="relu")
            # b = tf.keras.layers.BatchNormalization()
            # d = tf.keras.layers.Dropout(0.2)
            self.convs.append(l)
            # self.norms.append(b)
            output_len = output_len - (k-1)
            output_channels = f

        self.last_kernel_shape = (output_len, output_channels)
        self.flatten = tf.keras.layers.Flatten()
        self.LSTM = tf.keras.layers.LSTM(output_len, return_sequences=False, return_state=False)
        self.out = tf.keras.layers.Dense(code_size)

    def call(self, inputs, training=False, LSTM = False):

        x = self.convs[0](inputs)
        # x = self.norms[0](x)
        # for conv, norm in zip(self.convs[1:], self.norms[1:]):
        #     x = conv(x)
        #     x = norm(x, training=training)

        for conv in self.convs[1:]:
            x = conv(x)

        assert x.shape[1:] == self.last_kernel_shape
        # print(x.shape)
        if LSTM or self.LSTM_layer:
            x = self.LSTM(x) 
        x = self.flatten(x)
        # print(x.shape)
        # print(LSTM or self.LSTM_layer)

        x = self.out(x)
        return x

class Decoder(tf.keras.Model):
    def __init__(self, code_size, last_kernel_shape, output_shape, filters, kernel_sizes, LSTM_size):
        super(Decoder, self).__init__()

        assert len(last_kernel_shape) == 2
        assert len(output_shape) == 2 # (x, y) x = # of samples, y = samples n variables

        self.code_size = code_size
        self.last_kernel_shape = last_kernel_shape
        self.expected_output_shape = output_shape

        flat_len = last_kernel_shape[0] * last_kernel_shape[1]

        self.expand = tf.keras.layers.Dense(flat_len)
        # self.LSTM = tf.keras.layers.LSTM(LSTM_size, return_sequences=True, return_state=False)
        self.reshape = tf.keras.layers.Reshape(last_kernel_shape)

        self.convs = []
        # self.norms = []

        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            l = tf.keras.layers.Conv1DTranspose(f, k)
            # b = tf.keras.layers.BatchNormalization()
            self.convs.append(l)
            # self.norms.append(b)
        
        

    def call(self, inputs, training=False):
        # print(np.shape(inputs))
        x = self.expand(inputs)
        
        # print(np.shape(x))
        x = self.reshape(x)
        # print(np.shape(x), "hjkhkhkj")
        # x = self.LSTM(x)
        # print(np.shape(x), "jlkjlkjl")
        # for conv, norm in zip(self.convs, self.norms):
            # x = norm(x, training=training)
            # x = conv(x)
        for conv in self.convs:
            x = conv(x)
        assert self.expected_output_shape == x.shape[1:]
        # print(np.shape(x), "Dsadasdasdasdasdas")
        return x

_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)
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

        self.encode = Encoder(input_shape, code_size, filters, kernel_sizes, LSTM = False)

        decoder_filters = list(filters[:len(filters)-1])
        decoder_filters.append(input_shape[1])
        last_kernel_shape = self.encode.last_kernel_shape

        self.decode = Decoder(code_size, last_kernel_shape, input_shape, decoder_filters,
                kernel_sizes, LSTM_size=input_shape[0])

        self.loss = loss
        self.optimizer = optimizer

def ED(X):
    return tf.math.reduce_euclidean_norm(X, 1)

# @tf.function
def train_step(X, Y, distance, auto_encoder, optimizer=_optimizer, loss = _mse_loss, alpha = 1, LSTM = False):
    with tf.GradientTape() as tape:
        # print(np.shape(X))
        X_codes = auto_encoder.encode(X, training=True, LSTM = LSTM)
        Y_codes = auto_encoder.encode(Y, training=True, LSTM = LSTM)

        X_decodes = auto_encoder.decode(X_codes, training=True)
        Y_decodes = auto_encoder.decode(Y_codes, training=True)
        
        similarity_loss = 0
        if alpha > 0:
            subtraction = tf.abs(ED(X_codes - Y_codes) - distance)
            similarity_loss = tf.math.reduce_sum(subtraction) / np.shape(distance)
        reconstruction_loss = loss(X, X_decodes) + loss(Y, Y_decodes)

        # print("\nrec_loss:", reconstruction_loss, "simi_loss:", similarity_loss)
        loss = reconstruction_loss + alpha * similarity_loss

        trainables = auto_encoder.encode.trainable_variables + auto_encoder.decode.trainable_variables

    gradients = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(gradients, trainables))
    return loss
