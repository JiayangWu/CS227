import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import random
from kshape import _sbd as SBD
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00015)
#_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
_mse_loss = tf.keras.losses.MeanSquaredError()
class LSTM_Ae(tf.keras.Model):
    @classmethod
    def __init__(self, **kwargs):

        X = kwargs["input"]

        if "optimizer" in kwargs:
            optimizer   = kwargs["optimizer"]
        else:
            optimizer   = _optimizer
        
        if "loss" in kwargs:
            loss        = kwargs["loss"]
        else:
            loss        = _mse_loss

        input_shape = X.shape
        # optimizer/learning rate
        self.embedding_size = 22000 
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.encoder_rnn_size = 1 # will tune more
        self.decoder_rnn_size = 1 # will tune more
        self.time_series_length = input_shape[0]
        self.window_size = input_shape[1]
        #self.feature_length = input_shape[2]
	
		# embeddings, encoder, decoder, and feed forward layers
		#Encoder layers
        #self.embedding1 = tf.keras.layers.Embedding(self.time_series_length, self.embedding_size, input_length=self.window_size) 
        self.encoder = tf.keras.layers.LSTM(self.window_size, return_sequences=True, return_state=True)
        
        self.embedding2 = tf.keras.layers.Embedding(self.time_series_length, self.embedding_size, input_length=self.window_size) 
        self.decoder = tf.keras.layers.LSTM(self.decoder_rnn_size, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(self.time_series_length, activation='softmax')

    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        # TODO:
        #1) Pass your french sentence embeddings to your encoder 
        encoder_embedding = self.embedding1(encoder_input)
        encoder_rnn_layer, encoder_last_input, encoder_last_cell_state = self.encoder(encoder_embedding)
        encoder_final_state = [encoder_last_input, encoder_last_cell_state]

        #2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
        decoder_embedding = self.embedding2(decoder_input)
        decoder_rnn_layer, decoder_last_input, decoder_last_cell_state = self.decoder(decoder_embedding, initial_state = encoder_final_state)

        #3) Apply dense layer(s) to the decoder out to generate probabilities
        prbs = self.dense_1(decoder_rnn_layer)

        return prbs

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
        # f = 10
        # k = 3
        
        # l = tf.keras.layers.Conv1D(filters=10, kernel_size=3, strides=1, padding="SAME", kernel_initializer=tf.random_normal_initializer(stddev=.1))
        # b = tf.keras.layers.BatchNormalization()
        # self.convs.append(l)
        # self.norms.append(b)

        # l = tf.keras.layers.Conv1D(filters=10, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=.1))
        # b = tf.keras.layers.BatchNormalization()
        # self.convs.append(l)
        # self.norms.append(b)

        # b = tf.keras.layers.BatchNormalization()
        # l = tf.keras.layers.Conv1D(filters=10, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=.1))
        # self.convs.append(l)
        # self.norms.append(b)

        # output_len = output_len - (k-1)
        # output_channels = 10
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
        # f = 10
        # k = 3
        # l = tf.keras.layers.Conv1DTranspose(f, k)
        # b = tf.keras.layers.BatchNormalization()
        # self.convs.append(l)
        # self.norms.append(b)

        # l = tf.keras.layers.Conv1DTranspose(f, k)
        # b = tf.keras.layers.BatchNormalization()
        # self.convs.append(l)
        # self.norms.append(b)

        # l = tf.keras.layers.Conv1DTranspose(f, k)
        # b = tf.keras.layers.BatchNormalization()
        # self.convs.append(l)
        # self.norms.append(b)

    def call(self, inputs, training=False):
        x = self.expand(inputs)
        x = self.reshape(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x, training=training)
            x = conv(x)
        assert self.expected_output_shape == x.shape[1:]
        return x



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

# train_step with LSTM auto-encoder
def train_step(X, Y, distance, auto_encoder, optimizer=_optimizer, loss = _mse_loss):
    with tf.GradientTape() as tape:
        print('X.shape: ', X.shape)
        #encoder_embedding_X = auto_encoder.embedding1(X)
        #encoder_embedding_Y = auto_encoder.embedding1(Y)
        X_codes, encoder_last_input_X, encoder_last_cell_state_X = auto_encoder.encoder(X, training=True)
        Y_codes, encoder_last_input_Y, encoder_last_cell_state_Y = auto_encoder.encoder(Y, training=True)
        encoder_final_state_X = [encoder_last_input_X, encoder_last_cell_state_X]
        encoder_final_state_Y = [encoder_last_input_Y, encoder_last_cell_state_Y]
        
        X_decodes, decoder_last_input_X, decoder_last_cell_state_X = auto_encoder.decoder(X_codes, training=True)
        Y_decodes, decoder_last_input_Y, decoder_last_cell_state_Y = auto_encoder.decoder(Y_codes, training=True)
        
        similarity_distance = 0
        # distance, path = fastdtw(codes, input, dist=euclidean)
        for i in range(len(X_codes) - 1):
            # print(flatten_and_normalize(input[i]))
            # print(flatten_and_normalize(codes[i]))
            # print('X_decodes.shape: ', X_decodes.shape)
            # print('X_decodes[i].shape: ', X_decodes[i].shape)
            # print('np.array(X_decodes[i]).squeeze().shape: ', np.array(X_decodes[i]).squeeze().shape)
            codes_a, codes_b = np.array(X_decodes[i]).squeeze(), np.array(Y_decodes[i]).squeeze()
            similarity_distance += abs(euclidean(codes_a, codes_b) - distance[i])

        similarity_distance /= len(X_codes) - 1

        # print(loss(input, decodes), similarity_distance)
        #loss = loss(input, decodes) + similarity_distance
        rec_loss = loss(X, X_decodes) + loss(Y, Y_decodes)
        print('rec_loss: ', rec_loss)
        loss = loss(X, X_decodes) + loss(Y, Y_decodes) + similarity_distance

        trainables = auto_encoder.encoder.trainable_variables + auto_encoder.decoder.trainable_variables

    gradients = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(gradients, trainables))
    return loss


# @tf.function
# train_step with vanilla auto-encoder
# def train_step(X, Y, distance, auto_encoder, optimizer=_optimizer, loss = _mse_loss):
#     with tf.GradientTape() as tape:
#         X_codes = auto_encoder.encode(X, training=True)
#         Y_codes = auto_encoder.encode(Y, training=True)
#         X_decodes = auto_encoder.decode(X_codes, training=True)
#         Y_decodes = auto_encoder.decode(Y_codes, training=True)
        
#         similarity_distance = 0
#         # distance, path = fastdtw(codes, input, dist=euclidean)
#         for i in range(len(X_codes) - 1):
#             # print(flatten_and_normalize(input[i]))
#             # print(flatten_and_normalize(codes[i]))
#             codes_a, codes_b = X_decodes[i], Y_decodes[i]
#             similarity_distance += abs(euclidean(codes_a, codes_b) - distance[i])

#         similarity_distance /= len(X_codes) - 1

#         # print(loss(input, decodes), similarity_distance)
#         #loss = loss(input, decodes) + similarity_distance
#         loss = loss(X, X_decodes) + loss(Y, Y_decodes) + similarity_distance

#         trainables = auto_encoder.encode.trainable_variables + auto_encoder.decode.trainable_variables

#     gradients = tape.gradient(loss, trainables)
#     optimizer.apply_gradients(zip(gradients, trainables))
#     return loss

# train_step taking in single input array

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

