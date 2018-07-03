'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.88 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function
import numpy as np
import cPickle


np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D, LSTM
from keras.datasets import imdb
from keras import backend as K
from keras.optimizers import Adadelta
from load_data import load_data_shuffle
from keras.preprocessing import sequence as sq
from keras.layers import Dense, Dropout, Activation, Lambda,merge,Input,TimeDistributed,Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import keras.backend.tensorflow_backend as K

#config = K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})


# tf_config = K.tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# session = K.tf.Session(config=tf_config)
# K.set_session(session)

# config = K.tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, \
#                         allow_soft_placement=True, device_count = {'CPU': 4})
# session = K.tf.Session(config=config)
# K.set_session(session)

# set parameters:
max_features = 21540#14300
maxlen = 400
batch_size = 10
embedding_dims = 200
nb_filter = 150
filter_length = 3
hidden_dims = 100
nb_epoch = 14

cvs = [1,2,3,4,5]
accs = []
for cv in cvs:
    print('Loading data for cv...', cv)

    X_train, y_train, X_test, y_test, X_val, y_val = load_data_shuffle(cv)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    X_train = sq.pad_sequences(X_train, maxlen=maxlen)
    X_test = sq.pad_sequences(X_test, maxlen=maxlen)
    X_val = sq.pad_sequences(X_val, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_val shape:', X_val.shape)
    print('X_test shape:', X_test.shape)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()

    input_layer = Input(shape=(maxlen,),dtype='int32', name='main_input')
    emb_layer = Embedding(max_features,
                          embedding_dims,
                          input_length=maxlen
                          )(input_layer)
    def max_1d(X):
        return K.max(X, axis=1)

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size 3:

    con3_layer = Convolution1D(nb_filter=nb_filter,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)(emb_layer)

    pool_con3_layer = Lambda(max_1d, output_shape=(nb_filter,))(con3_layer)


    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size 4:

    con4_layer = Convolution1D(nb_filter=nb_filter,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)(emb_layer)

    pool_con4_layer = Lambda(max_1d, output_shape=(nb_filter,))(con4_layer)


    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size 5:

    con5_layer = Convolution1D(nb_filter=nb_filter,
                        filter_length=7,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)(emb_layer)

    pool_con5_layer = Lambda(max_1d, output_shape=(nb_filter,))(con5_layer)


    cnn_layer = merge([pool_con3_layer, pool_con5_layer,pool_con4_layer ], mode='concat')


    #LSTM


    x = Embedding(max_features, embedding_dims, input_length=maxlen)(input_layer)
    lstm_layer = LSTM(128)(x)

    cnn_lstm_layer = merge([lstm_layer, cnn_layer], mode='concat')

    dense_layer = Dense(hidden_dims*2, activation='sigmoid')(cnn_lstm_layer)
    output_layer= Dropout(0.2)(dense_layer)
    output_layer = Dense(3, trainable=True,activation='softmax')(output_layer)




    model = Model(input=[input_layer], output=[output_layer])
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)

    model.compile(loss='categorical_crossentropy',
                  optimizer="adamax",
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint('CNN-LSTM-weights/cv'+str(cv)+'weights.hdf5',
                                 monitor='val_acc', verbose=0, save_best_only=True,
                                 mode='max')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              callbacks=[checkpoint],
              validation_data=(X_val, y_val))

    model.load_weights('CNN-LSTM-weights/cv'+str(cv)+'weights.hdf5')
    model.compile(loss='categorical_crossentropy',
                  optimizer="adamax",
                  metrics=['accuracy'])
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print("Acc:", acc)
    accs.append(acc)
print ("Average Acc:", K.np.mean(K.np.array(accs)))
