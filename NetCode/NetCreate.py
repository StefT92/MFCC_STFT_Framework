"""
PARAMETRIC NEURAL NETWORK GENERATOR

This class can be called to generate neural networks structures with varying input parameters.
NN TYPES:   CNN, MLP, ...
POST PROCESSING UTILITIES.

Requires Keras v1.1.2 or higher

Author: S.Tomassetti <tomassetti.ste@gmail.com>, 2017
"""

from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Embedding, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Merge, LSTM, Convolution1D, MaxPooling1D,UpSampling1D
from keras.optimizers import SGD, Adadelta, Adam, Adamax
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import cross_validation

import numpy as np

import time


def ctx(data, ctx_fr=10, RESHAPE=True):
    """
    Context creation for MLP network

    :param data: Data to contextualize with shape [num_utterance, freq_shape, time_shape]
    :param ctx_fr: Number of context frames. Total context shape will be 2*ctx_fr+1
    :param RESHAPE: if True method will perform automatic reshape on data
    :return: data, ctx_size
    """
    ctx_size = 2 * ctx_fr + 1

    ret = []
    for utt in data:
        l = utt.shape[0]
        ur = []
        for t in range(l):
            f = []
            for s in range(t - ctx_fr, t + ctx_fr + 1):
                if (s < 0):
                    s = 0
                if (s >= l):
                    s = l - 1
                f.append(utt[s, :])
            ur.append(f)
        ret.append(np.array(ur))

    ret = np.array(ret)

    if RESHAPE:

       ret = np.reshape(ret, [-1,data.shape[1]*ctx_size])

    return np.array(ret), ctx_size


def mainMLP(N_Layers, features_length, layer_dim, activations, final_layer_activation, batchnormalization, dropout_vec, learn_rate,decay,MomentumMax,optimizer):

    """MLP Creator:

    :param N_layers: number of MLP layers
    :param features_length: input dim of fitst layer
    :param layer_dim: layers dimension
    :param activations:
    :param batchnormalization: -1 or 1, if 1 features are normalized batch by batch
    :param dropout_vec: dropout vector
    :param learn_rate
    :param decay
    :param MomentumMax
    :param optimizer

    :return model
    """
    model = Sequential()
    for i in xrange(N_Layers-1):
        if i == 0:
            input_dim = features_length
        else:
            input_dim = layer_dim[i - 1]
        model.add(Dense(input_dim=input_dim, output_dim=layer_dim[i], init='normal'))
        model.add(Activation(activations))
        if len(dropout_vec) > i and dropout_vec[i] > 0:
            model.add(Dropout(dropout_vec[i]))
        if batchnormalization != -1:
            model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

    model.add(Dense(output_dim=layer_dim[len(layer_dim)-1], init='normal'))
    model.add(Activation(final_layer_activation))
    # if batchnormalization != -1:
    #     model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

    if optimizer == 'sgd':
        opt = SGD(lr=learn_rate, decay=decay, momentum=MomentumMax, nesterov=True)
    elif optimizer == 'adamax':
        opt = Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    elif optimizer == 'adam':
        opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)


    model.compile(loss='mse', optimizer=opt)

    return model


def mainMLP_custom_cost(N_Layers, features_length, layer_dim, activations, final_layer_ativation, batchnormalization, dropout_vec, learn_rate,decay,MomentumMax,optimizer, CUSTOM_COST_FUNCTION=True, custom_cost_function_name=''):
    """
    MLP create using custom cost function defined by use
    :param N_Layers:
    :param features_length:
    :param layer_dim:
    :param activations:
    :param final_layer_ativation:
    :param batchnormalization:
    :param dropout_vec:
    :param learn_rate:
    :param decay:
    :param MomentumMax:
    :param optimizer:
    :param CUSTOM_COST_FUNCTION:
    :return:
    """
    model = Sequential()
    for i in xrange(N_Layers-1):
        if i == 0:
            input_dim = features_length
        else:
            input_dim = layer_dim[i - 1]
        model.add(Dense(input_dim=input_dim, output_dim=layer_dim[i], init='normal'))
        model.add(Activation(activations))
        if len(dropout_vec) > i and dropout_vec[i] > 0:
            model.add(Dropout(dropout_vec[i]))
        if batchnormalization != -1:
            model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
    model.add(Dense(output_dim=layer_dim[len(layer_dim)-1], init='normal'))
    model.add(Activation(final_layer_ativation))

    # if batchnormalization != -1:
    #     model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

    if optimizer == 'sgd':
        opt = SGD(lr=learn_rate, decay=decay, momentum=MomentumMax, nesterov=True)
    elif optimizer == 'adamax':
        opt = Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    elif optimizer == 'adam':
        opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)


    if CUSTOM_COST_FUNCTION:
        model.compile(loss=custom_cost_function_name, optimizer=opt)
    else:
        model.compile(loss='mse', optimizer=opt)

    return model


def mainCNN(CNN_layerSize, MLP_layerSize, optimizer, activations, final_layer_activation, dropout_vec, dropout_vec_MLP, learn_rate, MomentumMax, decay, poolsize, receptiveField, tensor_shape,last_max_pooling_over_time,batchnormalization):

    """ CNN Creator

    :param CNN_layerSize: Sizes of con layers
    :param MLP_layerSize: Sizes of MLP layers
    :param optimizer:
    :param activations:
    :param dropout_vec: dropout vector CNN layers
    :param dropout_vec_MLP: dropout vector MLP layers
    :param learn_rate:
    :param MomentumMax:
    :param decay:
    :param poolsize:
    :param receptiveField:
    :param tensor_shape:
    :param last_max_pooling_over_time: Max pooling along time dimension at the end of conv layers
    :param batchnormalization:

    :return: model

    """
    N_Layers = len(CNN_layerSize)
    model = Sequential()

    for i in xrange(N_Layers):

        if i == 0:
            model.add(Convolution2D(CNN_layerSize[i], receptiveField[i][0], receptiveField[i][1], border_mode='valid',
                                    init='normal', input_shape=(1, tensor_shape[2], tensor_shape[3])))
        else:
            model.add(Convolution2D(CNN_layerSize[i], receptiveField[i][0], receptiveField[i][1], init='normal'))

        model.add(Activation(activations))

        if len(dropout_vec) > i and dropout_vec[i] > 0:
            model.add(Dropout(dropout_vec[i]))
        if batchnormalization != -1:
            model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
        if poolsize[i] != [-1, -1]:
            model.add(MaxPooling2D(pool_size=(poolsize[i][0], poolsize[i][1])))

    if last_max_pooling_over_time != -1:
        model.add(MaxPooling2D(pool_size=(1, model.layers[-1].output_shape[3])))
    model.add(Flatten())
    # Note: Keras does automatic shape inference.

    for i in xrange(len(MLP_layerSize)-1):

        model.add(Dense(output_dim=MLP_layerSize[i], init='normal'))
        model.add(Activation(activations))
        if len(dropout_vec_MLP) > i and dropout_vec_MLP[i] > 0:
            model.add(Dropout(dropout_vec_MLP[i]))
        if batchnormalization != -1:
            model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

    model.add(Dense(output_dim=MLP_layerSize[len(MLP_layerSize) - 1], init='normal'))
    model.add(Activation(final_layer_activation))

    if optimizer == 'sgd':
        opt = SGD(lr=learn_rate, decay=decay, momentum=MomentumMax, nesterov=True)
    elif optimizer == 'adamax':
        opt = Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    elif optimizer == 'adam':
        opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)

    model.compile(loss='mse', optimizer=opt)

    return model


def MainAE(N_Layers, features_length, layer_dim, activations, batchnormalization, learn_rate, decay, MomentumMax, optimizer):

    model = Sequential()

    for i in xrange(N_Layers):
        if i == 0:
            input_dim = features_length
            model.add(LSTM(output_dim=layer_dim[i], input_shape=(None,input_dim), return_sequences=True, activation=activations))
            if batchnormalization != -1:
                model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
        else:

            model.add(LSTM(output_dim=layer_dim[i], return_sequences=True, activation=activations))
            if batchnormalization != -1:
                model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

    model.add(LSTM(layer_dim[len(layer_dim)-1], activation=activations, return_sequences=False))
    if batchnormalization != -1:
        model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
    model.add(Dense(features_length, activation='linear'))

    if optimizer == 'sgd':
        opt = SGD(lr=learn_rate, decay=decay, momentum=MomentumMax, nesterov=True)
    elif optimizer == 'adamax':
        opt = Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    elif optimizer == 'adam':
        opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)


    model.compile(optimizer=opt, loss='mse')

    return model


def MainAE_CNN(N_Layers_Conv, N_layers_MLP, features_length, layer_dim, maxPooling, N_filters, CNN_kernel_sizes,
               activations, batchnormalization, learn_rate, decay, MomentumMax, optimizer):

    model = Sequential()

    for i in xrange(0, N_Layers_Conv):
        if i == 0:
            model.add(Convolution1D(nb_filter=N_filters[0], filter_length=CNN_kernel_sizes[0],
                                    activation=activations, init="uniform", border_mode="same",
                                    input_shape=(features_length, 1)))
        else:
            model.add(Convolution1D(N_filters[i], CNN_kernel_sizes[i], activation=activations,
                                    init='uniform', border_mode='same'))

        if maxPooling[i] != 0:
            model.add(MaxPooling1D(maxPooling[i], border_mode='same'))

        if batchnormalization != -1:
            model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

    if N_layers_MLP > 0:

        in_flatten_dim = model.output_shape
        model.add(Flatten())
        intermediate_dim = model.output_shape[-1]

        for i in xrange(0, N_layers_MLP):
            model.add((Dense(layer_dim[i], activation=activations, init='normal')))
            if batchnormalization != -1:
                model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

        for i in reversed(xrange(0, N_layers_MLP)):
            model.add((Dense(layer_dim[i], activation=activations, init='normal')))
            if batchnormalization != -1:
                model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

        model.add((Dense(intermediate_dim, activation=activations, init='normal')))
        model.add(Reshape(target_shape=(in_flatten_dim[1],in_flatten_dim[2])))

    for i in reversed(xrange(0, N_Layers_Conv)):

        model.add(Convolution1D(N_filters[i], CNN_kernel_sizes[i], activation=activations, border_mode='same'))

        if batchnormalization != -1:
            model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))

        if maxPooling[i] != 0:
            model.add(UpSampling1D(maxPooling[i]))

    model.add(Flatten())

    model.add(Dense(output_dim=features_length, activation='linear'))

    if optimizer == 'sgd':
        opt = SGD(lr=learn_rate, decay=decay, momentum=MomentumMax, nesterov=True)
    elif optimizer == 'adamax':
        opt = Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    elif optimizer == 'adam':
        opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)

    model.compile(optimizer=opt, loss='mse')

    return model


def training_testing(X_train, Y_train, X_test, model, TRAIN_TEST, trained_network, ES_epochs, minibatch_size, maxEpochs, VALIDATION, LOAD_MODEL=True, CUSTOM_COST_FUNCTION=False, custom_cost_function_name=''):

    """Training and Testing Method, saving and reloading model in .yaml and weights in .h5

    :param X_train:
    :param Y_train:
    :param X_test:
    :param model:
    :param TRAIN_TEST: 0 for TEST 1 for TRAIN
    :param trained_network: Trained Net PATH
    :param ES_epochs: early stopping epochs
    :param minibatch_size: mini batch size
    :param maxEpochs: number of epochs
    :param VALIDATION: if 1 TRAINING set is splitted in TRAIN and VAL
    :param LOAD_MODEL: if true, in testing we reload a .yaml file from trained_netowrk path

    :return: network_test_output, history
    """
    history = []
    if TRAIN_TEST == 1:
        begin_time = time.time()

        if VALIDATION == 1:

            X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

            print ('\nTraining on %i samples' % X_train.shape[0])
            print ('Validating on %i samples' % X_val.shape[0])
            print ('Testing on %i samples\n' % X_test.shape[0])

            checkpointer = ModelCheckpoint(filepath=trained_network+'.h5', verbose=1, save_best_only=True)
            ES = EarlyStopping(monitor='val_loss', patience=ES_epochs, verbose=0)
            history = model.fit(X_train, Y_train, validation_data=[X_val,Y_val], batch_size=minibatch_size, nb_epoch=maxEpochs, callbacks=[ES, checkpointer], verbose=2)

        else:
            print ('Training on %i samples' % X_train.shape[0])
            print ('Testing on %i samples\n' % X_test.shape[0])

            history = model.fit(X_train,Y_train,batch_size=minibatch_size, nb_epoch=maxEpochs, verbose=2)

        print "Training  done: it's about to start the test phase."
        end_time = time.time()
        print('Training took %f minutes' % ((end_time - begin_time) / 60.0))
        print "saving the weights and TESTING..."
        model_yaml = model.to_yaml()
        with open(trained_network + '.yaml', "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights(trained_network + '.h5', overwrite=True)
        print("Saved model to disk")
        # model.save_weights(trained_network, overwrite=True)
        network_test_output = model.predict(X_test, batch_size=1, verbose=2)
    else:
        if LOAD_MODEL:
            # load YAML and create model
            yaml_file = open(trained_network + '.yaml', 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            if CUSTOM_COST_FUNCTION:
                loaded_model = model_from_yaml(loaded_model_yaml,
                                               custom_objects={"custom_cost_function": custom_cost_function_name})
            else:
                loaded_model = model_from_yaml(loaded_model_yaml)

            # load weights into new model
            loaded_model.load_weights(trained_network + '.h5')
            print("Loaded model from disk")
            network_test_output = loaded_model.predict(X_test, batch_size=1, verbose=2)
        else:
            model.load_weights(trained_network + '.h5')
            network_test_output = model.predict(X_test, batch_size=1, verbose=2)
            model_yaml = model.to_yaml()
            with open(trained_network + '.yaml', "w") as yaml_file:
                yaml_file.write(model_yaml)
    # model.summary()
    return network_test_output, history