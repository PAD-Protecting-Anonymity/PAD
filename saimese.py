from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.

def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (24, )
left_input = Input(input_shape)
right_input = Input(input_shape)
kernels = 100
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Dense(kernels, activation='relu', input_shape=input_shape, kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(Dense(kernels, activation='relu', kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(Dense(kernels, activation='sigmoid', kernel_initializer=W_init,kernel_regularizer=l2(1e-3)))

#encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#merge two encoded inputs with the l1 distance between them

L1_distance = lambda x: K.abs(x[0]-x[1])

both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])

prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)

siamese_net = Model(input=[left_input,right_input],output=prediction)
#optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

optimizer = Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

siamese_net.count_params()