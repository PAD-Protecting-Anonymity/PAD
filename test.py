import numpy as np

a = [0,0,1,1]
b = [0,1,0,1]
c = [0,1,0,1]
arr = np.array([a,b,c])
y = np.array([0,1,0])
x1 = arr
x2 = arr

data = np.array(np.dstack((x1, x2)))
print(data)

# output_shape=(1,) + input_shape[2:]
# test = [lambda x: x[:,:,:,0])(data)

test = data[:,:,0]
test1 = data[:,:,1]
print(data.shape, arr.shape, y.shape, test.shape, test1.shape)


# # Use scikit-learn to grid search the learning rate and momentum
# import copy
# import numpy
# import numpy as np
# import numpy.random as rng
# import pandas as pd
# from sklearn.model_selection import GridSearchCV
# import keras
# from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.optimizers import SGD, RMSprop, Adam
# from keras.layers import Dense, Dropout, Activation, Add, Merge, Input, merge
# from keras.preprocessing import sequence
# from keras import backend as K
# from keras.callbacks import ModelCheckpoint
# from keras.utils.generic_utils import get_custom_objects
# import pdb


# class Model_Tuning:
                
#         def __init__(self, data_pairs, similarity_labels):
#                 self.input_shape = data_pairs[0][0].shape
#                 self.kernels = 100
#                 X1 = []
#                 X2 = []
#                 for x, y in data_pairs:
#                         X1.append(x)
#                         X2.append(y)
#                 self.number_classes = len(np.unique(similarity_labels)) 
#                 train_portion = 0.8

#                 kernels = 100
#                 s_size = len(similarity_labels)
#                 self.x1_train = X1[:int(s_size * train_portion)]
#                 self.x2_train = X2[:int(s_size * train_portion)]
#                 self.x1_test = X1[int(s_size * train_portion):]
#                 self.x2_test = X2[int(s_size * train_portion):]
#                 self.y_train = similarity_labels[:int(s_size * train_portion)]
#                 self.y_test = similarity_labels[int(s_size * train_portion): ]

#                 # labels = keras.utils.to_categorical(similarity_labels, self.number_classes)
#                 # self.y_train = keras.utils.to_categorical(self.y_train, self.number_classes)
#                 # self.y_test = keras.utils.to_categorical(self.y_test, self.number_classes)

#                 self.x1_train = np.array(self.x1_train)
#                 self.x2_train = np.array(self.x2_train)
#                 self.y_train = np.array(self.y_train)

#                 self.x1_test = np.array(self.x1_test)
#                 self.x2_test = np.array(self.x2_test)
#                 self.y_test = np.array(self.y_test)



#         def create_modell(self, neurons=1, learn_rate=0.01):
#                 model = Sequential()
#                 model.add(Dense(self.kernels, input_shape=(448, 96, 2, )))
#                 get_custom_objects().update({'custom_activation': Activation(self.custom_activation)})
#                 model.add(Activation(self.custom_activation))
#                 model.add(Dense(2,activation='sigmoid'))
#                 optimizer = RMSprop()
#                 siamese_net.compile(loss=self.contrastive_loss, optimizer=optimizer)
#                 return siamese_net

#         def create_model(self, neurons=1, learn_rate=0.01):
#                 # create model
#                 model = Sequential()
#                 model.add(Dense(neurons, input_dim=self.input_shape[0], activation='relu'))
#                 model.add(Dense(2,  activation='sigmoid'))
#                 # Compile model
#                 optimizer = SGD(lr=learn_rate)
#                 model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#                 return model


#         def get_best_param(self):
#                 # fix random seed for reproducibility
#                 seed = 7
#                 numpy.random.seed(seed)
#                 # create model
#                 model = KerasClassifier(build_fn=self.create_modell, epochs=100, batch_size=10, verbose=0)
#                 # define the grid search parameters
                
#                 learn_rate = [0.001, 0.01, 0.1]
#                 neurons = [10, 5, 10, 15]
#                 batch_size = [10, 15, 20, 25, 30]
#                 epochs = [10, 50, 100, 200]

#                 param_grid = dict(neurons=neurons, learn_rate=learn_rate)
#                 grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#                 x1 = self.x1_train
#                 x2 = self.x1_train
                
#                 self.maxlen = self.input_shape[0]
#                 x1 = sequence.pad_sequences(x1, maxlen=self.maxlen)
#                 x2 = sequence.pad_sequences(x2, maxlen=self.maxlen)
        
#                 data = np.array(np.dstack((x1, x2)))
#                 print(data.shape, self.y_train.shape)
#                 # exit()
#                 grid_result = grid.fit(data, self.y_train)

#                 # summarize results
#                 print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#                 means = grid_result.cv_results_['mean_test_score']
#                 # stds = grid_result.cv_results_['std_test_score']
#                 params = grid_result.cv_results_['params']
#                 for mean, stdev, param in zip(means, stds, params):
#                         print("%f (%f) with: %r" % (mean, stdev, param))

#         def contrastive_loss(self, y_true, y_pred):
#                 margin = 1
#                 return K.mean((1-y_true) * 0.5 * K.square(y_pred) + 0.5 * y_true * K.square(K.maximum(margin - y_pred, 0))) 

#         def custom_activation(self, x):
#                 # pdb.set_trace()
#                 # print(x)
#                 x1 = x[:,0,:]
#                 x2 = x[:,1,:]
#                 return K.abs(x1-x2)




# # mt = Model_Tuning(data_pairs, similarity_labels)
# # met.get_best_param()





# Use scikit-learn to grid search the learning rate and momentum
import copy
import numpy
import numpy as np
import numpy.random as rng
import pandas as pd
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Add, Merge, Input, merge
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import ModelCheckpoint

class Model_Tuning:
                
        def __init__(self, data_pairs, similarity_labels):
                self.input_shape = data_pairs[0][0].shape
                self.kernels = 100
                X1 = []
                X2 = []
                for x, y in data_pairs:
                        X1.append(x)
                        X2.append(y)
                self.number_classes = len(np.unique(similarity_labels)) 
                train_portion = 0.8

                kernels = 100
                s_size = len(similarity_labels)
                self.x1_train = X1[:int(s_size * train_portion)]
                self.x2_train = X2[:int(s_size * train_portion)]
                self.x1_test = X1[int(s_size * train_portion):]
                self.x2_test = X2[int(s_size * train_portion):]
                self.y_train = similarity_labels[:int(s_size * train_portion)]
                self.y_test = similarity_labels[int(s_size * train_portion): ]

                # labels = keras.utils.to_categorical(similarity_labels, self.number_classes)
                # self.y_train = keras.utils.to_categorical(self.y_train, self.number_classes)
                # self.y_test = keras.utils.to_categorical(self.y_test, self.number_classes)

                self.x1_train = np.array(self.x1_train)
                self.x2_train = np.array(self.x2_train)
                self.y_train = np.array(self.y_train)

                self.x1_test = np.array(self.x1_test)
                self.x2_test = np.array(self.x2_test)
                self.y_test = np.array(self.y_test)



        def create_modell(self, neurons=1, learn_rate=0.01):
                left_input = Input(self.input_shape)
                right_input = Input(self.input_shape)

                model = Sequential()
                model.add(Dense(self.kernels, input_shape=(448, 96, 2,)))
                
                x1 = Lambda(lambda x: x[:,:,:,0], output_shape=(1,) + input_shape[1:2])(model)
                x2 = Lambda(lambda x: x[:,:,:,1], output_shape=(1,) + input_shape[1:2])(model)

                model = Sequential()
                model.add(Dense(kernels, activation='relu', input_shape = input_shape))
                model.add(Dense(kernels, activation='relu'))
                model.add(Dense(kernels, activation='sigmoid'))

                encoded_l = model(x1)
                encoded_r = model(x2)

                L1_distance = lambda x: K.abs(x[0]-x[1])
                both = merge([encoded_l, encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
                prediction = Dense(number_classes,activation='sigmoid')(both)
                siamese_net = Model(input=[left_input,right_input],output=prediction)
                optimizer = RMSprop()
                siamese_net.compile(loss=self.contrastive_loss, optimizer=optimizer)
                return siamese_net

        def create_model(self, neurons=1, learn_rate=0.01):
                # create model
                model = Sequential()
                model.add(Dense(neurons, input_dim=self.input_shape[0], activation='relu'))
                model.add(Dense(2,  activation='sigmoid'))
                # Compile model
                optimizer = SGD(lr=learn_rate)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                return model


        def get_best_param(self):
                # fix random seed for reproducibility
                seed = 7
                numpy.random.seed(seed)
                # create model
                model = KerasClassifier(build_fn=self.create_modell, epochs=100, batch_size=10, verbose=0)
                # define the grid search parameters
                
                learn_rate = [0.001, 0.01, 0.1]
                neurons = [10, 5, 10, 15]
                batch_size = [10, 15, 20, 25, 30]
                epochs = [10, 50, 100, 200]

                param_grid = dict(neurons=neurons, learn_rate=learn_rate)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
                x1 = self.x1_train
                x2 = self.x1_train
                
                self.maxlen = self.input_shape[0]
                x1 = sequence.pad_sequences(x1, maxlen=self.maxlen)
                x2 = sequence.pad_sequences(x2, maxlen=self.maxlen)
        
                data = np.array(np.dstack((x1, x2)))
                print(data.shape, self.y_train.shape)
                # exit()
                grid_result = grid.fit(data, self.y_train)

                # summarize results
                print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                means = grid_result.cv_results_['mean_test_score']
                # stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                        print("%f (%f) with: %r" % (mean, stdev, param))

        def contrastive_loss(self, y_true, y_pred):
                margin = 1
                return K.mean((1-y_true) * 0.5 * K.square(y_pred) + 0.5 * y_true * K.square(K.maximum(margin - y_pred, 0))) 


# mt = Model_Tuning(data_pairs, similarity_labels)
# met.get_best_param()





# Use scikit-learn to grid search the learning rate and momentum
import copy
import numpy
import numpy as np
import numpy.random as rng
import pandas as pd
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Add, Merge, Input, merge, Lambda
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import ModelCheckpoint

class Model_Tuning:
                
        def __init__(self, data_pairs, similarity_labels):
                self.input_shape = data_pairs[0][0].shape
                self.kernels = 100
                X1 = []
                X2 = []
                for x, y in data_pairs:
                        X1.append(x)
                        X2.append(y)
                self.number_classes = len(np.unique(similarity_labels)) 
                train_portion = 0.8

                kernels = 100
                s_size = len(similarity_labels)
                self.x1_train = X1[:int(s_size * train_portion)]
                self.x2_train = X2[:int(s_size * train_portion)]
                self.x1_test = X1[int(s_size * train_portion):]
                self.x2_test = X2[int(s_size * train_portion):]
                self.y_train = similarity_labels[:int(s_size * train_portion)]
                self.y_test = similarity_labels[int(s_size * train_portion): ]

                # labels = keras.utils.to_categorical(similarity_labels, self.number_classes)
                # self.y_train = keras.utils.to_categorical(self.y_train, self.number_classes)
                # self.y_test = keras.utils.to_categorical(self.y_test, self.number_classes)

                self.x1_train = np.array(self.x1_train)
                self.x2_train = np.array(self.x2_train)
                self.y_train = np.array(self.y_train)

                self.x1_test = np.array(self.x1_test)
                self.x2_test = np.array(self.x2_test)
                self.y_test = np.array(self.y_test)



        def create_modell(self, neurons=1, learn_rate=0.01):
                left_input = Input(self.input_shape)
                right_input = Input(self.input_shape)

                model = Sequential()
                model.add(Dense(self.kernels, input_shape=(2, 448, 96)))
                
                x1 = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(model)
                x2 = Lambda(lambda x: x[:,1,:,:], output_shape=(1,) + input_shape[2:])(model)



                

                model = Sequential()
                model.add(Dense(kernels, activation='relu', input_shape = input_shape))
                model.add(Dense(kernels, activation='relu'))
                model.add(Dense(kernels, activation='sigmoid'))

                encoded_l = model(x1)
                encoded_r = model(x2)

                L1_distance = lambda x: K.abs(x[0]-x[1])
                both = merge([encoded_l, encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
                prediction = Dense(number_classes,activation='sigmoid')(both)
                siamese_net = Model(input=[left_input,right_input],output=prediction)
                optimizer = RMSprop()
                siamese_net.compile(loss=self.contrastive_loss, optimizer=optimizer)
                return siamese_net

        def create_model(self, neurons=1, learn_rate=0.01):
                # create model
                model = Sequential()
                model.add(Dense(neurons, input_dim=self.input_shape[0], activation='relu'))
                model.add(Dense(2,  activation='sigmoid'))
                # Compile model
                optimizer = SGD(lr=learn_rate)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                return model


        def get_best_param(self):
                # fix random seed for reproducibility
                seed = 7
                numpy.random.seed(seed)
                # create model
                model = KerasClassifier(build_fn=self.create_model, epochs=100, batch_size=10, verbose=0)
                # define the grid search parameters
                
                learn_rate = [0.001, 0.01, 0.1]
                neurons = [10, 5, 10, 15]
                batch_size = [10, 15, 20, 25, 30]
                epochs = [10, 50, 100, 200]

                param_grid = dict(neurons=neurons, learn_rate=learn_rate)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
                x1 = self.x1_train
                x2 = self.x1_train
                
                self.maxlen = self.input_shape[0]
                x1 = sequence.pad_sequences(x1, maxlen=self.maxlen)
                x2 = sequence.pad_sequences(x2, maxlen=self.maxlen)
        
                data = np.array(list(zip(x1, x2)))
                print(data.shape, self.y_train.shape)
                # exit()
                grid_result = grid.fit(data, self.y_train)

                # summarize results
                print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                means = grid_result.cv_results_['mean_test_score']
                # stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                        print("%f (%f) with: %r" % (mean, stdev, param))

        def contrastive_loss(self, y_true, y_pred):
                margin = 1
                return K.mean((1-y_true) * 0.5 * K.square(y_pred) + 0.5 * y_true * K.square(K.maximum(margin - y_pred, 0))) 


# mt = Model_Tuning(data_pairs, similarity_labels)
# met.get_best_param()