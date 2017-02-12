#
# Deep Learning Autonomous Car
# Model Builder module
#
# Dimitrios Traskas
#
#
import json

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.activations import relu, softmax
from keras.models import model_from_json

#
# The ModelBuilder sets up the neural network and runs the optimization
# 
class ModelBuilder:
    
    # Initialise the ModelBuilder with parameters from the configurator
    def __init__(self, config):
        self.image_shape = config.image_shape
        self.reduced_shape = config.reduced_shape
        self.batch_size = config.batch_size        
        self.split_size = config.split_size        

    # Builds a model using the Keras library
    def initialise(self):
        self.model = Sequential()
        
        self.model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', input_shape=self.reduced_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
        self.model.add(Flatten())    
        self.model.add(Dense(1164, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, init='normal'))

        return self.model

    # Saves the model and the weights
    def save_model(self):
        print('Saving model...')
        json = self.model.to_json()
        self.model.save_weights('model.h5')
        with open('model.json', 'w') as file:
            file.write(json)

    # Loads an existing model and weights
    def load_model(self):
        with open('model.json', 'r') as jfile:
            self.model = model_from_json(jfile.read())

        self.model.compile("adam", "mse")    
        self.model.load_weights('model.h5')
        
        return self.model
        
        