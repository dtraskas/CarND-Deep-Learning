#
# Deep Learning Autonomous Car
#
#
#
import numpy as np
import pandas as pd
import cv2, os, json

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.activations import relu, softmax

# Definitions for the size of the images loaded and channels available
WIDTH = 320
HEIGHT = 160
CHANNELS = 3   

# Loads all the image data from jpg files (used for testing)
def read_images(path):    
    sample_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])    
    image_array = np.ndarray((sample_count, HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)     
    cnt = 0
    for filename in os.listdir(path):
        fullFilename = os.path.join(path,filename)        
        image = readImage(fullFilename)
        if image is not None:
            image_array[cnt] = image
            cnt += 1    
    return image_array

# Reads an image from the filename specified 
def read_image(filename):
    return cv2.imread(filename, cv2.IMREAD_COLOR)    

# Loads the driving log (used for testing)
def load_log(filename):
    data = pd.read_csv(filename, header=None, names=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])
    return data

# Creates an image generator to be used by the model
def image_batch_generator(path, batch_size):    
    print("Loading driving log...")
    driving_log = load_log(path + '/driving_log.csv')
    while True:
        cnt = 0
        image_array = np.ndarray((batch_size, HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
        angle = np.zeros(batch_size)
        for index, row in driving_log.iterrows():    
            filename = row['center']
            new_filename = filename.split('/IMG/')[1]
            image = readImage(path + "/IMG/" + new_filename)
            if image is not None:
                image_array[cnt] = image            
                angle[cnt] = row['angle']

            cnt += 1        
            if cnt == batch_size:
                yield(image_array, angle)
                cnt = 0

# Builds a model
def build_model():
    
    model = Sequential()
    model.add(Convolution2D(3, 3, 3, border_mode='valid', activation='relu', input_shape=(HEIGHT, WIDTH, CHANNELS)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(43))
    model.add(Activation('softmax'))
    return model

def train_model(model):
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    checkpointer = ModelCheckpoint(filepath="mode.h5", verbose=1, save_best_only=True)

    history = model.fit_generator(image_batch_generator('track_one_data', 100), samples_per_epoch=100, nb_epoch=10, callbacks=[checkpointer])

if __name__ == '__main__':
    
    model = build_model()
    print(model.summary())
    train_model(model)

    