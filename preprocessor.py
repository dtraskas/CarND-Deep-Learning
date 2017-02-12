#
# Deep Learning Autonomous Car
# Data pre-processing module
#
# Dimitrios Traskas
#
#
import numpy as np
import pandas as pd

from skimage import io, transform
from scipy.misc import imread, imresize

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import math
#
# The Data Preprocessor loads all the necessary training data and applies transformations to it
# 
class PreProcessor:

    # Initialise the Preprocessor with parameters from the configurator
    def __init__(self, config):
        self.data_headers = ['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed']
        self.data_path = config.data_path        
        self.image_shape = config.image_shape
        self.reduced_shape = config.reduced_shape
        self.batch_size = config.batch_size        
        self.split_size = config.split_size        

    # Loads the driving log in a Pandas dataframe for easy access
    def load_log(self, filename):
        print("Loading driving log...")        
        self.data = pd.read_csv(self.data_path + "/" + filename, header=None, names=self.data_headers)

    # Prepares training and validation sets
    def initialise(self):        
        print("Splitting to training and validation sets...")        
        self.X_train = np.array(self.data['center'].values)
        self.y_train = np.array(self.data['angle'].values)

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train, self.y_train, test_size=self.split_size, random_state=12)

    # Returns the batch generator with training data
    def get_train_generator(self):
        generator = ImageDataGenerator()        
        height, width, channels = self.image_shape        
        image_paths = self.prepare_paths(self.X_train)
        image_array = np.empty((len(image_paths), height, width, channels), dtype=np.uint8)        
        angles = np.empty_like(self.y_train)

        for cnt, filename in enumerate(image_paths):                        
            angle = self.y_train[cnt]    
            if (angle > -0.95 and angle < 0.95 and angle != 0):
                image = self.read_image(self.data_path + "/IMG/" + filename) 
                if np.random.choice(2):
                    image_array[cnt] = np.fliplr(image)
                    angles[cnt] = angle * -1
                else:
                    image_array[cnt] = image 
                    angles[cnt] = angle

        image_array = self.resize(image_array)
        return generator.flow(image_array, angles, batch_size=self.batch_size)

        #return generator.flow(image_array, self.y_train, batch_size=self.batch_size)
        #return self.batch_generator(self.X_train, self.y_train)

    # Returns the batch generator with validation data
    def get_validation_generator(self):
        generator = ImageDataGenerator()
        height, width, channels = self.image_shape
        image_paths = self.prepare_paths(self.X_validation)
        image_array = np.empty((len(image_paths), height, width, channels), dtype=np.uint8)
        
        for cnt, filename in enumerate(image_paths):          

            image_array[cnt] = self.read_image(self.data_path + "/IMG/" + filename)

        image_array = self.resize(image_array)
        return generator.flow(image_array, self.y_validation, batch_size=self.batch_size)
        #return self.batch_generator(self.X_validation, self.y_validation)

    # Returns the count of training data
    def get_train_count(self):
        return len(self.X_train)
    
    # Returns the count of validation data
    def get_validation_count(self):
        return len(self.X_validation)

    # Reads an image from the filename specified 
    def read_image(self, filename):
        return imread(filename)            

    # Resizes an array of images to the shape provided
    def resize(self, images):
        height, width, channels = self.reduced_shape
        resized = []
        for cnt, image in enumerate(images):
            resized.append(transform.resize(image, (height, width)))    
        return np.array(resized, dtype='float32')

    # Creates an image generator to be used by the model
    def batch_generator(self, image_list, angles):  
        image_count = len(image_list)
        height, width, channels = self.image_shape
        while True:        
            index_list = np.random.choice(image_count, self.batch_size)
            image_array = np.empty((self.batch_size, height, width, channels), dtype=np.uint8)
            angle_array = angles[index_list].astype(float)

            for cnt, filename in enumerate(image_list[index_list]):                        
                image_array[cnt] = self.read_image(self.data_path + "/IMG/" + filename)

            image_array = self.resize(image_array)        
            yield image_array, angle_array

    # Used for testing a model
    def read_images(self):
        height, width, channels = self.image_shape
        image_paths = np.array(self.data['center'].values)
        image_paths = self.prepare_paths(image_paths)
        image_array = np.empty((len(image_paths), height, width, channels), dtype=np.uint8)
        
        for cnt, filename in enumerate(image_paths):                        
            image_array[cnt] = self.read_image(self.data_path + "/IMG/" + filename)

        image_array = self.resize(image_array)
          
        return image_array

    # Strips out any unecessary paths and standardises the path format
    def prepare_paths(self, image_list):
        prepared_paths = np.array(image_list)
        for cnt, filename in enumerate(image_list):            
            prepared_paths[cnt] = filename.rsplit("/")[-1]

        return prepared_paths

    # Returns the x values
    def get_xvalues(self):
        return np.array(self.data['center'].values)

    # Returns the y values    
    def get_yvalues(self):
        return np.array(self.data['angle'].values)