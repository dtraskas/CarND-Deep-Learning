#
# Deep Learning Autonomous Car
# Learning Module
#
# Dimitrios Traskas
#
#
import numpy as np
from keras.optimizers import Adam
from preprocessor import PreProcessor
from configurator import Configurator
from modelbuilder import ModelBuilder

if __name__ == '__main__':
    
    # Model definitions that are passed to the configurator
    epochs = 10
    batch_size = 128    
    split_size = 0.1
    image_shape = (160, 320, 3)
    reduced_shape = (80, 160, 3)
    learning_rate = 0.0001
    bTrain = True

    # Check if we are training or doing simple tests on test data
    if (bTrain):
        print('Start training...')

        # prepare the configurator and preprocessor
        configurator = Configurator('udacity_data', image_shape, reduced_shape, batch_size, split_size)
        preprocessor = PreProcessor(configurator)
        preprocessor.load_log("driving_log.csv")

        # essentially splits the data to training and validation sets
        preprocessor.initialise()

        # retrieves the batch generators that are so useful with Keras
        train_generator = preprocessor.get_train_generator()
        validation_generator = preprocessor.get_validation_generator()
        
        # initialise the CNN model in the ModelBuilder
        model_builder = ModelBuilder(configurator)
        model = model_builder.initialise()
        model.load_weights('initials_weights.h5')
        model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=['accuracy'])            
        #model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])            
        history = model.fit_generator(train_generator, samples_per_epoch=preprocessor.get_train_count(), 
                                    nb_epoch=epochs, 
                                    validation_data=validation_generator, nb_val_samples=preprocessor.get_validation_count())     
        model_builder.save_model()
        print('Completed training!')
    else:
        print('Start testing...')

        configurator = Configurator('test_data', image_shape, reduced_shape, batch_size, split_size)
        preprocessor = PreProcessor(configurator)
        preprocessor.load_log("driving_log.csv")

        model_builder = ModelBuilder(configurator)
        model = model_builder.load_model()
        predictions = model.predict(preprocessor.read_images())
        actuals = preprocessor.get_yvalues()

        cnt = 0
        for value in np.nditer(predictions):
            print(str(value) + ' vs. ' + str(actuals[cnt]))
            cnt += 1

        print('Completed testing!')