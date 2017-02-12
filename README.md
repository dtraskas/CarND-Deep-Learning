#**Self Driving Car - Udacity**
#**Behavioral Cloning**

###Project Goals

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

###Model Architecture and Training Strategy
For this project I chose the Keras framework that sits on top of Tensorflow. Keras offers a high level framework and a number of useful features such as Data Generators for image pre-processing, that make the code easier to read, reusable and faster to develop.

For my model I chose the Nvidia architecture as mentioned in their recent published paper (https://arxiv.org/abs/1604.07316) 'End to End Learning for Self-Driving Cars'. My architecture uses 5 convolutional layers, 4 fully connected layers and one output layer. The Nvidia architecture utilises images of 66x200 pixels however in my case I am resizing the images from 160x320 to 80x160 by half. The network uses RELU activation units and convolutional layers with 2x2 stride as well as dropout layers added to reduce overfitting. To further reduce overfitting the model was trained and validated on two different datasets which used a predefined split size.

Each of the convolutional layers increases the depth until the last convolutional layer that gets flattened to 1164 neurons as described in the Nvidia architecture. Subsequently hidden layers of 100, 50, 10 are added until the last output neuron that essentially predicts the steering angle.

In order to simplify the creation of a model and the collection of data I created two main classes, the ModelBuilder and the PreProcessor. The PreProcessor essentially loads the driving log and the images associated with it. 

The model uses an Adam optimizer with a specified learning rate that has proven in my experiments to work well. The loss function was set to the mean squared error between predicted and actuals. 


####4. Appropriate training data

For training and validation data I captured image

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 