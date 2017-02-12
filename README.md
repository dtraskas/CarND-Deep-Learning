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

For my model I chose the NVIDIA architecture as mentioned in their recent, with 5 convolutional layers, 4 fully connected layers and one output layer. The NVIDIA architecture utilises images of 66x200 pixels and in my case I am resizing the image from 160x320 to 80x160 so by half. Dropout layers are added to reduce overfitting.


####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

For training and validation data I captured image

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 