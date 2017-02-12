#**Self Driving Car - Udacity**
#**Behavioral Cloning**

###Project Goals
---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

###Running the code 
---

In order to run this code you will need the following:

- [Udacity Simulator](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
- Generated images that are stored in a folder at the root level 
- Install all the Python required packages (numpy, pandas, keras, tensorflow, gevent_socketio, scikit_learn, flask, eventlet)
- The model.json (architecture) and model.h5 (model weights)
- Run ```cmd python model.py ``` in order to train a new model and ```cmd python drive.py model.json ``` while the Udacity simulator is running in order to run in autonomous mode and observe your self-driving car

###Model Architecture and Training Strategy
---

For this project I chose the Keras framework that sits on top of Tensorflow. Keras offers a high level framework and a number of useful features such as Data Generators for image pre-processing, that make the code easier to read, reusable and faster to develop.

For my model I chose the Nvidia architecture as mentioned in their recent published paper (https://arxiv.org/abs/1604.07316) 'End to End Learning for Self-Driving Cars'. My architecture uses 5 convolutional layers, 4 fully connected layers and one output layer. The Nvidia architecture utilises images of 66x200 pixels however in my case I am resizing the images from 160x320 to 80x160 by half. The network uses RELU activation units and convolutional layers with 2x2 stride as well as dropout layers added to reduce overfitting. To further reduce overfitting the model was trained and validated on two different datasets which used a predefined split size.

Each of the convolutional layers increases the depth until the last convolutional layer that gets flattened to 1164 neurons as described in the Nvidia architecture. Subsequently hidden layers of 100, 50, 10 are added until the last output neuron that essentially predicts the steering angle.

In order to simplify the creation of a model and the collection of data I created two main classes, the ModelBuilder and the PreProcessor. The PreProcessor essentially loads the driving log and the images associated with it. 

The model uses an Adam optimizer so the learning rate was not tuned manually. The loss function was set to the mean squared error between predicted and actuals. 

For training and validation data I had real difficulty getting enough good images by driving on both tracks. I experimented heavily with the Udacity provided datasets but unfortunately that did not prove to work very well either. At some point I had a decent model that was working well for the majority of the track except one area after the bridge. In this final submission I am using initial weights loaded from a good initial model and I start to train from that point onwards using extreme scenarios where I always move from the side of the road to the middle. The two datasets provide a good number of examples with center lane driving and recoveries from the left and right sides of the road.

###Data Generators and pre-processing
---

Initially I created my own batch generator that I was populating with images and steering angles every time the training algorithm was requesting for training data. However I wanted to augment the provided data so in the end I utilised the Keras ImageDataGenerator which proved to be really useful. Initially I am resizing the images and then for the training data I shift the images horizontally and vertically by a small amount. I also filter out extreme angles or angles that are very close to zero in order to avoid having a model that is heavily biased to examples with zero angles which are the majority in the datasets. 

