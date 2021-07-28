# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on NVIDIA's End to End Learning for Self-Driving Cars and utilizes a convolutional neural network Which has a depth of 5 Convs. 

The model includes RELU layers to introduce nonlinearity (code line 87), and the data is normalized in the model using a Keras lambda layer (code line 72). 

#### 2. Attempts to reduce overfitting in the model

To minimize overfitting, the model includes numerous dropout layers between the dense layers (first example: model.py lines 106). These layers decrease overfitting by removing outputs from a layer of a neural network at random, forcing more neurons to learn the network's features

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was 0.0001 (model.py line 87).

#### 4. Appropriate training data

For training my model, I used the data provided by Udacity as a starting point. The automobile drifted off the track at the first tight turn due to the model architecture and data provided. As a result, I determined it would be preferable to include some additional training data focusing on drift recovery.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVIDIA End to End Learning for Self-Driving Cars I thought this model might be appropriate because it was suggested by the instructors 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding a dropout layers

then i captured some data and flipped it and randomize it 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

```
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

#model.add(... finish defining the rest of your model architecture here ...)
model.add(Cropping2D(cropping=((65, 20), (0, 0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

#model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center][home/CarND-Behavioral-Cloning-P3/center_2016_12_01_13_30_48_404.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to always be near the center These images show what a recovery looks like starting from ... :

![alt text][home/CarND-Behavioral-Cloning-P3/left_2016_12_01_13_33_25_551.jpg]
![alt text][home/CarND-Behavioral-Cloning-P3/right_2016_12_01_13_42_58_402.jpg]


Then I repeated this process on track two in order to get more data points.


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 . I used an adam optimizer so that manually training the learning rate wasn't necessary.
