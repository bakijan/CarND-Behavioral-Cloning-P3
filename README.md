# **Behavioral Cloning** 


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
[image2]: ./training_img.png "Training image"
[image6]: ./input_and_cropped.png "Original and Cropped Image"
[image7]: ./learning_curve.png "Learning Curve"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.ipynb code cell [11]) 

The model includes RELU layers to introduce nonlinearity after each convolutional layer with Keras, and the data is normalized in the model using a Keras lambda layer (model.ipynb code cell [10]). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.ipynb code cell [11]). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py code cell [13]).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of images from center, left and right cameres. I offseted left and right camera images by 0.2 to mimicing the adjusting steering wheel when car drive off the center.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different well known architechtures. 

My first step was to use a convolution neural network model similar to the Lenet model. I thought this model might be appropriate because prediction of medel is only one veriable and driving scene is simulated landscape ratherthan real world traffic. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropout layer after last convolutional layer and first and second second fully connected layer. Then ,car able to drive through the bridge autonomous mode.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model is nVIDIA'S End-to-End Deeplearning architecture (model.ipynb code cell [11]) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image 							| 
| Normalization    		| lambda x: (x / 255.0) - 0.5 					| 
| Cropping2D    		| 90x320x3 RGB image 							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 43x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 20x77x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 8x37x48 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 6x35x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 4x33x64 	|
| RELU					|												|
| Dropout				| probability 0.5								|
| Flatten				| Output 8448		    						|
| Fully connected		| Output 100		            				|
| Dropout				| Probability 0.5								|
| Fully connected		| Output 50		                			    |
| Dropout				| Probability 0.5								|
| Fully connected		| Output 10             						|
| Fully connected		| Output 1              						|

#### 3. Creation of the Training Set & Training Process

For this project, I used training data set provided by Udacity which consitant of 24108 images captured by three cameras located at center, left and right. 

![alt text][image2]

Scince buttom and top part of images not usufull for deciding stering direction, I cropped this protion at the first layer of Keras model. Sampel image before and after the cropping looks like this:

![alt text][image6]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by learning curve below. I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image7]