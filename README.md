## Traffic Sign Classification
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I will deep neural networks and convolutional neural networks to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out my model on images of German traffic signs that you find on the web.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results

### Dependencies
This lab requires docker images:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
* [gcr.io/tensorflow/tensorflow:latest-gpu](https://tensorflow.google.cn/install/install_linux?hl=zh-cn)

Both of the above two docker images can be used for training model, however, the first is based on CPU and the second based on GPU.

### Dataset

Download the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which the images are already resized to 32x32. It contains a training, validation and test set.


[//]: # (Image References)
[image1]: ./trainingset.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/1.jpg "Traffic Sign 1"
[image5]: ./test_images/2.jpg "Traffic Sign 2"
[image6]: ./test_images/3.jpg "Traffic Sign 3"
[image7]: ./test_images/4.jpg "Traffic Sign 4"
[image8]: ./test_images/5.jpg "Traffic Sign 5"
[image9]: ./test_images/6.jpg "Traffic Sign 6"
[image10]: ./test_images/7.jpg "Traffic Sign 7"
[image11]: ./test_images/8.jpg "Traffic Sign 8"
[image12]: ./test_images/9.jpg "Traffic Sign 9"
[image13]: ./test_images/10.jpg "Traffic Sign 10"

Data Set Summary & Exploration
---
#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:
* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32 ,3)
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a bar chart showing how the data distributed.

![alt text][image1]

Model Architecture
---
#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

I didn't convert the images to grayscale, in my opinion, the RGB has more useful information than gray.  
I normalized the image data, because well distributed data well accelate training operation and increase accuracy.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| Input     | Output      |
|:---------------------:|:---------------------------------------------:|:---------:|:-----------:| 
| Convolution       	| kernel: 5x5; stride:1x1; padding: valid  	    | 32x32x3   | 28x28x9     |
| Max pooling	      	| kernel: 2x2; stride:2x2; padding: valid 	    | 28x28x9   | 14x14x9     |
| Convolution       	| kernel: 3x3; stride:1x1; padding: valid 	    | 14x14x9   | 12x12x32    |
| Max pooling	      	| kernel: 2x2; stride:1x1; padding: valid  		| 12x12x32  | 10x10x32    |
| Convolution       	| kernel: 3x3; stride:1x1; padding: valid 	    | 10x10x32  | 8x8x96      |
| Max pooling	      	| kernel: 2x2; stride:2x2; padding: valid  	    | 8x8x96    | 4x4x96      |
| Flatten				| Input 5x5x32 -> Output 800					| 4x4x96    | 1536        |
| Fully connected		| connect every neurel with next layer 		    | 1536      | 800         |
| Fully connected		| connect every neurel with next layer	        | 800       | 400         |
| Fully connected		| connect every neurel with next layer  		| 400       | 200         |
| Fully connected		| output 43 probabilities for each lablel  		| 200       | 43          |

Training 
---
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
* LEARNING_RATE = 0.0006
* EPOCHS = 35
* BATCH_SIZE = 128

The docker image: udacity/carnd-term1-starter-kit are used for data preprossing.  
The docker image: gcr.io/tensorflow/tensorflow:latest-gpu is used for training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* accuracy of training set: 99.7%
* accuracy of validation set: 96.6%
* accuracy of test set: 94.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  
 My first architecure is LeNet, cause 
* What were some problems with the initial architecture?  
 The LeNet model seem doesn't include enough parameters, that accuracy is low.
* How was the architecture adjusted and why was it adjusted?   
 Firstly, I add a convo layer and a fully connect layer to increase accuracy.
 And, I find that the accuracy on training set is up to 97%, but that of testing set is under 93%, so it must be overfitting.
 Therefore, I add dropout to the first three fully connect layer, and successfully, the accuracy on testing set can reach 95%.
* Which parameters were tuned? How were they adjusted and why?  
 I increase epoch from 10 to 35, cause I think 10 epoch isn't enough to complete training.
 And I also decrease learning rate from 0.01 to 0.006, in case ignore the best point.


Testing
---
#### 1. Choose ten German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13]  

The label for each sign is: [0, 1, 3, 5, 8, 23, 26, 30 ,32, 42]

As we can see, that the 7th figure is overexposed, it can hardly be recognized by eyes. Therefore, the 7th figure may be difficult for the model to classify.   
The 5th and 10 th figure isn't exposed enough or was taken at night, they are not light enough, so they might also be mistaken.   
As for the other figures, I think they are clear enough for the model recognized.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set .

Here are the results of the prediction:

| Image	label		    | Prediction label	       | 
|:---------------------:|:------------------------:| 
| 0      		        | 0   					   | 
| 1    			        | 1 		          	   |
| 3					    | 3					       |
| 5	      	         	| 5				 		   |
| 8			            | 8      		           |
| 23      		        | 23   					   | 
| 26     			    | 26 					   |
| 30					| 30					   |
| 32	      	    	| 32			 		   |
| 42		        	| 42      				   |


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%

#### 3. Describe how certain the model is when predicting on each of the 10 new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The first 5 **logist** for each images are as follows (rounded to xx):

| Image	label   | 1th	     | 2th	     | 3th	     | 4th	      | 5th	      |
|:-------------:|:----------:|:---------:|:---------:| :---------:|:---------:| 
| 0     	    | 0: 26   	 | 8: 21     | 1: 15     | 37: 12     | 4: 11     |
| 1     	    | 1: 23   	 | 2: 6      | 11: 6     | 0: 5       | 18: 4     |
| 3				| 3: 71   	 | 5: 27     | 32: 22    | 2: 8       | 2: 7      |
| 5	      		| 5: 37   	 | 3: 26     | 2: 10     | 4: 9       | 10: 5     |
| 8				| 8: 41   	 | 7: 18     | 5: 9      | 4: 6       | 0: 6      |
| 23        	| 23: 47     | 10: 26    | 20: 13    | 19: 12     | 29: 11    | 
| 26     		| 26: 16  	 | 18: 14    | 22: 4     | 4:2        | 2: 1      |
| 30			| 30: 26  	 | 17: 9     | 26: 6     | 18: 6      | 11: 6     |
| 32	      	| 32: 34  	 | 6: 17     | 1: 11     | 31: 11     | 15: 9     |
| 42			| 42: 42  	 | 41: 24    | 12: 12    | 16: 9      | 6: 7      |
 
The first 5 **softmax probabilities** for each images are as follows (rounded to xx%):

| Image	label   | 1th	     | 2th	     | 3th	     | 4th	      | 5th	      |
|:-------------:|:----------:|:---------:|:---------:| :---------:|:---------:| 
| 0     	    | 0: 100%    | 8: 0%     | 1: 0%     | 37: 0%     | 4: 0%     |
| 1     	    | 1: 100%    | 2: 0%     | 11:0%     | 0: 0%      | 18: 0%    |
| 3				| 3: 100%    | 5: 0%     | 32: 0%    | 2: 0%      | 2: 0%     |
| 5	      		| 5: 100%    | 3: 0%     | 2: 0%     | 4: 0%      | 10: 0%    |
| 8				| 8: 100%    | 7: 0%     | 5: 0%     | 4: 0%      | 0: 0%     |
| 23        	| 23: 100%   | 10: 0%    | 20: 0%    | 19: 0%     | 29: 0%    | 
| 26     		| 26: 100%   | 18: 0%    | 22: 0%    | 4: 0%      | 2: 0%     |
| 30			| 30: 100%   | 17: 0%    | 26: 0%    | 18: 0%     | 11: 0%    |
| 32	      	| 32: 100%   | 6: 0%     | 1: 0%     | 31: 0%     | 15: 0%    |
| 42			| 42: 100%   | 41: 0%    | 12: 0%    | 16: 0%     | 6: 0%     |   


We can see that all the 10 images can be correctly recognized by looking at the softmax probabilities. Which means the model can almost be used for recognizing traffic signs. :)

