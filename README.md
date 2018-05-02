# Traffic Sign Classification

Overview
---
In this project, I used deep neural networks and four classic convolutional neural network architectures to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out my model on images of German traffic signs that I find on the web.

The Project
---
The goals / steps of this project are the following:
* Load and explore the data set 
* Realize LeNet architecture and use ReLu, mini-batch gradient descent and dropout. 
* Use AlexNet to recognize traffic signs and use L2 regulization, learning rate decay and data augmentation to optimize it. 
* Analyze the softmax probabilities of the new images
* Summarize the results

### Dependencies
python3.5  
matplotlib (2.1.1)  
opencv-python (3.3.1.11)  
numpy (1.13.3)  
tensorflow-gpu (1.4.1)  
sklearn (0.19.1)  

### Dataset
Download the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which the images are already resized to 32x32. It contains a training, validation and test set.


[//]: # (Image References)
[image1]: ./result_images/trainingset.jpg "Visualization"
[lenet]: ./result_images/lenet.png "lenet"
[alexnet]: ./result_images/alexnet.png "alexnet"
[image2]: ./test_images/1.jpg "Traffic Sign 1"
[image3]: ./test_images/2.jpg "Traffic Sign 2"
[image4]: ./test_images/3.jpg "Traffic Sign 3"
[image5]: ./test_images/4.jpg "Traffic Sign 4"
[image6]: ./test_images/5.jpg "Traffic Sign 5"
[image7]: ./test_images/6.jpg "Traffic Sign 6"
[image8]: ./test_images/7.jpg "Traffic Sign 7"
[image9]: ./test_images/8.jpg "Traffic Sign 8"
[image10]: ./test_images/9.jpg "Traffic Sign 9"
[image11]: ./test_images/10.jpg "Traffic Sign 10"

[Data pre-process.ipynb](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/Data%20pre-process.ipynb)
---

I used the numpy library to calculate summary statistics of the traffic signs data set:
* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32 ,3)
* The number of unique classes/labels in the data set is: 43


Here is an exploratory visualization of the training data set. It is a bar chart showing how the data distributed.

![alt text][image1]


I didn't convert the images to grayscale as suggest in class, in my opinion, the RGB has more useful information than gray.  
So I normalized the image data from [0, 255] to [-1, 1], because well distributed data well accelate training operation and increase accuracy.


[LeNet.ipynb](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/LeNet.ipynb)
---
The [LeNet](http://219.216.82.193/cache/10/03/yann.lecun.com/b1a1c4acb57f1b447bfe36e103910875/lecun-01a.pdf) model is proposed by Yann LeCun in 1998, it is the most classific cnn model for image recognition, its architecture is as following: 

![alt text][lenet]

In the LeNet architecture I realized for traffic signs recognition, three tricks as used as follows:

- 1 ReLu  
ReLu nonlinear function is used as the activation function after the convolutional layer. More information about ReLu and other activation functions can be find at [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M&index=6&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&t=0s).  
- 2 Mini-batch gradient descent  
Mini-batch gradient descent is the combine of batch gradient descent and stochastic gradient descent, it is based on the statistics to estimate the average of gradient of all the training data by a batch of selected samples.
- 3 Dropout  
Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. It is proposed in the paper [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://219.216.82.193/cache/2/03/jmlr.org/9b2dcdb089f9b8f19cea175c9d6b5150/srivastava14a.pdf). It is usually after fully connected layers. Awkwardly, there is a very small problem that LeNet will not overfitting to trainging set sometimes. Thus the dropout will not play a big role or even make the model worse for simple like LeNet. And the training set error maybe be higher than validation set error while training.

My LeNet consists of the following layers:

| Layer         		|     Description	        					| Input     | Output      |
|:---------------------:|:---------------------------------------------:|:---------:|:-----------:| 
| Convolution       	| kernel: 5x5; stride:1x1; padding: valid  	    | 32x32x3   | 28x28x6     |
| Max pooling	      	| kernel: 2x2; stride:2x2;               	    | 28x28x6   | 14x14x6     |
| Convolution       	| kernel: 5x5; stride:1x1; padding: valid 	    | 14x14x6   | 10x10x16    |
| Max pooling	      	| kernel: 2x2; stride:2x2;                		| 10x10x16  | 5x5x16      |
| Flatten				| Input 5x5x16 -> Output 400					| 5x5x16    | 400         |
| Fully connected		| connect every neurel with next layer 		    | 400       | 120         |
| Fully connected		| connect every neurel with next layer	        | 120       | 80          |
| Fully connected		| output 43 probabilities for each lablel  		| 80        | 43          |


Training 
---
I have turned the following three hyperparameters to train my model.  
LEARNING_RATE = 1e-2  
EPOCHS = 50  
BATCH_SIZE = 128  
It takes about 2 minutes to train the model on GetForce 750 ti.

The results are:
* accuracy of training set: 96.6%
* accuracy of validation set: 92.0%
* accuracy of test set: 89.7%

We can see that the model is overfitting to the training data and the accuracy on validation set is a little lower than on training set. The LeNet model is efficient and simple, many cnn architectures are inspired by it, like AlexNet.

[AlexNet.ipynb](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/AlexNet.ipynb)
---

[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) is the first popularized CNN architecture in computer vision developed by Alex Krizhevsky, Geoffrey Hinton, and Ilya Sutskever. It is the champion of ImageNet ILSVRC challenge in 2012 and significantly outperformed the second runner-up. The AlexNet has a similar architecture with LeNet, but it is deeper and bigger.

![alt text][alexnet]

Cause the input dimension and output dimension of traffic signs recognition on GTRSB is 32x32x3 and 43, which is different from the original dimension of AlexNet, so I made some change to fit the requirement. And the architecture I realized for recognizing traffic signs as the following table:

| Layer         		|     Description	        					| Input     | Output      |
|:---------------------:|:---------------------------------------------:|:---------:|:-----------:| 
| Convolution       	| kernel: 5x5; stride:1x1; padding: valid  	    | 32x32x3   | 28x28x9     |
| Max pooling	      	| kernel: 2x2; stride:2x2; 	                    | 28x28x9   | 14x14x9     |
| Convolution       	| kernel: 3x3; stride:1x1; padding: valid 	    | 14x14x9   | 12x12x32    |
| Max pooling	      	| kernel: 2x2; stride:2x2;  		            | 12x12x32  | 6x6x32      |
| Convolution       	| kernel: 3x3; stride:1x1; padding: same 	    | 6x6x32    | 6x6x48      |
| Convolution       	| kernel: 3x3; stride:1x1; padding: same 	    | 6x6x48    | 6x6x64      |
| Convolution       	| kernel: 3x3; stride:1x1; padding: same 	    | 6x6x64    | 6x6x96      |
| Max pooling	      	| kernel: 2x2; stride:2x2;  	                | 6x6x96    | 3x3x96      |
| Flatten				| Input 3x3x96 -> Output 864					| 3x3x96    | 864         |
| Fully connected		| connect every neurel with next layer 		    | 864       | 400         |
| Fully connected		| connect every neurel with next layer	        | 400       | 160         |
| Fully connected		| output 43 probabilities for each lablel  		| 160       | 43          |

 
Apart from this, I have used following methods to make the model work better:

- Learning rate decay  
In training deep networks, when the learning rate is large, the system contains too much kinetic energy and the parameter vector bounces around chaotically, ubable to settle down into deeper; when the learning rate is small, you will be wasting computation bouncing around chaotically with little improvement for a long time. If the learning rate can decay from large to small while training, the network will move fast at the begining and improve little by little in the end. There are three commonly used types of method: step dacay, exponential decay and 1/t decay, more information can be found [here](http://cs231n.github.io/neural-networks-3/#anneal) and [here](https://zhuanlan.zhihu.com/p/32923584). Cause I use tensorflow to realize AlexNet and exponential dacay are used for learning decay, so I choose it as my method, its usage can be find [here](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay) is used to decay learning rate. Maybe it is not a good method, since there is tow more hyper parameters (decay_step and decay_rate) to tune. 
- Adam optimization  
[Adam](https://arxiv.org/abs/1412.6980) is a popular optimization recently proposed by Diederik P. Kingma and Jimmy Ba, like previous proposed Adagrad and RMSprop, it is a kind of adaptive learning rate method. With Adam, we don't have to use learning rate decay and tune three parameters for perfect learning rate. It is fabilous, so I will use it in most of times. After adapting Adam, the accuracy for training set, validation set and testing set are 99.9%, 96.9% and 94.2% respectively. The model is a little overfitting to training set, so some regularization methods are used to reduce it.
- L2 regulization  
L2 regulization is used to reduce overfitting by adding regulization loss to loss function, it is based on the assume that the bigger regulization loss is the more complex the model is. It is well known that complex model is more easily overfit to training set, thus, through reducing regulization loss to make the model simpler.
The regulization loss is the sum of L2 norm of weights for each layer multiple regulization parameter `lambda` in most cases, `lambda` is a small positive number that controls the regulization degree. Tensorflow documetn for how to use l2 regulization can be find [here](https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss).  

Training 
---
I have turned the following three hyperparameters to train my model.
* LEARNING_RATE = 5e-4
* EPOCHS = 30
* BATCH_SIZE = 128
* keep_prop = 0.5
* LAMBDA = 1e-5


The results are:  
| Dataset         		|   Accuracy	|
|:---------------------:|:-------------:|
| training set       	| 100.0%  	    |
| validation set	    | 96.0% 	    |
| testing set       	| 94.6% 	    |

Testing on new images
---

Here are ten German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]  

The label for each sign is: [0, 1, 3, 5, 8, 23, 26, 30 ,32, 42]

As we can see, that the 7th figure is overexposed, it can hardly be recognized by eyes. Therefore, the 7th figure may be difficult for the model to classify.   
The 5th and 10 th figure isn't exposed enough or was taken at night, they are not light enough, so they might also be mistaken.   
As for the other figures, I think they are clear enough for the model recognized.  


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

References
---
[The German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)  
[Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition](https://www.sciencedirect.com/science/article/pii/S0893608012000457?via%3Dihub)  
[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://219.216.82.193/cache/13/03/yann.lecun.com/a46bf8e4b17c2a9e46a2a899a68a0a0d/sermanet-ijcnn-11.pdf)  
[The German Traffic Sign Recognition Benchmark: A multi-class classification competition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6033395)  
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
