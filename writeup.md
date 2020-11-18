# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./images/distribution.png "Visualization"
[imageall]: ./images/allsigns.png "All signs"
[image2]: ./images/randomsigns.png "Random signs"
[image3a]: ./images/distort_orig.jpg "Original image"
[image3b]: ./images/distort_generated.jpg "Distorted image"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/danthe42/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?

* The size of the validation set is ?

* The size of test set is ?

* The shape of a traffic sign image is ?

* The number of unique classes/labels in the data set is ?

  Answers are in the second code cell of the jupyter notebook:

```
The size of training set is 34799
The size of the validation set is 4410
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43
```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

It is a bar chart showing how the images are distributed among the different labels in the data sets (in percentage of the given data set size). If can be seen that there is a big variance in the data samples: In the train set, the most images are with label class 2 (Speed limit (50hm/h)), about 5.75% of all images. At the other end of the spectrum are class 0 (Speed limit (20km/h)), class 19 (Dangerous curve to the left), or class 42 (End of no passing by vehicles over 3.5 metric tons) each with very small number of samples (about 0.5 - 0.6%). 

![alt text][image1]

I chose a random image from each class, just to visualize the traffic signs which are used to train the network: 

![alt text][imageall]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not convert the data to grayscale, as I was thinking that the sign's color can also be useful. (red/blue can be important) 

These are 5x5 images from the train set:

![alt text][image2]

I normalized the color pixel values in the images, so as to set the mean value to 0, and also put the values into the [-1,1] interval. That way the training of the network can be more efficient and/or quicker. ( normalizeimage() method in code )

After a few unsuccessful runs, I realized that the rare signs in the training set are often not recognized correctly. To fix this deficiency, I decided to generate additional data, and I expand the training set to include exactly the same amount of data for each class. ( extend_training_set() method in the code)

To generate a new image in a given class based on an other preexisting image in that same class, I had to distort/modify it randomly. I either rotated it by a small random angle, or added/decreased all pixel values with some random noise.    ( random_distort_image() method in the source code )

Here is an example of an original image and an augmented image generated from it:

![alt text][image3a]

![alt text][image3b]

 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5    | 1x1 stride, VALID padding, outputs 28x28x6 |
| TANH		| Hyperbolic tangent activation function |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 		|
| Convolution 5x5	  | 1x1 stride, VALID padding, outputs 10x10x16 |
| TANH	| Hyperbolic tangent activation function |
| Average pooling	| 2x2 stride,  outputs 5x5x16 |
| Flatten	| Flatten the previous layers to 400 units |
| Fully connected		| A fully connected layer with 400 inputs and 350 outputs |
| Sigmoid	| A sigmoid activation layer for the 350 outputs |
| Fully connected	| The second fully connected layer with 350 inputs and 200 outputs |
| Sigmoid	| A sigmoid activation layer for the 200 outputs |
| Fully connected	| A third fully connected layer with 200 inputs and 43 outputs |
| Softmax | The softmax activation to normalize the network's outputs |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 25 epochs, and used my expanded train data set split into 128 image long batches as steps in the training process. 

I set 0.001 as the initial learning rate, and later, after each 10th epoch, this value gets halved, because I wanted the learning curve to get smoother at later phases and use smaller values to modify the network parameters with.   

During training, after executing the network with an image, cross entropy is always calculated between the two inputs: the vector coming from the softmax output of the network, and the one hot encoded value of the correct label value. The average of these cross entropy values (which is the loss function) in each batches will be minimized by using tensorflow's built in AdamOptimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


