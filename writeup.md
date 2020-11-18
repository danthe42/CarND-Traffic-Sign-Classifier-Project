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
[image4]: ./images/customimages.png "Custom images"


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

```
Accuracy on the expanded train dataset = 100.00%
Accuracy on the validation dataset = 95.12%
Accuracy on the (never seen) test dataset = 93.65%
```

The starting point for the neural net architecture was the LeNet architecture, which is well known, used, and proven architecture for neural networks for image recognition tasks.  

Later on I adjusted a few things on this architecture: 

- I increased the size of the fully connected layers (for 43 labels, it looked appropriate), 
- Decided to change the activation functions in the convolution layer to tanh because I didn't like the original RELU, as it is unbounded.
- Changed the second pooling operation from max_pool to avg_pool, as it gave better results.    

As the model's result even on the never before seen test dataset is above 93.5%, and test results on new images downloaded from the net were correct when I examined them, in my opinion the network is working well. However, I'm sure it can be enhanced further with more work.    


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image4]

The left side of the first image is bent. The blue color on the second is much darker then the similar images in the train set (Based on a few images in the train dataset, I could not check all of them). The third and fourth are similar to each other and blurred. And the brightness is not consistent on the 5th image. 



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed Limit (30 km/h) | Speed Limit (30 km/h) |
| Keep right | Keep right |
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Pedestrians	| Pedestrians	|
| Stop	| Stop      				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is better than the accuracy on the test set, the reason of that can be that there are many dark/too bright/very blurry traffic signs in the data sets.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

The top 5 probabilities are logged out for each images:

```
Sign #1:
Probability: 99.9999% Prediction: 'Speed limit (30km/h)' 
Probability: 0.0001% Prediction: 'Speed limit (20km/h)' 
Probability: 0.0000% Prediction: 'Speed limit (50km/h)' 
Probability: 0.0000% Prediction: 'No entry' 
Probability: 0.0000% Prediction: 'Speed limit (70km/h)' 
Sign #2:
Probability: 100.0000% Prediction: 'Keep right' 
Probability: 0.0000% Prediction: 'Go straight or right' 
Probability: 0.0000% Prediction: 'Dangerous curve to the right' 
Probability: 0.0000% Prediction: 'Turn left ahead' 
Probability: 0.0000% Prediction: 'Road work' 
Sign #3:
Probability: 100.0000% Prediction: 'Right-of-way at the next intersection' 
Probability: 0.0000% Prediction: 'End of no passing by vehicles over 3.5 metric tons' 
Probability: 0.0000% Prediction: 'Beware of ice/snow' 
Probability: 0.0000% Prediction: 'Double curve' 
Probability: 0.0000% Prediction: 'Slippery road' 
Sign #4:
Probability: 99.6848% Prediction: 'Pedestrians' 
Probability: 0.3125% Prediction: 'Right-of-way at the next intersection' 
Probability: 0.0022% Prediction: 'General caution' 
Probability: 0.0002% Prediction: 'Double curve' 
Probability: 0.0001% Prediction: 'Speed limit (100km/h)' 
Sign #5:
Probability: 99.9999% Prediction: 'Stop' 
Probability: 0.0001% Prediction: 'Road work' 
Probability: 0.0000% Prediction: 'Yield' 
Probability: 0.0000% Prediction: 'No entry' 
Probability: 0.0000% Prediction: 'Bicycles crossing' 
```

For the 1st, 2nd, 3rd, and 5th images, the network gave the correct result with a very high probability. Actually, I found these numbers too high, so I first thought that it's a case of overfitting. But after trying to recognize a few additional images from the net, I had to accept that it's not the case.  

I've chosen the 4th image because it's similarity with the 3rd. Exactly this is what can be seen here. The second probability is 0.3% and the probable sign candidate is just the traffic sign on the 3rd image. The network successfully "noticed" that these two are similar. The 3rd probability is very low, 0.0022%, but definitely the red triangle with the exclamation mark is the third most similar sign among the 43 traffic signs.   


