# **Traffic Sign Recognition** 

## README

---
This is my implementation of a traffic sign recognition program, a project for Udacity's Self-Driving Cars Nanodegree. Here below I detail my coverage of project requirements.

Files in the project:

 - `additional/` contains additional pictures of traffic signs, not part of datasets
 - `datasets/` directory to hold a [dataset of German traffic signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
 - `.idea/` PyCharm project
 - `README.md` this file
 - `README_pics/` contains images for this document
 - `signnames.csv` list of class numbers as used in datasets, each class number with the corresponding traffic sign name
 - `Traffic_Sign_Classifier.html` Dump from a completed computation of Traffic_Sign_Classifier.ipynb
 - `Traffic_Sign_Classifier.ipynb` Jupyter Notebook with the classifier program

[//]: # (Image References)

[additional1]: ./README_pics/1.jpg "Traffic Sign 1"
[additional2]: ./README_pics/2.jpg "Traffic Sign 2"
[additional3]: ./README_pics/3.jpg "Traffic Sign 3"
[additional4]: ./README_pics/4.jpg "Traffic Sign 4"
[additional5]: ./README_pics/5.jpg "Traffic Sign 5"
[chart]: ./README_pics/epochs_chart.jpg "Training Chart"
[histogram]: ./README_pics/classes_histogram.jpg "Classes Histogram"
[mis_histogram]: ./README_pics/mis_histogram.jpg "Misclassified Classes Histogram"
[activation_pre]: ./README_pics/activation_pre.jpg "Activation Map Before Training"
[activation_post]: ./README_pics/activation_post.jpg "Activation Map After Training"
[new_histogram]: ./README_pics/new_histogram.jpg "New Images Probability Distribution"

---
### Dependencies
Download the [zip file](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) with a dataset of German traffic signs, unzip it and place the `.p` files in directory `./datasets`

The following Python packages need to be installed:
 - cv2
 - matplotlib
 - numpy
 - prettytable
 - sklearn
 - tensorflow

Usage of a GPU supported by Tensorflow is advisable. Computation takes a couple of minutes on an NVIDIA GTX 970.

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

After loading the dataset, the program provides a summary of the traffic signs dataset, see Sect. "Step 1: Dataset Summary & Exploration" in the Jupyter Notebook.   

* The size of training set is 34799 samples.
* The size of validation set is 4410 samples. 
* The size of test set is 12630 samples.
* The shape of a traffic sign image is (32, 32, 3), i.e. an image of 32x32 pixels with 3 channels (RGB).
* The number of unique classes in the data set is 43.

**Note: the dataset and the program number classes with integers from 0 to 42.**

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Next, the code in Sect. "Step 1: Dataset Summary & Exploration" picks nine images scattered across the training dataset and shows them along with the respective class. Images are choosen across the dataset, instead of taking the first nine, which would look very much alike. See Notebook cell [8].  

A histogram, reproduced below, shows the frequency of each class in each dataset. We can see that different classes are represented with very different frequency in any given dataset. However, given a class, its frequency seems quite consistent across the three datasets: training, validation and test.

![Histogram][histogram]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I have defined a number of functions useful to pre-process datasets, see cell [4] of the Notebook. Specifically, `expand_dynamic_range()`, `normalize_with_averages()`, `normalize()` and `pre_process_dataset()` allow to:
  - Convert every image to YUV color space, and then to grayscale, by dropping channels U and V and retaining channel Y
  - Expand the dynamic range of every image, i.e. determine the minimum and maximum pixel values of the image, and then scale the pixel values to range from -1 to 1.
  - Determine the average value of pixels in the training set, and then subtract it from pixel values of all datasets (i.e. all datasets are normalised using the average from the training set).

Pre-processing input images in this way allows to obtain a higher accuracy on the validation set.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Data are loaded already split in the three datasets from pickled files, right at the beginning of Sect. "Step 0: Load the Data" of the notebook.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Function `build_network()` builds the neural network layer by layer. It calls `get_conv_layer()` and `get_fully_connected_layer()` to get a convolutional and a fully connected layer respectively (not inclusive of any activation function or pooling). Implementation is in cell [5] of the Notebook.

Function `build_network()` also applies L2 regularization to all fully connected layers; it is possible to tune it, or drop it altogether, when defining the loss function to be optimised.

The network architecture is derived from LeNet, which proved successful in similar tasks of image recognition. However, I have increased the number of filters in each convolutional layer, and widened the fully connected layers, which increased the trained network accuracy on the validation set. Table below illustrates the sequence of layers, each with the shape of its input and output. 


| Layer         		|     Parameters	        	      	| Output   |  
|:----------------------|:--------------------------------------|:---------| 
| Input         		| 32x32x1 grayscale image		     	| 32x32x1  |
| 64xConvolution    	| 5x5 kernel, 1x1 stride, valid padding | 28x28x64 |
| Activation			| RELU                      	        | 28x28x64 |
| Max pooling	      	| 2x2 stride                        	| 14x14x64 |
| 64xConvolution     	| 5x5 kernel, 1x1 stride, valid padding | 10x10x64 |
| Activation			| RELU                           		| 10x10x64 |
| Max pooling	      	| 2x2 stride            	    		| 5x5x64   |
| Flatten       	    |                                       | 1600     |
| Dropout       		| 30% prob. to drop              	    | 1600     |
| Fully connected		| L2 regularization                 	| 384      |
| Dropout       		| 30% prob. to drop                 	| 384      |
| Activation			| RELU                      	    	| 384      |
| Fully connected		| L2 regularization                  	| 192      |
| Dropout       		| 30% prob. to drop             	    | 192      |
| Activation			| RELU                      	    	| 192      |
| Fully connected		| L2 regularization                   	| 43       |
 
Tensorflow allows naming nodes in the computation graph with strings. With this I can retrieve a reference to those nodes later in the program, with no need to keep it stored in Python variables. I have assigned names to the convolutional layers output, after activation but before max-pooling, in order to later display their activation maps.

I then define nodes of the computation graph necessary to calculate the loss, along with other performance metrics, and to classify new traffic sign images; see cell [11] in the Notebook. 

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Sect. "Model Architecture" in the Notebook, cell [11], sets essential parameters for the optimisation:
 - Number of epochs to run the training, 100
 - The batch size for training data, 256
 - Initial learning rate, 0.001
 - Rate of decay for the learning rate, 0.95
 - How often decay is applied to the learning rate, once every 3 epochs
 - Weight for the cost of L2 regularization, 0 (turns off L2 regularization).

Optimisation algorithm is Adam, as implemented by Tensorflow.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Sect. "Train, Validate and Test the Model" starts with two helper functions, `evaluate()` to classify a dataset in batches, and evaluate the accuracy on that dataset, and `output_activation_map()` to plot the activation map of a given layer and input image. Both have been adapted from code provided by Udacity.
 
 Cell [9] of the Notebook shows additional traffic sign images I downloaded from the web, and for one of them the program will show the activation map, before network training. I will use it to compare it with the activation map after network training.

Then it is time to perform the optimisation, whose loop is in cell [17] of the Notebook, and display reports of the optimisation process and result, cells [19] to [22].

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.978 
* test set accuracy of 0.971

After optimisation, the program charts accuracy on training and validation set, and the loss function, against epochs. The chart shows a sound evolution of metrics, with loss trending toward near zero, training accuracy converging to 1, and the validation accuracy closely oscillating around the final value. There is no evidence of over or under fitting.

![Chart][chart]

The program selects across the validation set 9 images that have been misclassified and displays them along with histograms of their respective softmax probabilities. The selected images are all either very dark or very bright, and in some case the traffic sign doesn't fill the frame (it is small).
 
![Histogram][mis_histogram]
 
 Further in the reporting, the program computes precision and recall for every class of traffic sign and presents them in a table. They are calculated with function `metrics.precision_recall_fscore_support()` form Scikit-Learn. This information could be used to try to further improve the classification accuracy, by augmenting the training dataset with artificially generated samples representative of classes with a low score.
 
 And finally, the model accuracy against the test set is printed. I completed tuning of the network and the learning process based on the validation test only, with this print disabled. This way I avoided to leek information from the test set into the training set. 
 
I started from a simplification of LeNet architecture, that I had used in a previous project, to classify of hand-written digits. Initially I used only 6 filters in the first convolutional layer, and 16 in the second. I increased them both to 64, and also widened the fully connected layers. As a result, training time increased, but so did accuracy on the evaluation dataset.

I have then introduced conversion to gray-scale, converting the input images to YUV colour space and keeping the Y channel alone, and introduced drop-out after each fully connected layer (except the classification layer). Both changes further improved accuracy on the validation set. After trying different values for the drop rate, I settled on a 30%, the same for every drop-out layer. 

Experimentation with L2 normalisation didn't lead to further improvements, therefore I left its support in code, but turned it off. I believe drop-out did a job good enough in preventing overfitting, and L2 normalisation didn't help further.

Weights in bias vectors are initialised with a small positive value, as it is common practice, while weights of filters and fully connected layers use a variation of Xavier initialisation (He et al. 2015). See http://cs231n.github.io/neural-networks-2/#init for a detailed description. The intention is to have uniform variance at the output of neurons right after initialisation, which empirically showed to improve convergence rate. I cannot claim I observed any noticeable improvement, I presume the architecture I adopted is not deep enough for that initialisation technique to make a difference, compared to plain random Gaussian initialisation.   

Charting loss and accuracy per epoch proved invaluable not only to check for over/under fitting, but also to tune the learning rate. With a constant learning rate that was too small, validation accuracy would converge too slowly, and still slowly improve after hundreds of epochs. With a learning rate too high, accuracy (and loss) would not converge at all, but keep oscillating erratically. I then switched to an exponentially decaying learning rate, and tuned its parameters with the aid of the chart.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

These are five German traffic signs that I found on the web; I cropped the original pictures to meet the expected input size:

![Image 1][additional1]
![Image 2][additional2]
![Image 3][additional3]
![Image 4][additional4]
![Image 5][additional5]

The first image should be the easiest to classify, as it is well exposed, the sign centred and filling the frame, well distinct from the background. The second image is dark and foggy. The third and the forth have clutter in the background. The fifth is out of focus, and the sign is blurred.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Cell [24] of the Notebook classifies the new images. The trained network correctly classified all of them, an accuracy of 100%, which is consistent with the network accuracy against the test set of 97.1%.

The (all correctly) classified traffic signs were:
 - Yield
 - General caution
 - Road work
 - Roundabout mandatory
 - Speed limit (120km/h)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Cells [26] of the Notebook prints, for every new image, the 5 predictions with highest softmax probability, along with their probabilities. The information is reported here below.

Correct| Pred1 |  Prob1   | Pred2 |  Prob2   | Pred3 |  Prob3   | Pred4 |  Prob4   | Pred5 |  Prob5   |
|:-------|:-------|:---------|:-------|:---------|:-------|:---------|:-------|:---------|:-------|:---------|
|   13  |    13  |1.00e+00 |   3   | 2.25e-35 |   0   | 0.00e+00 |   1   | 0.00e+00 |   2   | 0.00e+00 |
|   18  |   18  | 1.00e+00 |   26  | 7.03e-11 |   27  | 9.13e-15 |   37  | 6.00e-17 |   24  | 8.79e-18 |
|   25  |   25  | 1.00e+00 |   20  | 4.32e-14 |   38  | 4.25e-14 |   23  | 5.07e-16 |   11  | 2.35e-16 |
|   40  |   40  | 1.00e+00 |   12  | 5.04e-07 |   11  | 2.80e-12 |   42  | 1.22e-12 |   16  | 8.28e-13 |
|   8   |   8   | 1.00e+00 |   7   | 4.64e-12 |   5   | 2.04e-14 |   0   | 1.11e-14 |   4   | 4.86e-15 |
|8 |   8    | 1.00e+00 |   7    | 6.42e-16 |   4    | 2.31e-23 |   2    | 5.03e-24 |   40   | 1.28e-24 |

Cell [27] of the Notebook draws the histogram of probability distribution for each new images, also reproduced here.

![Histogram2][new_histogram]

 It is interesting to see that the network reports to be "certain" of the additional images classification: softmax probability of every predicted class is virtually 1. This in spite of having chosen 4 of them thinking they would be challenging to classify correctly.

Let's now look at the activation map after the RELUs of the first convolutional layer, when a certain image is fed to the network, before training; that is, with all weights initialized randomly. I output it in cell [16] of the Notebook.

This is the input image of choice:

![Image 4][additional4]

And this is the activation map, before network training:

![Activation_pre][activation_pre]

Later, in cell [28] of the Notebook, I plotted the same activation map again, for the same input image, but this time after training of the network.

![Activation_post][activation_post]

Features of the input image are recognisable in both activation maps, before and after training. This obviously doesn't prove per se that the network has learned to detect those features, leave alone before the training process! However, we can notice that the activation map taken before training often shows activation in correspondence to the sky, while the activation map after training doesn't. That makes sense, as the sky is not relevant to classify the traffic sign, and it looks like the network has "learned" to ignore it.
