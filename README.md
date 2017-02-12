#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**



[//]: # (Image References)

[image1]: . ModelDiagram.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed
nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2

pool_size = (2, 2)

	# convolution kernel size
kernel_size = (3, 3)

	# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 127.5 - 1.,input_shape=input_shape,output_shape=input_shape))

model.add(Convolution2D(16, 3, 3,border_mode='valid',input_shape=input_shape))
	# Applying ReLU
model.add(Activation('relu'))
	# The second conv layer will convert 16 channels into 8 channels
model.add(Convolution2D(8, 3, 3))
	# Applying ReLU
model.add(Activation('relu'))
	# The second conv layer will convert 8 channels into 4 channels
model.add(Convolution2D(4, 3,3))
	# Applying ReLU
model.add(Activation('relu'))
	# The second conv layer will convert 4 channels into 2 channels
model.add(Convolution2D(2, 3,3))
	# Applying ReLU
model.add(Activation('relu'))
	# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=pool_size))
	# Apply dropout of 25%
model.add(Dropout(0.25))

	# Flatten the matrix. The input has size of 360
model.add(Flatten())
	# Input 360 Output 16
model.add(Dense(16))
	# Applying ReLU
model.add(Activation('relu'))
	# Input 16 Output 16
model.add(Dense(16))
	# Applying ReLU
model.add(Activation('relu'))
	# Input 16 Output 16
model.add(Dense(16))
	# Applying ReLU
model.add(Activation('relu'))
	# Apply dropout of 50%
model.add(Dropout(0.5))
	# Input 16 Output 1
model.add(Dense(1))

My model consists of a convolution neural network and give below code will explain the use of filter and depth of each layer 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually and learning rate is 0.00001 .

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
1. Center camera images and their angles
2. Left camera images and their angles added to a small constant(0.16)
3. Right camera images and their angles added to a small constant(0.16)
4. Inverted images of 1, 2, 3 with their angles multiplied by -1
5. Blurred images of 1, 2, 3 with their angles unchanged
6. Noisy images of 1,2, 3 with their angles unchanged
7. Blurred versions of 4
8. Noisy versions of 4


For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was from the  [Nvidia Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) . But this approach was not working as out of box . 

My first step was to use a convolution neural network model similar to the as Nvidia paper . I thought this model might be appropriate because as it was alread tested 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
