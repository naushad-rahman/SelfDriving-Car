#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**



[//]: # (Image References)

[image1]: . /images/ModelDiagram.png "Model Visualization"
[image3]: ./examples/left (1).jpg.png "Recovery Image"
[image4]: ./examples/left (2).jpg.png "Recovery Image"
[image5]: ./examples/left (3).jpg.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./images/Flipped Image.png "Flipped Image"
[image8]: ./images/Flipped Image.png "Flipped Image"
[image9]: ./images/Flipped Image.png "Flipped Image"


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

Since , We donot use the complete image feature . I have trimmed the image upto 40% from top and 10% from botton . Which also reduce the image size.

We have added more data set to improve its robustness by adding the Noise , blurr images which helped me to take sharp turns . 

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

I have capture , one from center lane driving . Here is an example image of center lane driving
![center Lane ][center.jpg]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if the car goes off tracl . These images show what a recovery looks like starting from ... :

![ left][left (1).jpg]
![left][left (2).jpg]
![left][left (3).jpg]


After the collection process, I had 24,108 number of data points. I then preprocessed this data by 
1. Inverted images  
2. Blurred images 
3. Noisy images 
4. Blurred versions of 1
5. Noisy versions of 1
so each image has 5 copy with different feature sets . 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by I have only experiment with 1 ,5 ,10 and 15 since i find 15 doesnot give much significant improvment so i stick with 10  .I used an adam optimizer so that manually training the learning rate wasn't necessary.
