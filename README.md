# Behaviour cloning


## Approach for Designing Model Architecture

1. Initially I set up drive.py to output a constant steering angle just to wire things up.
2. After that, I create a very simple neural network with a single convolutional layer and a single fully connected layer to set up the pipeline for `drive.py` to make predictions using a trained model. With this simple model, I could not get very far on the track - it was hardly a few seconds before the car steered off the track.

3. I have experimented with a [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I need to preprocessing of the data to able to make it working

4. Since complete image details is not required , Our Main is to take feature from road . So to get the only road detail . I have trimmed the top 40 % and 10% from bottom . These processing alos reduce the image .


5. In the data processing and to generate much data i have inverted image around the vertical axis for all the images from the center, left and right cameras and multiplied the corresponding steering angle by -1. I have even added  significantly more data through several augmentation techniques like: adding noise to the image and blurring the image. 

## The Model

The model is very similar to the NVIDIA model with a few additional layers and dropout added to the fully connected layers for preventing overfitting:

![model](ModelDiagram.png)


## Model Training



- Normalization of data was done within the network using a Lambda function in Keras.

- Several techniques were used for preventing overfitting like dropout for fully connected layers .78


- Initially starting with a learning rate of 0.001, I was unable to keep the car on the track. Lowering the learning rate significantly helped me and I decided to go with a learning rate of 0.`00001`.

- The model was trained using an Adam optimizer with a learning rate of `0.00001`. The training was done for 10 epochs and on each iteration, a checkpoint in keras was executed to save the model with the lowest validation error.


## Data Set Generation

The data set was developed with all the following images

1. Center camera images and their angles
2. Left camera images and their angles added to a small constant(0.16)
3. Right camera images and their angles added to a small constant(0.16)
4. Inverted images of 1, 2, 3 with their angles multiplied by -1
5. Blurred images of 1, 2, 3 with their angles unchanged
6. Noisy images of 1,2, 3 with their angles unchanged
7. Blurred versions of 4
8. Noisy versions of 4

### Examples:

#### Center image

![center](images/center.png)

#### Pre-processed Center image

![pre processed center](images/pre-center.png)

#### Blurred Center image

![blurred center](images/center-blur.png)

#### Noisy Center image

![noisy center](images/center-noise.png)

#### Flipped Center image

![flipped center](images/center-flipped.png)

## Video Footage

Example can be found [here](https://www.youtube.com/watch?v=XuTiITj86H4)



