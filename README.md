# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Center.jpg "Center"
[image2]: ./Left.jpg "Left"
[image3]: ./Right.jpg "Right"

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

I have employed the CommaAi architecture for this regression problem.The details of as follows
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
|Normalizing Layer |
|Cropping Layer (75,25) | 60x320x3 RGB image
| Convolution 8x8x16     	| 4x4 stride, same padding, outputs 15x80x16 	|
| ELU					|										|
| Convolution 5x5x32	    | 2x2 stride, same padding, outputs 8x40x32      									|
| ELU		|         									| 									|
| Convolution 5x5x64 | 2x2 strides same padding outputs 4x20x64
|Flatten|
|Dropout(.2)|
|ELU
|Dense(512) | Parameters 2621952
|Dropout(.5)
|ELU
|Dense(1)| Parameters 513

    Total params: 2,689,649
    Trainable params: 2,689,649
    Non-trainable params: 0

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
```python
model.compile(optimizer="adam", loss="mse",metrics = ['accuracy'])
```

#### 4. Appropriate training data

Although i tried collection data from the simulator,due to hardware constraints i was not able to record quality data to be fed for training and hence only used the data provided by the udacity.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I initially started out with the **NVIDIA** architecture.The architecture was a very deep architecture and the final result was having the simulator car driving around the track successfully but it was swirling a lot.I then started out with **COMMA.AI** architecture.The model was not very deep but was a very simple model.The car was swirling a lot less and the turns were smoother.This conditions should not be the case to chose an architecute but due to hardware constraints I was compairing the final output of the model based on this ie the simulation of the car in autonomous mode.

#### 2. Final Model Architecture

The final model architecture was **Comma.ai** architecture.
Here the code for the model.
```python
model = Sequential()
### To make data normalized ####
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(160,320,3)))
### To crop the data 75 pixel from above the 25 pixels from belows as these pixels contains useless information
    model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer="adam", loss="mse",metrics = ['accuracy'])
```
### visualization of the architecture ###

```python
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_8 (Lambda)                (None, 160, 320, 3)   0           lambda_input_8[0][0]             
____________________________________________________________________________________________________
cropping2d_8 (Cropping2D)        (None, 60, 320, 3)    0           lambda_8[0][0]                   
____________________________________________________________________________________________________
convolution2d_22 (Convolution2D) (None, 15, 80, 16)    3088        cropping2d_8[0][0]               
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 15, 80, 16)    0           convolution2d_22[0][0]           
____________________________________________________________________________________________________
convolution2d_23 (Convolution2D) (None, 8, 40, 32)     12832       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 8, 40, 32)     0           convolution2d_23[0][0]           
____________________________________________________________________________________________________
convolution2d_24 (Convolution2D) (None, 4, 20, 64)     51264       elu_2[0][0]                      
____________________________________________________________________________________________________
flatten_8 (Flatten)              (None, 5120)          0           convolution2d_24[0][0]           
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 5120)          0           flatten_8[0][0]                  
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 5120)          0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_29 (Dense)                 (None, 512)           2621952     elu_3[0][0]                      
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           dense_29[0][0]                   
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 512)           0           dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_30 (Dense)                 (None, 1)             513         elu_4[0][0]                      
```

### Model Performace every epochs ###
```python
Epoch 1/10
6428/6428 [==============================] - 15s - loss: 0.1021 - acc: 0.1739 - val_loss: 0.0414 - val_acc: 0.1704
Epoch 2/10
6428/6428 [==============================] - 14s - loss: 0.0431 - acc: 0.1801 - val_loss: 0.0360 - val_acc: 0.1741
Epoch 3/10
6428/6428 [==============================] - 14s - loss: 0.0368 - acc: 0.1773 - val_loss: 0.0306 - val_acc: 0.1897
Epoch 4/10
6428/6428 [==============================] - 14s - loss: 0.0327 - acc: 0.1749 - val_loss: 0.0309 - val_acc: 0.1692
Epoch 5/10
6428/6428 [==============================] - 14s - loss: 0.0310 - acc: 0.1845 - val_loss: 0.0285 - val_acc: 0.1748
Epoch 6/10
6428/6428 [==============================] - 14s - loss: 0.0301 - acc: 0.1767 - val_loss: 0.0290 - val_acc: 0.1679
Epoch 7/10
6428/6428 [==============================] - 14s - loss: 0.0290 - acc: 0.1814 - val_loss: 0.0284 - val_acc: 0.1947
Epoch 8/10
6428/6428 [==============================] - 14s - loss: 0.0278 - acc: 0.1730 - val_loss: 0.0253 - val_acc: 0.1772
Epoch 9/10
6428/6428 [==============================] - 14s - loss: 0.0274 - acc: 0.1722 - val_loss: 0.0272 - val_acc: 0.1810
Epoch 10/10
6428/6428 [==============================] - 14s - loss: 0.0268 - acc: 0.1711 - val_loss: 0.0255 - val_acc: 0.1835
```
#### 3. Creation of the Training Set & Training Process

I used the Udacity training data.The data was divided into following number of examples 
Training **6428**
Validation **1608**
Using the following code 
```python
from sklearn.model_selection import train_test_split
# pf is the panda data frame after reading the csv file.
train_samples, validation_samples = train_test_split(pf, test_size=0.20)
```

Then using generator I extracted the data and fed into the generator.The generator was extracting randomly either the left or right or center camera data. 


It was also flipping the image horizontally to avoid over fitting of the left turn centered data.The lane of the first track is left centered and to avoid that we either have to collect data buy running the car around the lane opposite or flip the images of the current track horizontally.

I also added 0.25 or -0.25 correction on the steering angle if i used left or right image.This correction helped us gathering data for recovery case.

![image1] ![image2] ![image3]

*here the generator code*.

```python
def generator_Data(data,batch_size=64):
            from scipy.misc import imread, imsave
            num_samples = len(data)
            while 1: # Loop forever so the generator never terminates
                for offset in range(0, num_samples, batch_size):
                    images = []
                    angles = []
                    choices = ['left','center','right']
                    choice_dict = {'left':0.25,'center':0.0,'right':-0.25}
                    for i in range(offset,(offset+batch_size)):  
                            if( i < num_samples):
                                choice = random.choice(choices)
                                index = data.index[i]
                                path = data[choice][index]
                                image = imread(path)
                                measurement = data['steering'][index]+choice_dict[choice]
                                if random.random() > 0.5:
                                    image, measurement = flipped(image, measurement)
                                images.append(image)
                                angles.append(measurement)
                    yield np.array(images), np.array(angles)
```
Images that were being fed to the model were Normalized using the following piece of code. 
 ```python
 #here x is the image.
 model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(160,320,3)))
 ```

Image was cropped before being fed to the conv layer using the follwing peice of code.The model was cropped 75 pixel from above and 25 pixel from below.These pixels contains useless information like the hills and the hood of the car.
```python
model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3)))
```
### For Training Process ###
1. I used an Epoch of 10
2. Number of training samples the model was iterating each epoch was equal to the number of training data.
3. Number of validation samples the model was iterating each epoch was equal to the number of validation data.
4. Batch size i chose was 64. Higher batch size might make the program run out of memory. 