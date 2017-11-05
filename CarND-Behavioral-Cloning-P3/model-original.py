
######################
# Let's begin!
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import time
import shutil
import os
import sys
import random
import cv2
import math
import json

import keras
from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from IPython.display import display # Allows the use of display() for DataFrames

# Visualizations will be shown in the notebook.
#%matplotlib inline

#################################
# Loading Dataset

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
data = pd.read_csv('./data/driving_log.csv', names=columns)

print("Dataset Columns:", columns, "\n")
print("Shape of the dataset:", data.shape, "\n")
print(data.describe(), "\n")

print("Data loaded...")


#######################
# Exploring Dataset

binwidth = 0.025

# histogram before image augmentation
plt.hist(data.steering_angle,bins=np.arange(min(data.steering_angle), max(data.steering_angle) + binwidth, binwidth))
plt.title('Number of images per steering angle')
plt.xlabel('Steering Angle')
plt.ylabel('# Frames')
plt.show()


##############################
# Data Partitioning

# Get randomized datasets for training and validation

# shuffle data
data = data.reindex(np.random.permutation(data.index))

num_train = int((len(data) / 10.) * 9.)

X_train = data.iloc[:num_train]
X_validation = data.iloc[num_train:]

print("X_train has {} elements.".format(len(X_train)))
print("X_valid has {} elements.".format(len(X_validation)))


##########################################
# Configurable Variables

# image augmentation variables
CAMERA_OFFSET = 0.25
CHANNEL_SHIFT_RANGE = 0.2
WIDTH_SHIFT_RANGE = 100
HEIGHT_SHIFT_RANGE = 40

# processed image variables
PROCESSED_IMG_COLS = 320
PROCESSED_IMG_ROWS = 160
PROCESSED_IMG_CHANNELS = 3

# model training variables
NB_EPOCH = 8
BATCH_SIZE = 128 #256


###################################
# Image Augmentation Functions

# flip images horizontally
def horizontal_flip(img, steering_angle):
    flipped_image = cv2.flip(img, 1)
    steering_angle = -1 * steering_angle
    return flipped_image, steering_angle

# shift range for each channels
def channel_shift(img, channel_shift_range=CHANNEL_SHIFT_RANGE):
    img_channel_index = 2 # tf indexing
    channel_shifted_image = random_channel_shift(img, channel_shift_range, img_channel_index)
    return channel_shifted_image

# shift height/width of the image by a small fraction
def height_width_shift(img, steering_angle):
    rows, cols, channels = img.shape
    
    # Translation
    tx = WIDTH_SHIFT_RANGE * np.random.uniform() - WIDTH_SHIFT_RANGE / 2
    ty = HEIGHT_SHIFT_RANGE * np.random.uniform() - HEIGHT_SHIFT_RANGE / 2
    steering_angle = steering_angle + tx / WIDTH_SHIFT_RANGE * 2 * .2
    
    transform_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])
    
    translated_image = cv2.warpAffine(img, transform_matrix, (cols, rows))
    return translated_image, steering_angle


def brightness_shift(img, bright_value=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if bright_value:
        img[:,:,2] += bright_value
    else:
        random_bright = .25 + np.random.uniform()
        img[:,:,2] = img[:,:,2] * random_bright
    
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

# crop the top 1/5 of the image to remove the horizon and the bottom 25 pixels to remove the car’s hood
def crop_resize_image(img):
    shape = img.shape
    img = img[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    img = cv2.resize(img, (PROCESSED_IMG_COLS, PROCESSED_IMG_ROWS), interpolation=cv2.INTER_AREA)    
    return img

def apply_random_transformation(img, steering_angle):
    
    transformed_image, steering_angle = height_width_shift(img, steering_angle)
    transformed_image = brightness_shift(transformed_image)
    # transformed_image = channel_shift(transformed_image) # increasing train time. not much benefit. commented
    
    if np.random.random() < 0.5:
        transformed_image, steering_angle = horizontal_flip(transformed_image, steering_angle)
            
    transformed_image = crop_resize_image(transformed_image)
    
    return transformed_image, steering_angle


def read_image(fn):
    img = load_img(fn)
    img = img_to_array(img) 
    return img

test_fn = "data/IMG/center_2017_02_12_19_53_07_198.jpg"
steering_angle = 0.0617599

test_image = read_image(test_fn)

plt.subplots(figsize=(5, 18))

# original image
plt.subplot(611)
plt.xlabel("Original Test Image, Steering angle: " + str(steering_angle))
plt.imshow(array_to_img(test_image))

# horizontal flip augmentation
flipped_image, new_steering_angle = horizontal_flip(test_image, steering_angle)
plt.subplot(612)
plt.xlabel("Horizontally Flipped, New steering angle: " + str(new_steering_angle))
plt.imshow(array_to_img(flipped_image))

# channel shift augmentation
channel_shifted_image = channel_shift(test_image, 255)
plt.subplot(613)
plt.xlabel("Random Channel Shifted, Steering angle: " + str(steering_angle))
plt.imshow(array_to_img(channel_shifted_image))

# width shift augmentation
width_shifted_image, new_steering_angle = height_width_shift(test_image, steering_angle)
new_steering_angle = "{:.7f}".format(new_steering_angle)
plt.subplot(614)
plt.xlabel("Random HT and WD Shifted, New steering angle: " + str(new_steering_angle))
plt.imshow(array_to_img(width_shifted_image))

# brightened image
brightened_image = brightness_shift(test_image, 255)
plt.subplot(615)
plt.xlabel("Brightened, Steering angle: " + str(steering_angle))
plt.imshow(array_to_img(brightened_image))

# crop augmentation
cropped_image = crop_resize_image(test_image)
plt.subplot(616)
plt.xlabel("Cropped and Resized, Steering angle: " + str(steering_angle))
_ = plt.imshow(array_to_img(cropped_image))


######################################
# Keras generator for subsampling

def load_and_augment_image(line_data):
    i = np.random.randint(3)
    
    if (i == 0):
        path_file = line_data['left'][0].strip()
        shift_angle = CAMERA_OFFSET
    elif (i == 1):
        path_file = line_data['center'][0].strip()
        shift_angle = 0.
    elif (i == 2):
        path_file = line_data['right'][0].strip()
        shift_angle = -CAMERA_OFFSET
         
    steering_angle = line_data['steering_angle'][0] + shift_angle
    
    img = cv2.imread(path_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, steering_angle = apply_random_transformation(img, steering_angle)
            
    return img, steering_angle

# generators in multi-threaded applications is not thread-safe. Hence below:
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self
    
    def __next__(self):
        with self.lock:
            return self.it.__next__()
        
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

generated_steering_angles = []
threshold = 1

@threadsafe_generator
def generate_batch_data(_data, batch_size = 32):
    
    batch_images = np.zeros((batch_size, PROCESSED_IMG_ROWS, PROCESSED_IMG_COLS, PROCESSED_IMG_CHANNELS))
    batch_steering = np.zeros(batch_size)
    
    while 1:
        for batch_index in range(batch_size):
            row_index = np.random.randint(len(_data))
            line_data = _data.iloc[[row_index]].reset_index()
            
            # idea borrowed from Vivek Yadav: Sample images such that images with lower angles 
            # have lower probability of getting represented in the dataset. This alleviates 
            # any problems we may ecounter due to model having a bias towards driving straight.
            
            keep = 0
            while keep == 0:
                try:
                    x, y = load_and_augment_image(line_data)
                except:
                    continue    
                
                if abs(y) < .1:
                    val = np.random.uniform()
                    if val > threshold:
                        keep = 1
                else:
                    keep = 1
            
            batch_images[batch_index] = x
            batch_steering[batch_index] = y
            generated_steering_angles.append(y)
        yield batch_images, batch_steering


####################
# Examples of generated images

iterator = generate_batch_data(X_train, batch_size=10)
sample_images, sample_steerings = iterator.__next__()

plt.subplots(figsize=(20, 5))
for i, img in enumerate(sample_images):
    plt.subplot(2, 5, i+1)
    plt.axis('off')
    plt.title("Steering: {:.4f}".format(sample_steerings[i]))
    plt.imshow(img)
plt.show()


####################
# Model Architecture and training

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(PROCESSED_IMG_ROWS, PROCESSED_IMG_COLS, PROCESSED_IMG_CHANNELS)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu', name='Conv1'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv2'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv3'))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512, activation='elu', name='FC1'))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1, name='output'))
model.summary()

# compile
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mse', metrics=[])


######################################
# Model Visualization

from keras.utils.visualize_util import plot

plot(model, to_file='model.png', show_shapes=True)

img = read_image('model.png')

# original image
plt.subplots(figsize=(5,10))
plt.subplot(111)
plt.axis('off')
plt.imshow(array_to_img(img))


############################################
# Model Fitting

class LifecycleCallback(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global threshold
        threshold = 1 / (epoch + 1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        print('BEGIN TRAINING')
        self.losses = []

    def on_train_end(self, logs={}):
        print('END TRAINING')
        
# Calculate the correct number of samples per epoch based on batch size
def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil(num_batches)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


########################################
# Let the training begin!

lifecycle_callback = LifecycleCallback()       

train_generator = generate_batch_data(X_train, BATCH_SIZE)
validation_generator = generate_batch_data(X_validation, BATCH_SIZE)

samples_per_epoch = calc_samples_per_epoch((len(X_train)*3), BATCH_SIZE) 
nb_val_samples = calc_samples_per_epoch((len(X_validation)*3), BATCH_SIZE) 

history = model.fit_generator(train_generator, 
                              validation_data = validation_generator,
                              samples_per_epoch = samples_per_epoch, 
                              nb_val_samples = nb_val_samples,
                              nb_epoch = NB_EPOCH, verbose=1, nb_worker=1, pickle_safe=True, 
                              callbacks=[lifecycle_callback])


#######################################################
#Save Model


model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save("./model.h5")
model.save_weights("./model_weights.h5")
print("Saved model to disk")


###########################################
# Analysis

plt.hist(generated_steering_angles, bins=np.arange(min(generated_steering_angles), max(generated_steering_angles) + binwidth, binwidth))
plt.title('Number of augmented images per steering angle')
plt.xlabel('Steering Angle')
plt.ylabel('# Augmented Images')
plt.show()


#############################################
# Plots

# list all data in history
print(history.history.keys())

# summarize history for epoch loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.show()

# summarize history for batch loss
batch_history = lifecycle_callback.losses
plt.plot(batch_history)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('batches')
plt.show()


#######################################
# Some layer visualizations

# Layer visualizations

test_fn = "data/IMG/center_2017_02_12_19_53_07_198.jpg"

def visualize_model_layer_output(layer_name):
    model2 = Model(input=model.input, output=model.get_layer(layer_name).output)

    img = load_img(test_fn)
    img = crop_resize_image(img_to_array(img))
    img = np.expand_dims(img, axis=0)

    conv_features = model2.predict(img)
    print("conv features shape: ", conv_features.shape)
    
    # plot features
    plt.subplots(figsize=(5, 5))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.imshow(conv_features[0,:,:,i], cmap='gray')
    plt.show()

visualize_model_layer_output('Conv1')

visualize_model_layer_output('Conv2')


##################################
# The Final Test

from IPython.display import YouTubeVideo
YouTubeVideo('mYejcv8uDkw')

YouTubeVideo('TlTQVpRr6N8')




