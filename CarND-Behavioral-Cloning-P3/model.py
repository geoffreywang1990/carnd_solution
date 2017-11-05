
###############################################################################
# Global configurations
###############################################################################
DATASET_PATH = "./data/driving_log.csv"
INPUT_IMAGE_ROWS = 60
INPUT_IMAGE_COLS = 120
INPUT_IMAGE_CHANNELS = 3
AUGMENTATION_FACTOR = 3 # how many times each image in the dataset will be augmented
AUGMENTATION_NUM_BINS = 100 # in how many hit bins will the angles be grouped for augmentation
AUGMENTATION_BIN_MAX_PERC = 7   #  the percentagehow of the total images one hit bin can have
BATCH_SIZE = 32
NUM_EPOCHS = 10


###############################################################################
# Loading and exploring the dataset
###############################################################################
import csv
import matplotlib.pyplot as plt

print("\nLoading the dataset from file ...")
def load_dataset(file_path):
    dataset = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                dataset.append({'center':line[0], 'left':line[1], 'right':line[2], 'steering':float(line[3]), 
                            'throttle':float(line[4]), 'brake':float(line[5]), 'speed':float(line[6])})
            except:
                continue # some images throw error during loading    
    return dataset

dataset = load_dataset(DATASET_PATH)
print("Loaded {} samples from file {}".format(len(dataset),DATASET_PATH))

print("\nExploring the dataset ...")
# It plots the histogram of an arrray of angles: [0.0,0.1, ..., -0.1]
def plot_steering_histogram(steerings, title, num_bins=100):
    plt.hist(steerings, num_bins)
    plt.title(title)
    plt.xlabel('Steering Angles')
    plt.ylabel('# Images')
    plt.show()

# It plots the histogram of an arrray of associative arrays of angles: [{'steering':0.1}, {'steering':0.2}, ..., {'steering':-0.1}]
def plot_dataset_histogram(dataset, title, num_bins=100):
    steerings = []
    for item in dataset:
        steerings.append( float(item['steering']) )
    plot_steering_histogram(steerings, title, num_bins)

# Plot the histogram of steering angles before the image augmentation
plot_dataset_histogram(dataset, 'Number of images per steering angle before image augmentation', num_bins=AUGMENTATION_NUM_BINS)
print("Exploring the dataset complete.")


###############################################################################
# Partitioning the dataset into training(80%), validation (19%) and testing(1%)
###############################################################################

from random import shuffle
from sklearn.model_selection import train_test_split

print("\nPartitioning the dataset ...")

# Images of sequences must be found in all 3 datasets for a better generalization
shuffle(dataset)

X_train, X_validation = train_test_split(dataset, test_size=0.2)
X_validation, X_test = train_test_split(X_validation, test_size=0.05)

print("X_train has {} elements.".format(len(X_train)))
print("X_validation has {} elements.".format(len(X_validation)))
print("X_test has {} elements.".format(len(X_test)))
print("Partitioning the dataset complete.")


###############################################################################
# Dataset augmentation based on OpenCV methods and Keras methods immplemented here:
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
###############################################################################

import cv2
import numpy as np
from keras.preprocessing.image import *

# Flip image horizontally, flipping the angle positive/negative
def horizontal_flip(image, steering_angle):
    flipped_image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return flipped_image, steering_angle

# Shift width/height of the image by a small fraction of the total value, introducing an small angle change
def height_width_shift(image, steering_angle, width_shift_range=50.0, height_shift_range=5.0):
    # translation
    tx = width_shift_range * np.random.uniform() - width_shift_range / 2
    ty = height_shift_range * np.random.uniform() - height_shift_range / 2
    
    # new steering angle
    steering_angle += tx / width_shift_range * 2 * 0.2 
    
    transform_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    rows, cols, channels = image.shape
    
    translated_image = cv2.warpAffine(image, transform_matrix, (cols, rows))
    return translated_image, steering_angle

# Increase the brightness by a certain value or randomly
def brightness_shift(image, bright_increase=None):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    if bright_increase:
        image_hsv[:,:,2] += bright_increase
    else:
        bright_increase = int(30 * np.random.uniform(-0.3,1))
        image_hsv[:,:,2] = image[:,:,2] + bright_increase
    
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image

# Shift range for each channels
def channel_shift(image, intensity=10, channel_axis=2):
    image = random_channel_shift(image, intensity, channel_axis)
    return image

# Rotate the image randomly up to a range_degrees
def rotation(image, range_degrees=5.0):
    #image = random_rotation(image, range_degrees)
    degrees = np.random.uniform(-range_degrees, range_degrees)
    rows,cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1.0)
    image = cv2.warpAffine(image, matrix, (cols,rows), borderMode=cv2.BORDER_REPLICATE)
    return image

# Zoom the image randomly up to zoom_range, where 1.0 means no zoom and 1.2 a 20% zoom
def zoom(image, zoom_range=(1.0,1.2)): 
    #image = random_zoom(image, zoom_range)
    # resize
    factor = np.random.uniform(zoom_range[0], zoom_range[1])
    height, width = image.shape[:2]
    new_height, new_width = int(height*factor), int(width*factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # crop margins to match the initial size
    start_row = int((new_height-height)/2)
    start_col = int((new_width-width)/2)
    image = image[start_row:start_row + height, start_col:start_col + width]
    
    return image

# Crop and resize the image
def crop_resize_image(image, cols=INPUT_IMAGE_COLS, rows=INPUT_IMAGE_ROWS, top_crop_perc=0.1, bottom_crop_perc=0.2):
    height, width = image.shape[:2]
    
    # crop top and bottom
    top_rows = int(height*top_crop_perc)
    bottom_rows = int(height*bottom_crop_perc)
    image = image[top_rows:height-bottom_rows, 0:width]
    
    # resize to the final sizes even the aspect ratio is destroyed
    image = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_LINEAR)    
    return image

# Apply a sequence of random tranformations for a bettwe generalization and to prevent overfitting
def random_transform(image, steering_angle):
    
    # all further transformations are done on the smaller image to reduce the processing time
    image = crop_resize_image(image)
    
    # every second image is flipped horizontally
    if np.random.random() < 0.5: 
        image, steering_angle = horizontal_flip(image, steering_angle)
    
    image, steering_angle = height_width_shift(image, steering_angle)
    image = zoom(image)
    image = rotation(image)
    image = brightness_shift(image)   
    image = channel_shift(image)
    
    return img_to_array(image), steering_angle

# It loads an image from the disk into an array
def read_image(image_filename):
    image = load_img(image_filename)
    image = img_to_array(image) 
    return image


###############################################################################
# Testing each augmentation method independently on a random picture
###############################################################################

print("Testing each augmentation method independently ...")

test_image_filename = X_train[0]['center']
steering_angle = X_train[0]['steering']

test_image = read_image(test_image_filename)

plt.subplots(figsize=(20, 10))

# Initial image
plt.subplot(421)
plt.title("Initial test image, steering angle: {}".format(str(steering_angle)))
plt.axis('off')
plt.imshow(array_to_img(test_image))

# Horizontal flip
flipped_image, new_steering_angle = horizontal_flip(test_image, steering_angle)
plt.subplot(422)
plt.title("Horizontally flipped, new steering angle: {}".format(str(new_steering_angle)))
plt.axis('off')
plt.imshow(array_to_img(flipped_image))

# Channel shift
channel_shifted_image = channel_shift(test_image, 50)
plt.subplot(423)
plt.title("Channel shifted, steering angle: {}".format(str(steering_angle)))
plt.axis('off')
plt.imshow(array_to_img(channel_shifted_image))

# Width shift
width_shifted_image, new_steering_angle = height_width_shift(test_image, steering_angle)
new_steering_angle = "{:.8f}".format(new_steering_angle)
plt.subplot(424)
plt.title("Width and height shifted, new steering angle: {}".format(str(new_steering_angle)))
plt.axis('off')
plt.imshow(array_to_img(width_shifted_image))

# Brighteness shift
brightened_image = brightness_shift(test_image, 50)
plt.subplot(425)
plt.title("Brighteness shift, steering angle: {}".format(str(steering_angle)))
plt.axis('off')
plt.imshow(array_to_img(brightened_image))

# Zoom
zoomed_image = zoom(test_image)
plt.subplot(426)
plt.title("Zoomed in, steering angle: {}".format(str(steering_angle)))
plt.axis('off')
plt.imshow(array_to_img(zoomed_image))

# Rotation
rotated_image = rotation(test_image)
plt.subplot(427)
plt.title("Rotated, steering angle: {}".format(str(steering_angle)))
plt.axis('off')
plt.imshow(array_to_img(rotated_image))

# Crop and resize
cropped_image = crop_resize_image(test_image)
plt.subplot(428)
plt.title("Cropped and resized, steering angle: {}".format(str(steering_angle)))
plt.axis('off')
plt.imshow(array_to_img(cropped_image))

plt.show()
print("Testing each augmentation method independently complete.")


###############################################################################
# Keras generator for data augmentation
###############################################################################

# It loads one of the 3 images (center, left or right) and applies augumentation on it
# This method is called by the Keras generator
def load_and_augment_image(image_data, side_camera_offset=0.2):
    
    # select a value between 0 and 2 to swith between center, left and right image
    index = np.random.randint(3)
    
    if (index==0):
        image_file = image_data['left'].strip()
        angle_offset = side_camera_offset
    elif (index==1):
        image_file = image_data['center'].strip()
        angle_offset = 0.
    elif (index==2):
        image_file = image_data['right'].strip()
        angle_offset = - side_camera_offset
         
    steering_angle = image_data['steering'] + angle_offset
    
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # apply a misture of several augumentation methods
    image, steering_angle = random_transform(image, steering_angle)
            
    return image, steering_angle

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

# Define some global variables used for augumenting the images
# A list of generated angles in all epochs alltogether for showing the displaying the final histogram
augmented_steering_angles = [] 
# The number of generated angels in the current epoch, reseted at the end of each epoch
epoch_steering_count = 0 
# A dictionary with the hits for each angle, reseted at the end of each epoch
# for the angles is multiplied by 100 and converted to integer, with the range aproximately (-150, +150)
# for the angle -0.142323 the index will be -14
bin_range = int(AUGMENTATION_NUM_BINS / 4 * 3) # for AUGMENTATION_NUM_BINS = 200 the range is (-150, 150)
epoch_bin_hits = {k:0 for k in range(-bin_range, bin_range)} 

@threadsafe_generator
def generate_batch_data(dataset, batch_size = 32):
    global augmented_steering_angles
    global epoch_steering_count
    global epoch_bin_hits
    batch_images = np.zeros((batch_size, INPUT_IMAGE_ROWS, INPUT_IMAGE_COLS, INPUT_IMAGE_CHANNELS))
    batch_steering_angles = np.zeros(batch_size)
    
    while 1:
        for batch_index in range(batch_size):
            
            # select a random image from the dataset
            image_index = np.random.randint(len(dataset))
            image_data = dataset[image_index]

            while 1:
                try:
                    image, steering_angle = load_and_augment_image(image_data)
                except:
                    continue # some images throw error during loading/augmentation
                
                # images with smaller angles have a lower probability of getting represented in the dataset,
                # and the model tends to have a bias towards driving straight
                # for equalizing the histogram a dictionary of hits for each bin is used: epoch_bin_hits

                # get current bin index, for AUGMENTATION_NUM_BINS = 200 the angle is multiplied by 100
                bin_idx = int (steering_angle * AUGMENTATION_NUM_BINS / 2)
                
                # don't allow one bin to have more than AUGMENTATION_BIN_MAX_PERC percent of the total augmented angles in the current epoch,
                # except for the case when not enough (less than 500) angles are already augmented
                if( epoch_bin_hits[bin_idx] < epoch_steering_count*AUGMENTATION_BIN_MAX_PERC/AUGMENTATION_NUM_BINS 
                    or epoch_steering_count<5*AUGMENTATION_NUM_BINS ):
  
                    batch_images[batch_index] = image
                    batch_steering_angles[batch_index] = steering_angle
                    augmented_steering_angles.append(steering_angle)
                    
                    epoch_bin_hits[bin_idx] = epoch_bin_hits[bin_idx] + 1
                    epoch_steering_count = epoch_steering_count + 1
                    break
            
        yield batch_images, batch_steering_angles


###############################################################################
# Examples of augmented images
###############################################################################

print("\nShowing examples of augmented images ...")

iterator = generate_batch_data(X_train, batch_size=10)
sample_images, sample_steerings = iterator.__next__()

plt.subplots(figsize=(20, 5))
for i, image in enumerate(sample_images):
    plt.subplot(2, 5, i+1)
    plt.title("Steering: {:.4f}".format(sample_steerings[i]))
    plt.axis('off')
    plt.imshow(array_to_img(image))
plt.show()
print("\nShowing examples of augmented images complete.")


###############################################################################
# Model architecture based on comma.ai
###############################################################################

from keras.models import Sequential, Model
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.advanced_activations import ELU
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam

print("\nBuilding and compiling the model ...")

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(INPUT_IMAGE_ROWS, INPUT_IMAGE_COLS, INPUT_IMAGE_CHANNELS)))
# Block - conv
model.add(Convolution2D(16, 8, 8, border_mode='same', subsample=[4,4], activation='elu', name='Conv1'))
# Block - conv
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv2'))
# Block - conv
model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv3'))
# Block - flatten
model.add(Flatten())
model.add(Dropout(0.2))
model.add(ELU())
# Block - fully connected
model.add(Dense(512, activation='elu', name='FC1'))
model.add(Dropout(0.5))
model.add(ELU())
# Block - output
model.add(Dense(1, name='output')) 
model.summary()

# compile
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mse', metrics=[])
print("\nBuilding and compiling the model complete.")


###############################################################################
# Model fitting
###############################################################################

import keras
from keras.callbacks import Callback
import math

print("\nTraining the model ...")

class LifecycleCallback(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global epoch_steering_count
        global epoch_bin_hits
        global bin_range
        epoch_steering_count = 0        
        epoch_bin_hits = {k:0 for k in range(-bin_range, bin_range)} 

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        print('Beginning training')
        self.losses = []

    def on_train_end(self, logs={}):
        print('Ending training')
        
# Compute the correct number of samples per epoch based on batch size
def compute_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil(num_batches)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch

lifecycle_callback = LifecycleCallback()       

train_generator = generate_batch_data(X_train, BATCH_SIZE)
validation_generator = generate_batch_data(X_validation, BATCH_SIZE)

samples_per_epoch = compute_samples_per_epoch((len(X_train)*AUGMENTATION_FACTOR), BATCH_SIZE) 
nb_val_samples = compute_samples_per_epoch((len(X_validation)*AUGMENTATION_FACTOR), BATCH_SIZE) 

history = model.fit_generator(train_generator, 
                              validation_data = validation_generator,
                              samples_per_epoch = samples_per_epoch, 
                              nb_val_samples = nb_val_samples,
                              nb_epoch = NUM_EPOCHS, verbose=1, 
                              callbacks=[lifecycle_callback])

print("\nTraining the model ended.")


###############################################################################
# Saving the model
###############################################################################
print("\nSaving the model ...")
model.save("./model.h5")
print("\nSaving the model complete.")


###############################################################################
# Showing the results
###############################################################################

print("\nShowing the results ...")

# The histogram after the augmentation
plot_steering_histogram(augmented_steering_angles, 'Number of images per steering angle after augmentation', num_bins=AUGMENTATION_NUM_BINS)

# The model loss
#print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# The history for batch loss
batch_history = lifecycle_callback.losses
plt.plot(batch_history)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('batches')
plt.show()

print("\nShowing the results complete.")


###############################################################################
# Visualization of convolutional layers
###############################################################################

#Visualization of the convolutional layers 
test_file = X_test[0]['center']

def visualize_model_layer_output(model, layer_name, image_file):
    test_model = Model(input=model.input, output=model.get_layer(layer_name).output)

    image = load_img(image_file)
    image = crop_resize_image(img_to_array(image))
    image = np.expand_dims(image, axis=0)

    conv_features = test_model.predict(image)
    print("Convolutional features shape: ", conv_features.shape)
    
    # plot features
    plt.subplots(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.imshow(conv_features[0,:,:,i], cmap='gray')
    plt.show()

visualize_model_layer_output(model, 'Conv1', test_file)
visualize_model_layer_output(model, 'Conv2', test_file)
visualize_model_layer_output(model, 'Conv3', test_file)
