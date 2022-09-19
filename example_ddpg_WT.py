from interface import DDPGInterface

from keras import layers
import keras

import numpy as np

import os

def build_task_predictor(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

img_shape = (96, 96, 1)

#num_train_samples = 100
#num_val_samples = 50
#num_holdout_samples = 50

import os
import csv
import random

#import torch
#import yaml

#from torch.utils.data import Dataset

data_dir = '/Users/monicalin/Desktop/Harvard/Capstone/data'

def generate_data_files(sample_percentage=1):

    image_reference_list = []

    subdirectories = list(os.walk(data_dir, topdown=False))[:-1]
    for subdir in subdirectories:
        image_location = subdir[0]
        images = subdir[2]
        species_rating = image_location.rsplit('/', 1)[-1].replace('_', ' ')
        score = int(species_rating.rsplit(' ', 1)[-1])
        species_class = species_rating.rsplit(' ', 1)[:-1][0]
        if len(species_class.rsplit(' ', 1)) > 1:
            species = species_class.rsplit(' ')[0]
            animal_class = ' '.join(species_class.rsplit(' ')[1:])
        else:
            animal_class = 'Unknown'
            species = species_class

        for image in images:
            image_reference = (image_location, species, animal_class, image, score)
            image_reference_list.append(image_reference)
    return image_reference_list

# shuffle then split
seed = 1234
image_reference_list = generate_data_files()
random.Random(seed).shuffle(image_reference_list)

# might have to scale image
training = image_reference_list[:int(len(image_reference_list) * 0.6 * sample_percentage)]  # sample_percentage is just so we dont have to process the entire data set (faster)
x_train = []
for image_reference in training():
    print("Processing file number:" + str(image_reference))
    file_path = os.path.join(image_reference.image_location, image_reference.image)
    im_features = cv2.imread(file_path)
    im_features = cv2.cvtColor(im_features, cv2.COLOR_BGR2GRAY)
    x_train.append(im_features)

sample_percentage=1
validation = image_reference_list[-int(len(image_reference_list) * 0.2 * sample_percentage):]
x_val=[]
for image_reference in validation():
    print("Processing file number:" + str(image_reference))
    file_path = os.path.join(image_reference.image_location, image_reference.image)
    im_features = cv2.imread(file_path)
    im_features = cv2.cvtColor(im_features, cv2.COLOR_BGR2GRAY)
    x_val.append(im_features)

testing = image_reference_list[-int(len(image_reference_list) * 0.2 * sample_percentage):]
x_holdout=[]
for image_reference in testing():
    print("Processing file number:" + str(image_reference))
    file_path = os.path.join(image_reference.image_location, image_reference.image)
    im_features = cv2.imread(file_path)
    im_features = cv2.cvtColor(im_features, cv2.COLOR_BGR2GRAY)
    x_holdout.append(im_features)

num_train_samples = len(training)
num_val_samples = len(validation)
num_holdout_samples = len(testing)


y_train = [image_reference.score for image_reference in training]
y_val = [image_reference.score for image_reference in validation]
y_holdout = [image_reference.score for image_reference in testing]


task_predictor = build_task_predictor(img_shape)
task_predictor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # speciffy the loss and metric used to train target net and controller respectively

interface = DDPGInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape)

interface.train(6)

save_dir = 'temp'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

controller_weights_save_path = r'temp/train_session_1_ddpg_controller'
task_predictor_save_path = r'temp/train_session_1_task_predictor'
interface.save(controller_weights_save_path=controller_weights_save_path,
               task_predictor_save_path=task_predictor_save_path)


interface = DDPGInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape, load_models=True, controller_weights_save_path=controller_weights_save_path, task_predictor_save_path=task_predictor_save_path)

interface.train(6)

