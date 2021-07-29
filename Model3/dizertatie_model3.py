# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:44:13 2021

@author: Mihai
"""

import matplotlib.pyplot as plt 
import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow
import keras
import os
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers


IMAGE_WIDTH=280
IMAGE_HEIGHT=280
IMAGE_CHANNELS=3
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 32
EPOCHS = 15

# Root path
ROOT_PATH = "E:/Dizertatie/Dataset"

# 'train' path
TRAIN = ROOT_PATH + '/train/'

# 'test' path
TEST = ROOT_PATH + '/test/'

# 'validation' path
VAL = ROOT_PATH + '/val/'
VAL_NORMAL = ROOT_PATH + "/val/normal/"
VAL_SICK = ROOT_PATH + "/val/opacity/"

label_names = ['normal', 'opacity']
label_names_ = {label_names:i for i, label_names in enumerate(label_names)}
label_names_

from skimage.io import imread
import cv2

def load_data():
    
    datasets = [TRAIN,
               TEST]
    
    output = []
    
    for dataset in datasets:
        print("Loading:", dataset)
        
        images = []
        labels = []
        
        for folder in os.listdir(dataset):
            
            if folder != '.DS_Store':


                print("Folder:", folder)
                label = label_names_[folder]

                for file in os.listdir(dataset + '/' + folder):


                        try:

                            img_path = dataset + '/' + folder + '/' + file

                            image = cv2.imread(img_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, IMAGE_SIZE)

                            images.append(image)
                            labels.append(label)



                        except Exception as e:
                            print(e, file)
                
            else:
                continue
               
                
        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')
        
        output.append((images, labels))
        
    return output

(X_train, y_train), (X_test, y_test) = load_data()

from sklearn.utils import shuffle           

X_train, y_train = shuffle(X_train, y_train, random_state=42)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print(np.unique(y_train))
print(y_train.tolist().count(0))
print(y_train.tolist().count(1))

print(np.unique(y_test))
print(y_test.tolist().count(0))
print(y_test.tolist().count(1))

from skimage.io import imread
import cv2


def read_data(path, category):
    X = []
    Y = []
    
    for file in os.listdir(path):

        if file != '.DS_Store':

            try:

                image = cv2.imread(path + file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                X.append(image)

                Y.append(label_names_[category])

            except Exception as e:
                    print(e, file)
                    
        else:
            continue
    
    return np.array(X), np.array(Y)


X_val1, y_val1 = read_data(VAL_NORMAL, 'normal')
X_val2, y_val2 = read_data(VAL_SICK, 'opacity')

print(X_val1.shape)
print(y_val1.shape)
print(X_val2.shape)
print(y_val2.shape)

# Concatenare

X_val = [X_val1, X_val2]
X_val = np.concatenate(X_val)

y_val = [y_val1, y_val2]
y_val = np.concatenate(y_val)

print("Shape final 'X_val':", X_val.shape)
print("Shape final 'y_val':", y_val.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0
X_val = X_val / 255.0

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False, 
        zca_whitening=False,  
        rotation_range = 10,  
        zoom_range = 0.2,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip = True,  
        vertical_flip=False) 

validation_datagen = ImageDataGenerator()

datagen.fit(X_train)

expert_conv = keras.applications.resnet50.ResNet50(weights = 'imagenet', include_top= False, input_shape=(280,280,3))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling2D


for layer in expert_conv.layers:
      trainable = True
      layer.trainable = trainable

tweaked_model = Sequential()
tweaked_model.add(Reshape((280,280,3)))
tweaked_model.add(expert_conv)
tweaked_model.add(GlobalAveragePooling2D())

tweaked_model.add(Dense(280, activation = 'relu')) 
tweaked_model.add(Dropout(0.3))
tweaked_model.add(Dense(128, activation = 'relu'))
tweaked_model.add(Dense(1, activation = "sigmoid"))

opt = keras.optimizers.SGD(lr=1e-4, momentum=0.8)

from keras.optimizers import Adam

adam = Adam()

tweaked_model.compile(loss = "binary_crossentropy", optimizer = opt, metrics=['accuracy'])

history = tweaked_model.fit(datagen.flow(X_train,y_train, batch_size = 32) ,epochs = 100, 
                           validation_data = validation_datagen.flow(X_val, y_val))

test_eval = tweaked_model.evaluate(X_test, y_test)

tweaked_model.save("C:/Users/Mihai/dizertatie/Model3/model3.h5'")
