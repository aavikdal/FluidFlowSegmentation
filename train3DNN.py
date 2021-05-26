# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:42:24 2021

@author: Ådne
"""
from scipy import ndimage
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from augmentation import getAugmentation
import json
from nnUtils import process_image, process_mask
from UNet3DDropout import build_unet

# Define paths to the images and masks
path = "D:/Master/Data/deep_learning/extendedDataset/3D/allData/"
img_folder = "images/"
mask_folder = "masks/"
img_path = os.path.join(path, img_folder)
mask_path = os.path.join(path, mask_folder)

img_paths = [
        os.path.join(os.getcwd(), img_path, x)
        for x in os.listdir(img_path)]

mask_paths = [
        os.path.join(os.getcwd(), mask_path, x)
        for x in os.listdir(mask_path)]

print(f"Number of images: {len(img_paths)}")
print(f"Number of masks: {len(mask_paths)}")

# Build train and validation datasets consisting of paths
x_train, x_val, y_train, y_val = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)
print(f"Number of training images: {len(x_train)} \nNumber of validation images: {len(x_val)}")

# Preprocessing functions
def train_preprocessing(x, y):  
    def f(x,y):
        x = x.decode()
        y = y.decode()
        x = process_image(x)
        y = process_mask(y)
        
        # On-line data augmentation
        x, y = getAugmentation(img=x, mask=y, augProb=0.5)
        return x.astype('float32'), y.astype('int32')
    
    x, y = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    x = tf.expand_dims(x, axis=3)
    y = tf.one_hot(y, depth=3, dtype=tf.int32 )
    return x,y

def val_preprocessing(x,y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        x = process_image(x)
        y = process_mask(y)
        return x, y
    x, y = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    x = tf.expand_dims(x, axis=3)
    y = tf.one_hot(y, depth=3, dtype=tf.int32 )
    return x,y
    
# Define data loaders
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# Set batch size
batch_size = 4

# Load training and validation data
train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
        )

validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(val_preprocessing)
        .batch(batch_size)
        .prefetch(2))
    
# Load model
shape = (256,256,16,1)
num_classes = 3
modelName = "3DMultiResNetExtendedDataset100Epochs.h5"
model = build_unet(shape, num_classes)
   
# Set learning rate schedule
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

# Compile model
model.compile(
        loss="categorical_crossentropy",
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['categorical_accuracy']
        )

# Define callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(
        modelName, save_best_only=True
        )

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

#Set number of epochs
epochs = 100

# Train model
history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb])

# Save training data in a json-file
with open(modelName.split('.')[0] + '.json', 'w') as file:

    json.dump(history.history, file)
