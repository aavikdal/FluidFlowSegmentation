# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:01:47 2021

@author: Ådne
"""

from nnUtils import process_2D_image, process_mask
import os
from sklearn.model_selection import train_test_split
from augmentation2D import get2DAugmentation
from tensorflow import keras
import tensorflow as tf
from UNet2DDropout import build_unet
from multiResUNet2D import MultiResUnet 
import json

# Define model name
modelName = "2DMultiResUNetDropoutExtendedDataset100Epochs.h5"

# Define paths to images and masks
path = "D:/Master/Data/deep_learning/extendedDataset/2D/allData/"
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

print(f"Images: {len(img_paths)}")
print(f"Masks: {len(mask_paths)}")

# Build train and validation datasets consisting of paths
x_train, x_val, y_train, y_val = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)
print(f"Number of training images: {len(x_train)} \nNumber of validation images: {len(x_val)}")

def train_preprocessing(x, y):
    def f(x,y):
        x = x.decode()
        y = y.decode()
        x = process_2D_image(x)
        y = process_mask(y)
        
        # ADD AUGMENTATION FUNCTION HERE
        x, y = get2DAugmentation(img_slice=x, mask_slice=y, augProb=0.5)
        return x, y
    
    x, y = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    y = tf.one_hot(y, depth=3, dtype=tf.int32 )
    x.set_shape([H,W,1])
    y.set_shape([H,W,3])
    print(x.shape, y.shape)
    return x,y

def val_preprocessing(x,y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        x = process_2D_image(x)
        y = process_mask(y)
        return x.astype('float32'), y.astype('int32') 
    x, y = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    y = tf.one_hot(y, depth=3, dtype=tf.int32 )
    x.set_shape([H,W,1])
    y.set_shape([H,W,3])
    return x,y

# Define data loaders   
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 32
H, W = 256, 256

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
shape = (256,256,1)
num_classes = 3    
#model = build_unet(shape, num_classes)
model = MultiResUnet(height=256, width=256, n_channels=1)

# Train model
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

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

epochs = 100
history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb])

with open(modelName.split('.')[0] + '.json', 'w') as file:

    json.dump(history.history, file)
    


    