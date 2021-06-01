# Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from ipynb.fs.full.Prepare_data import load_data, tf_dataset
from ipynb.fs.full.Build_UNet import build_unet
import matplotlib.pyplot as plt

# Main method
if __name__ == "__main__":
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load dataset
    path = "D:/Data/Deep_learning/oil_wet/256/new_data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    # Hyperparameters
    shape = (256, 256, 3)
    num_classes = 3
    lr = 1e-4
    batch_size = 8
    epochs = 30  
    
    # Build and compile model
    model = build_unet(shape, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', 'categorical_accuracy'])
    
    # Process dataset
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    
    #Calculate number of steps
    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint("model.h5", verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3,factor=0.1, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    ]
    
    # Train model
    model.fit(train_dataset,
             steps_per_epoch=train_steps,
             validation_data=valid_dataset,
             validation_steps=valid_steps,
              epochs=epochs,
              callbacks=callbacks
             )
