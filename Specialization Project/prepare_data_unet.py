# Imports
import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split

# Functions
def process_data(data_path):
    # Read names
    image_names = next(os.walk(data_path + "/images"))[2]
    mask_names = next(os.walk(data_path + "/masks"))[2]
    # Create lists of image names and mask names
    images = [os.path.join(data_path, f"images/{name}") for name in image_names]
    masks = [os.path.join(data_path, f"masks/{name}") for name in mask_names]
    return images, masks
    
    
def load_data(path):
    # Load training and test data
    test_path = os.path.join(path, "test")
    train_x, train_y = process_data(path)
    test_x, test_y = process_data(test_path)
    # Split into training and validation data
    train_x, valid_x = train_test_split(train_x, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=0.2, random_state=42)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(x):
    # Converts images to floats between 0 and 1
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x
    
def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.int32)
    return x

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset
    
def preprocess(x,y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        
        image = read_image(x)
        mask = read_mask(y)
        
        return image, mask
    H = 256
    W = 256  
    image, mask = tf.numpy_function(f, [x,y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 3, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 3])
    
    return image, mask


    