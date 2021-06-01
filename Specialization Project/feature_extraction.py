'''
The feature extraction process is influenced by this Python for microscopists tutorial series: https://github.com/bnsreenu/python_for_microscopists
'''
# Imports
import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import pickle
import os
from tqdm import tqdm
from ipynb.fs.full.Prepare_data import load_data

# Functions
def extract_features(img_list, mask_list):
    image_dataset = pd.DataFrame()
    for i in tqdm(range(len(img_list))):
        df = pd.DataFrame()
        input_img = cv2.imread(img_list[i])
        mask = cv2.imread(mask_list[i])
        
        if input_img.ndim == 3 and input_img.shape[-1] == 3:
            img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        elif input_img.ndim == 2:
            img = input_img
        else:
            raise Exception("The module only works with grayscale and RGB images!")
        image_name = img_list[i].split("/")[-1].split(".")[0]
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values
        df['Image_Name'] = image_name + str(i)
        
        # Gabor features
        num = 1
        kernels = []
        for theta in range(2):
            theta = theta/4. * np.pi
            for sigma in (1, 3):
                for lamda in np.arange(0, np.pi, np.pi/4):
                    for gamma in (0.05, 0.5):
                        gabor_label = 'Gabor' + str(num)
                        ksize=5
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                        kernels.append(kernel)
                        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                        filtered_img = fimg.reshape(-1)
                        df[gabor_label] = filtered_img
                        num +=1

        # Canny edge
        edges = cv2.Canny(img, 100,200)
        edges1 = edges.reshape(-1)
        df['Canny Edge'] = edges1

        # Roberts edge
        edge_roberts = roberts(img)
        edge_roberts1 = edge_roberts.reshape(-1)
        df['Roberts'] = edge_roberts1

        # Sobel edge
        edge_sobel = sobel(img)
        edge_sobel1 =edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1

        # Scharr edge
        edge_scharr = scharr(img)
        edge_scharr1 =edge_scharr.reshape(-1)
        df['Scharr'] = edge_scharr1

        # Scharr edge
        edge_prewitt = prewitt(img)
        edge_prewitt1 =edge_prewitt.reshape(-1)
        df['Prewitt'] = edge_prewitt1

        # Gaussian with sigma=3
        gaussian_img = nd.gaussian_filter(img, sigma=3)
        gaussian_img1 = gaussian_img.reshape(-1)
        df['Gaussian s3'] = gaussian_img1

        # Gaussian with sigma=7
        gaussian_img2 = nd.gaussian_filter(img, sigma=7)
        gaussian_img3 = gaussian_img2.reshape(-1)
        df['Gaussian s7'] = gaussian_img3

        # Median with sigma = 3
        median_img = nd.median_filter(img, size=3)
        median_img1 = median_img.reshape(-1)
        df['Median s3'] = median_img1
        
        # Add mask
        if mask.ndim == 3 and mask.shape[-1] == 3:
            label = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif mask.ndim == 2:
            label = mask
        else: raise Exception("The module only works for grayscale and RGB images!")

        label_values = label.reshape(-1)
        df['Label_Value'] = label_values
        df['Mask_Name'] = image_name + "_mask" + str(i)

        image_dataset = image_dataset.append(df)
        # Save dataframe after 500 iterations
        if ( i % 500 == 0):
            image_dataset.to_pickle("D:/Data/Deep_learning/oil_wet/256/new_data/features/features_test_" + str(i) + ".pkl")
            image_dataset = pd.DataFrame()
            
    image_dataset.to_pickle("D:/Data/Deep_learning/oil_wet/256/new_data/features/features_test_" + str(i) + ".pkl")
    
    return image_dataset

# Main method
if __name__ == "__main__":
    path = "D:/Data/Deep_learning/oil_wet/256/new_data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    dataset_train = extract_features(train_x, train_y)
    dataset_test = extract_features(test_x, test_y)
