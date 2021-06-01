'''
Splits images and changes the color values
'''
# Imports
import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Functions
def changePixelValues(img):
    img2 = np.empty_like(img)
    img2[img==0] = 2
    img2[img == 2] = 1
    img2[img == 3] = 0
    return img2

# Function for splitting image
def splitTo128(img):
    results = []
    for i in range(2):
        for j in range(2):
            img2 = img[i*128 : (i+1)*128, j*128 : (j+1)*128][:]
            results.append(img2)
    return results

def splitTo256(img):
    results = []
    for i in range(3):
        for j in range(3):
            img2 = img[i*128 : (i*128) + 256, j*128 : (j*128) + 256 ][:]
            results.append(img2)
    return results

        
      
# Main method
if __name__ == "__main__":
    
    # Path and filenames
    image_name = "Cropped.tif"
    mask_name = "Cropped_High_Contrast_Final.tif_annotation.ome.tiff"
    path = "D:/Data/Oil_wet/Stacks/"
    
    # Read images
    multi_img_train = cv2.imreadmulti(path + image_name)
    multi_img_label = cv2.imreadmulti(path + mask_name )
    train_img_list = multi_img_train[1][:]
    label_img_list = multi_img_label[1][:]
        
    # Change pixel values of all masks (For less confusing color choice)
    masks = []
    for image in label_img_list:
        img = changePixelValues(image)
        masks.append(img)
    
    # Split images to 256 bit images and save
    for i in range(len(train_img_list)):
        train_images = splitTo256(train_img_list[i])
        mask_images = splitTo256(masks[i])
        for j in range(len(train_images)):
            cv2.imwrite("D:/Data/Deep_learning/oil_wet/256/images/" + "image_" + str(i) + "_" + str(j) +".tif", train_images[j])
            cv2.imwrite("D:/Data/Deep_learning/oil_wet/256/masks/" + "mask_" + str(i) + "_" + str(j) +".tif", mask_images[j])
            
    


    
    
