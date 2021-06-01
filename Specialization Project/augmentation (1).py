# Imports
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip, GaussNoise, Blur, Downscale
import os


# Functions
def load_data(path):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))
    return images, masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def augment_data(images, masks, save_path, augment):
    H = 256
    W = 256
    
    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("\\")[-1].split('.')
        image_name = name[0]
        image_extn = name[1]
        
        name = y.split("\\")[-1].split('.')
        mask_name = name[0]
        mask_extn = name[1]
        
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        
        print(x.shape, y.shape)
        if augment == True:
            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
            
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]
            
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]
            
            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]
            
            aug = GaussNoise(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]
            
            aug = Blur(p=1.0)
            augmented = aug(image=x, mask=y)
            x6 = augmented["image"]
            y6 = augmented["mask"]
            
            aug = Downscale(p=1.0)
            augmented = aug(image=x, mask=y)
            x7 = augmented["image"]
            y7 = augmented["mask"]
            
            save_images = [x, x1, x2, x3, x4, x5, x6, x7]
            save_masks = [y, y1, y2, y3, y4, y5, y6, y7]
        else: 
            save_images = [x]
            save_masks = [y]
            
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W,H))
            m = cv2.resize(m,(W, H))
            tmp_img_name = f"{image_name}_{idx}.{image_extn}"
            tmp_msk_name = f"{mask_name}_{idx}.{mask_extn}"
                
            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_img_name)
                
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            idx += 1

# Main method
if __name__ == "__main__":
    # Load images
    pathAugment = "D:/Data/Deep_learning/oil_wet/256"
    images, masks = load_data(pathAugment)
    print(f"Original images: {len(images)} - Original masks: {len(masks)}")
    
    # Create folder to save augmented images in
    create_dir("D:/Data/Deep_learning/oil_wet/256/augmented/images")
    create_dir("D:/Data/Deep_learning/oil_wet/256/augmented/masks")

    # Define path to save images at
    save_path = "D:/Data/Deep_learning/oil_wet/256/augmented"
    # Augment images
    augment_data(images, masks, save_path, augment=True)
    images, masks = load_data(save_path)
    print(f"Augmented images: {len(images)} - Augmented masks: {len(masks)}")

    
