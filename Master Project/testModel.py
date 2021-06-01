# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:38:46 2021

@author: Ådne
"""
from tensorflow import keras
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, jaccard_score
from tqdm import tqdm
import seaborn as sns
import json
from nnUtils import process_mask

def preprocessImg(img):
    img = img / 255.0
    img = img.astype("float32")
    #img = tf.expand_dims(img, axis=0)
    return img

def readImage(path):
    with open(path, 'rb') as f:
        img = np.load(f)
    img = preprocessImg(img)
    return img
   
def getLossAndAccuracyPlot(modelPath, fileName):
    f = open(modelPath + fileName)
    data = json.load(f)
    
    plt.rcParams.update({'font.size': 28})
    
    lossFig = plt.figure(figsize=(15,15))
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.legend(['train', 'validation'])
    plt.title("Model loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    accFig = plt.figure(figsize=(15,15))
    plt.plot(data['categorical_accuracy'])
    plt.plot(data['val_categorical_accuracy'])
    plt.legend(['train', 'validation'])
    plt.title("Model categorical accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    return lossFig, accFig
       
def getStats(prediction, mask):
    mask = mask.flatten()
    prediction = prediction.flatten()
    acc = accuracy_score(y_pred = prediction, y_true=mask,)
    jaccard= jaccard_score(y_pred = prediction, y_true=mask, average='micro')
    f1 = f1_score(y_pred = prediction, y_true=mask, average=None)
    precision = precision_score(y_pred=prediction, y_true=mask, average=None)
    recall = recall_score(y_pred=prediction, y_true=mask, average=None)
    print(f"Accuracy score: {acc} \nJaccard score: {jaccard}\nF1 score: {f1}\nPrecision score: {precision}\nRecall score: {recall}")
    return acc, jaccard, f1, precision, recall
    
def makePredictions(testPath, modelPath, modelName):
    # Load model
    model = keras.models.load_model(modelPath + modelName)
    # Loop through all test files
    imgFolder = "images/"
    maskFolder = "masks/"
    imgPath = testPath + imgFolder
    maskPath = testPath + maskFolder
    # Define empty arrays to store all predictions and masks in
    allPredictions = []
    allMasks = []
    allImages = []
    
    imgPaths = [
        os.path.join(os.getcwd(), imgPath, x)
        for x in os.listdir(imgPath)]

    maskPaths = [
        os.path.join(os.getcwd(), maskPath, x)
        for x in os.listdir(maskPath)]
    # Loop image files and make predictions
    for path in tqdm(imgPaths):
        img = readImage(path)
        allImages.append(img)
        p = model.predict(np.expand_dims(img, axis=0))
        p = np.argmax(p, axis=-1)
        if p.ndim ==4:
            p = p[0,:,:,:]
        elif p.ndim == 3:
            p = p[0,:,:]
        p = p.astype(np.uint8)
        allPredictions.append(p)
    # Load masks
    for path in tqdm(maskPaths):
        mask = process_mask(path)
        allMasks.append(mask)

    # Convert to numpy arrays
    allPredictionsNp = np.array(allPredictions)
    allMasksNp = np.array(allMasks)
    allImagesNp = np.array(allImages)
    return allPredictionsNp, allMasksNp, allImagesNp

def createConfusionMatrix(p, mask):
    axis_labels = ["Oil", "Solid", "Water"]
    conf_mat = confusion_matrix(y_true=mask.flatten(), y_pred=p.flatten(), normalize='true')
    ax = sns.heatmap(conf_mat, annot=True, xticklabels=axis_labels, yticklabels=axis_labels, cbar=False, square=True)
    axes = plt.gca()
    axes.xaxis.label.set_size(13)
    axes.yaxis.label.set_size(13)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    fig = ax.get_figure()
    return fig

def reportStats(path, acc, jaccard, f1, recall, precision):
    fileName = "stats.txt"
    f = open(path + fileName, 'w+')
    f.write(f"Accuracy score: {acc}\n")
    f.write(f"Jaccard score: {jaccard}\n")
    f.write(f"F1 score: {f1}\n")
    f.write(f"Recall score: {recall}\n")
    f.write(f"Precision score: {precision}\n")
    f.close()

def testModel(testPath, modelPath, modelName):
    # Create folder
    folderPath = "D:/Master/Data/deep_learning/test/testResults/" + modelName.split('.')[0]
    os.mkdir(folderPath)
    print(f"Created folder {modelName.split('.')[0]} in the test directory")
    allPredictions, allMasks, allImages = makePredictions(testPath, modelPath, modelName)
    print(f"Successfully made {allPredictions.shape[0]} predictions.")
    conf_mat = createConfusionMatrix(p=allPredictions, mask=allMasks)
    print("Successfully created confusion matrix")
    conf_mat.savefig(folderPath + "/" + modelName.split('.')[0]+"_conf_mat.png")
    print("Successfully saved confusion matrix")
    
    acc, jaccard, f1, precision, recall = getStats(allPredictions, allMasks)
    reportStats(folderPath + "/", acc, jaccard, f1, recall, precision)
    
    lossFig, accFig = getLossAndAccuracyPlot(modelPath, modelName.split('.')[0] + ".json")
    lossFig.savefig(folderPath + "/lossPlot.png")
    accFig.savefig(folderPath + "/accuracyPlot.png")
    
   

if __name__ == '__main__':
    testPath = "D:/Master/Data/deep_learning/extendedDataset/3D/allData/test/"
    modelPath = "models/"
    modelName = "2DMultiResUNetDropoutExtendedDataset100Epochs.h5"
    testModel(testPath, modelPath, modelName)
   
     
