# Imports
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Functions
def splitDF(df):
    X_train = df.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis=1)
    Y_train = df["Label_Value"].values
    return X_train, Y_train

def trainModel(model, path, df_name_list, batch_size, class_weight):
    for df_name in df_name_list:
        print(f"Starting training on: {df_name}")
        df = pd.read_pickle(path + df_name)
        indices = getIndices(df.shape[0], batch_size)
        for i in range(len(indices)-1):
            df_batch = df.iloc[indices[i]:indices[i+1]]
            x_train, y_train = splitDF(df_batch)
            sample_weights = compute_sample_weight(class_weight=class_weight, y=y_train)
            model.partial_fit(x_train, y_train, classes=np.unique(y_train), sample_weight=sample_weights)
            del x_train, y_train, df_batch
    return model

def getIndices(length, batch_size):
    indices = []
    index = 0
    at_end = False
    indices.append(index)
    while not at_end:
        if index + batch_size >= length:
            indices.append(length)
            at_end = True
        else:
            index = index + batch_size
            indices.append(index)
            
    return indices
    
        
def saveModel(model, model_name):
    pickle.dump(model, open("models/" + model_name, 'wb'))

# Main method
if __name__ === "__main__":
    # Paths and filenames
    path = "D:/Data/Deep_learning/oil_wet/256/new_data/features/"
    features = ["features_1500.pkl", "features_2000.pkl","features_0.pkl", "features_500.pkl", "features_1000.pkl",  "features_2853.pkl" ]
    features_test = ["features_test_0.pkl", "features_test_31.pkl"]

    # Class weights
    class_weight={0:4, 1:1, 2:1.5}

    # Define model
    sgd_model = SGDClassifier(loss='hinge', random_state=42, class_weight=class_weight)
    model = trainModel(model=sgd_model, path=path, df_name_list=features_2, batch_size=1000000, class_weight=class_weight)
    
    # Save model
    model_name = "sgd.h5"
    saveModel(model=model, model_name=model_name)
    
 