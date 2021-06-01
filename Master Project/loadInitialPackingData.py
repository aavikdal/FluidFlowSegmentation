# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:55:39 2021

@author: Ådne
"""
import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm

# Functions
# Change the location of the centers to fit the required diameter size
def scaleCenters(centers,radius, scalingFactor, displacement):
    newCenters = []
    radius = radius * scalingFactor
    for center in centers:
        newCenters.append([int(round(center[0] * scalingFactor)+ ceil(radius) + displacement), int(round(center[1] * scalingFactor)+ ceil(radius) + displacement),int(round(center[2] * scalingFactor)+ ceil(radius) + displacement) ])
    return newCenters

# Calculate the required size of the grid to fit all of the spheres
def calculateGridDimensions(minCenter, maxCenter, radius, desiredRadius):
    originalGrid = ceil(maxCenter - minCenter + 2* radius)
    desiredGrid = ceil(originalGrid * (desiredRadius / radius))
    return desiredGrid

# Draw the spheres in the grid
def drawSpheres(grid, centers, radius, epsilon=1.1):
    for center in tqdm(centers):
        print(center)
        a = center[0]
        b = center[1]
        c = center[2]
        for x in range(a - radius, a + radius + 1):
            for y in range(b-radius, b + radius + 1):
                for z in range(c - radius, c + radius + 1):
                    if abs((x-a)**2 +(y-b)**2 + (z-c)**2) < radius**2 + epsilon**2:
                        grid[x][y][z] = 255
    return grid
                    

if __name__ == '__main__':
    path = "D:/Master/Data/synthetic_bead_packs/initialSpherePack/"
    coordinatesName = "coordinates3.csv"
    coordinates = pd.read_csv(path + coordinatesName, header=None)
    diameter = 1.00000
    print(f"Coordinates shape : {coordinates.shape}")
    
    minimumCoordinate = min(coordinates[0].min(), coordinates[1].min(), coordinates[2].min())
    maximumCoordinate = max(coordinates[0].max(), coordinates[1].max(), coordinates[2].max())
    print(f"Minimum: {minimumCoordinate}, Maximum: {maximumCoordinate}")
    
    # Scale diameters
    originalDiameter = diameter
    originalRadius = originalDiameter / 2
    desiredDiameter = 160
    desiredRadius = int(round(desiredDiameter / 2))
    scalingFactor = ceil(desiredDiameter/originalDiameter)
    
    # Read centers into array
    centers = []
    for index, row in coordinates.iterrows():
        centers.append([row[0], row[1], row[2]])
        
    # Scale centers
    newCenters = scaleCenters(centers, originalRadius, scalingFactor, 0)

    # Create Grid
    size = calculateGridDimensions(minimumCoordinate, maximumCoordinate, originalRadius, desiredRadius )
    print(f"Size: {size}")
    grid = np.zeros([size, size, size])
    
    # Draw spheres in the grid
    grid = drawSpheres(grid, newCenters, desiredRadius)
    
    # Save grid to file
    fileName = "gridInitialPack.npy"
    with open(path + fileName, 'wb') as f:
        np.save(f, grid)
        
