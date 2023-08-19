# This is a sample Python script.
import sys

from DatasetsOp import Datasets
import  os
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from numpy import where
import numpy as np
#import cv2
import time

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#np.set_printoptions(edgeitems=8)
np.set_printoptions(linewidth=600)  #change setting of printing to the terminal
datSets = Datasets()
data = datSets.LoadDataset(datSets.path_labelized_m4)
print(data.shape[0])
print(data.shape[1])
idx = 0
selectedFrame = 0
dataFrame = []
while idx < data.shape[0]:
    if data[idx,0] == selectedFrame:
        dataFrame.append(data[idx,:])
    idx += 1

os.chdir(datSets.scriptPath)
np.savetxt('temp_data.csv',dataFrame,delimiter=',')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
