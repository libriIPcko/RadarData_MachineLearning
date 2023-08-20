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
import math

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#np.set_printoptions(linewidth=600)  #change setting of printing to the terminal
datSets = Datasets()
data = datSets.LoadDataset_specFrame(datSets.listLabMeas[7],0)

#Create a scatter plot with colors based on the third value
# plt.clf()
# plt.xlim(-5,5)
# plt.ylim(0,9)
# b1 = plt.scatter(data[:,datSets.x],data[:,datSets.y],c=data[:,datSets.snr], cmap='viridis')
# cbar = plt.colorbar()
# #TCFAR =α×Np
# plt.show()

#Calculation hypotenuse of points for creation dependance of f(hypotenuse) = SNR
data_c = np.empty((data.shape[0], data.shape[1] + 1))  # Add one extra column
jdx = 0
while jdx < data.shape[0]:
    calculated_value = math.sqrt(data[jdx,datSets.x]**2+data[jdx,datSets.y]**2)
    data_c[jdx,:-1] = data[jdx, :]  #Copy existing data
    data_c[jdx,-1] = calculated_value
    jdx += 1

    #Sorting depending on sort_column_index
#Column index by which you want to sort
sort_column_index = 9
# Get the indices that would sort the array based on the specified column
sorted_indices = np.argsort(data_c[:, sort_column_index])
# Use the sorted indices to rearrange the entire array
sorted_data = data_c[sorted_indices]

plt.clf()
plt.plot(sorted_data[:,9],sorted_data[:,datSets.snr])
plt.show()

os.chdir(datSets.scriptPath)
header = "frame,detObj,x,y,z,v,snr,noise,label,hypotenuse"
np.savetxt('temp_data_hypotenuse.csv',data_c,delimiter=',', header=header)

snrTreshold = 120
data_underTreshold = []
data_aboveTreshold = []
idx = 0
print(data.shape[0])
print(data.shape[1])
while idx < data.shape[0]:
    if snrTreshold < data[idx,datSets.snr]:
        #data_aboveTreshold = np.row_stack((data_aboveTreshold,data[idx,:]))
        data_aboveTreshold.append(data[idx, :])
    else:
        #data_underTreshold = np.row_stack((data_underTreshold, data[idx, :]))
        data_underTreshold.append(data[idx, :])
    idx += 1

data_aboveTreshold = np.array(data_aboveTreshold)
data_underTreshold = np.array(data_underTreshold)

print("DAT", data_aboveTreshold.shape[0])
print("DUT", data_underTreshold.shape[0])

plt.clf()
b1 = plt.scatter(data_aboveTreshold[:,datSets.x],data_aboveTreshold[:,datSets.y],c='black')
b2 = plt.scatter(data_underTreshold[:,datSets.x],data_underTreshold[:,datSets.y],c='red')
plt.xlim(-5,5)
plt.ylim(0,9)
plt.legend([b1,b2],["ATh","UTh"])
plt.show()

os.chdir(datSets.scriptPath)
print(datSets.scriptPath)
header = "frame,detObj,x,y,z,v,snr,noise,label"
np.savetxt('temp_data.csv',data,delimiter=',', header=header)









# See PyCharm help at https://www.jetbrains.com/help/pycharm/
