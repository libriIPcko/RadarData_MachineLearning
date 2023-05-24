# import libraries
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from numpy import where
import numpy as np
import cv2
import time

#############################################
#               INIT PARAMETERS
#2-11
measurement = 10
mirror = False      #mirroring the space
turnON_rectangleLabelization = True
turnON_figuredOutput = True


startFrame = 0
finalFrame = 50         #to the end of Frame is value: 999
#step Frame
#stepFrame = 1
##########################################

if measurement == 2:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer2.csv'
    pos_x = 2.5
    pos_y = 7.8
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 3:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer3.csv'
    pos_x = 0
    pos_y = 8.8
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 4:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer4.csv'
    pos_x = -4.5
    pos_y = 4.5
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 5:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer5.csv'
    pos_x = 4
    pos_y = 5.5
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 6:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer6.csv'
    pos_x = 0
    pos_y = 1
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 7:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer7.csv'
    pos_x = 0
    pos_y = 2.5
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 8:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer8.csv'
    pos_x = 3
    pos_y = 4
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 9:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer9.csv'
    pos_x = -2
    pos_y = 3
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 10:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer10.csv'
    pos_x = 3
    pos_y = 8.8
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 11:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer11.csv'
    pos_x = 0
    pos_y = 5.5
    if mirror == True:
        pos_x = pos_x * -1


data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)

with open('data.csv', 'w') as f:
    pass

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#vid_fileName = "SVM_anomalyDET"+path.split('/')[-1].split('.')[-2]+".mp4"
#f"SVM_an_v0/meas{measurement}_frame_{i:04d}.png"
vid_fileName = f"SVM_an_v0/"+path.split('/')[-1].split('.')[-2]+".mp4"
#vid_fileName = f"SVM_an_v0/meas{measurement}+path.split('/')[-1].split('.')[-2]+".mp4"
video_writer = cv2.VideoWriter(vid_fileName, fourcc, 30, (640, 480))

i = startFrame
if finalFrame == 999:
    lastFrame = data_all[-1, 0]
else:
    lastFrame = finalFrame

startTime_total = time.process_time()
outlineOutArray = np.empty((1,3))
print(outlineOutArray)
#Prepare data
while(i<lastFrame):
    startTime = time.process_time()
    focusedFrame = i
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1), np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))

    #Modelling
    # model specification
    #model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03).fit(data)
    #mod = 1
    #model = OneClassSVM(kernel='rbf', gamma=0.2, nu=0.5).fit(data)
    mod = 3
    model = OneClassSVM(kernel='rbf', gamma=0.2, nu=0.5).fit(data)
    #Prediction
    data_prediction = model.predict(data)

    #Filtering anomalies
    # filter outlier index
    outlier_index = where(data_prediction == -1)  # filter outlier values
    #df = pd.DataFrame(data).iloc[outlier_index]
    #outlier_values = df.iloc[outlier_index]
    outlier_values = np.array(pd.DataFrame(data).iloc[outlier_index])
    plt.clf()
    #np.savetxt("data.csv",outlier_values,delimiter=',')
    b1 = plt.scatter(data[:,0], data[:,1])
    b2 = plt.scatter(outlier_values[:,0], outlier_values[:,1], c="r")
    tempArray = np.zeros(outlier_values.shape[0])
    for j in range(outlier_values.shape[0]):
        tempArray[j] = focusedFrame
    tempArray.transpose()
    tempArray = np.concatenate((tempArray[:, np.newaxis], outlier_values), axis=1)

    # save the plot as an image
    fig = plt.gcf()
    plt.ylim(0,10)
    plt.xlim(-6,6)
    # To vid
    # Save the figure as a PNG image
    fig.savefig(f"SVM_an_v0/meas{measurement}_frame_{i:04d}.png")
    # Load the image and write it to the video file
    #img = cv2.imread(f"SVM_an_v0/frame_{i:04d}.png")
    img = cv2.imread(f"SVM_an_v0/meas{measurement}_frame_{i:04d}.png")
    video_writer.write(img)
    endTime = time.process_time()
    print("Current progress: %d/%d - Duration: %.5f" % (focusedFrame, lastFrame, (endTime - startTime)))
    outlineOutArray = np.concatenate((outlineOutArray, tempArray), axis=0)
    i = i + 1
video_writer.release()
endTime_total = time.process_time()
print("Total time: %.5f" % (endTime_total - startTime_total))
np.savetxt('data.csv',outlineOutArray,delimiter=',')

