# import libraries
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from numpy import where
import numpy as np
import cv2
import time


#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer2.csv'
path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/v2/mer2_LABELIZED.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/v2/mer3_LABELIZED.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/v2/mer4_LABELIZED.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/v2/mer5_LABELIZED.csv'


data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)

with open('data.csv', 'w') as f:
    pass

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vid_fileName = "SVM_anomalyDET"+path.split('/')[-1].split('.')[-2]+".mp4"
video_writer = cv2.VideoWriter(vid_fileName, fourcc, 30, (640, 480))

startTime_total = time.process_time()
i=0
#lastFrame = data_all[-1,0]
lastFrame = 2

outlineOutArray = np.empty((1,3))
print(outlineOutArray)
outArray = np.empty((0,data.shape[1]+1))

inlier_fedback = np.empty((0,1))
outlier_feedback = np.empty((0,1))

print(inlier_fedback.shape[0])
print(inlier_fedback.shape[1])
print('----')
print(outlier_feedback.shape[0])
print(outlier_feedback.shape[1])
print('----')


#Prepare data
while(i<lastFrame):
    startTime = time.process_time()
    focusedFrame = i
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1), np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))

    #Modelling
    # model specification
    mod = 3
    model = OneClassSVM(kernel='rbf', gamma=0.2, nu=0.5).fit(data)
    #Prediction
    data_prediction = model.predict(data)
    print(data_prediction)
    #Filtering anomalies
    # filter outlier index
    inlier_index = where(data_prediction != -1)
    outlier_index = where(data_prediction == -1)  # filter outlier values




    #df = pd.DataFrame(data).iloc[outlier_index]
    #outlier_values = df.iloc[outlier_index]
    outlier_values = np.array(pd.DataFrame(data).iloc[outlier_index])
    inlier_index = np.array(pd.DataFrame(data).iloc[inlier_index])




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
    # To vid
    # Save the figure as a PNG image
    fig.savefig(f"img/frame_{i:04d}.png")
    # Load the image and write it to the video file
    img = cv2.imread(f"img/frame_{i:04d}.png")
    video_writer.write(img)
    endTime = time.process_time()
    print("Current progress: %d/%d - Duration: %.5f" % (focusedFrame, lastFrame, (endTime - startTime)))
    outlineOutArray = np.concatenate((outlineOutArray, tempArray), axis=0)
    i = i + 1
video_writer.release()
endTime_total = time.process_time()
print("Total time: %.5f" % (endTime_total - startTime_total))

np.savetxt(f"mer2/outlineOutArrat.csv",outlineOutArray,delimiter=',')
np.savetxt(f"mer2/outlier_values.csv",outlier_values,delimiter=',')

##verification

