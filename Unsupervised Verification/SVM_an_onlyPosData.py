# import libraries

import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from numpy import where
import numpy as np
import cv2
import time

# path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer2.csv'
path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/v2/mer2_LABELIZED.csv'
# path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/v2/mer3_LABELIZED.csv'
# path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/v2/mer4_LABELIZED.csv'
# path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/v2/mer5_LABELIZED.csv'


data = np.genfromtxt(path, delimiter=',', skip_header=1)
data_all = np.genfromtxt(path, delimiter=',', skip_header=1)

with open('data.csv', 'w') as f:
    pass

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vid_fileName = "SVM_anomalyDET" + path.split('/')[-1].split('.')[-2] + ".mp4"
video_writer = cv2.VideoWriter(vid_fileName, fourcc, 30, (640, 480))

startTime_total = time.process_time()
i = 0
# lastFrame = data_all[-1,0]
lastFrame = 10

inlier_fedback = np.empty((0, 1))
outlier_feedback = np.empty((0, 1))

data_result = np.empty((0, 10))
frameOffset = 0
# Prepare data
while (i < lastFrame):
    data_result_frame = np.empty((0,10))
    startTime = time.process_time()
    focusedFrame = i
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1), np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))
    #data = data_all[indices[0]:indices[-1], :]

    # Modelling
    # model specification
    mod = 3
    model = OneClassSVM(kernel='rbf', gamma=0.2, nu=0.02).fit(data)
    # Prediction
    data_prediction = model.predict(data)

    # Filtering anomalies
    # filter outlier index
    inlier_index = where(data_prediction != -1)
    outlier_index = where(data_prediction == -1)  # filter outlier values

    # VARIABLE   FOR     CONCATENATE
    outlier_values = np.array(pd.DataFrame(data).iloc[outlier_index])
    # outlier_values = np.delete(outlier_values,outlier_values.shape[1]-1,axis=1)
    label_out = np.zeros((outlier_values.shape[0], 1))
    outlier_values = np.column_stack((outlier_values, label_out))

    inlier_values = np.array(pd.DataFrame(data).iloc[inlier_index])
    # inlier_values = np.delete(inlier_values,inlier_values.shape[1]-1,axis=1)
    label_in = np.ones((inlier_values.shape[0], 1))
    inlier_values = np.column_stack((inlier_values, label_in))

    c = 0
    index_inliner = np.empty((1,1))
    a = np.empty((0,9))
    a1 = np.empty((0, 9))
    while c < inlier_values.shape[0]:
        a2 = np.where(np.logical_and((data_all[indices[0]:indices[-1] + 1, 2] == inlier_values[c, 0]) , (data_all[indices[0]:indices[-1] + 1, 3] == inlier_values[c, 1])))
        a1 = data_all[a2[0]]
        a = np.row_stack((a,a1))
        c = c + 1


    zeroArray = np.zeros(a.shape[0])
    a = np.column_stack((a,zeroArray))
    sorted_indices_1 = np.argsort(a[:,1])[::1]
    sorted_arr_1 = a[sorted_indices_1,:]
    unique_arr_1 = np.unique(sorted_arr_1,axis=0)

    c = 0
    a = np.empty((0, 9))
    a1 = np.empty((0, 9))
    while c < outlier_values.shape[0]:
        a2 = where(np.logical_and((data_all[indices[0]:indices[-1] + 1, 2] == outlier_values[c, 0]) , (data_all[indices[0]:indices[-1] + 1, 3] == outlier_values[c, 1])))
        a1 = data_all[a2[0]]
        a = np.row_stack((a,a1))
        c = c + 1

    oneArray = np.ones(a.shape[0])
    a = np.column_stack((a, oneArray))
    sorted_indices_2 = np.argsort(a[:,1])[::1]
    sorted_arr_2 = a[sorted_indices_2,:]
    unique_arr_2 = np.unique(sorted_arr_2,axis=0)
    #data_result_frame = np.row_stack((unique_arr_1,unique_arr_2))
    data_result = np.row_stack((data_result,np.row_stack((unique_arr_1,unique_arr_2))))
    print("\nframe: %d" % i)
    print("total processed length: %d" % (inlier_values.shape[0] + outlier_values.shape[0]))
    print("length of frame data: %d" % data.shape[0])

    plt.clf()
    #b1 = plt.scatter(data[:, 0], data[:, 1])
    #b2 = plt.scatter(outlier_values[:, 0], outlier_values[:, 1], c="r")
    b1 = plt.scatter(unique_arr_1[:,2],unique_arr_1[:,3])           #inlier
    b2 = plt.scatter(unique_arr_2[:, 2], unique_arr_2[:, 3],c='red')        #outlier


    # save the plot as an image
    fig = plt.gcf()
    # To vid
    # Save the figure as a PNG image
    fig.savefig(f"img/frame_{i:04d}.png")
    # Load the image and write it to the video file
    img = cv2.imread(f"img/frame_{i:04d}.png")
    video_writer.write(img)
    # time process of evaluation
    endTime = time.process_time()
    print("Current progress: %d/%d - Duration: %.5f" % (focusedFrame, lastFrame, (endTime - startTime)))
    i = i + 1

video_writer.release()
endTime_total = time.process_time()
print("Total time: %.5f" % (endTime_total - startTime_total))
header = "frame, detObj, x,y,z,v,snr,noise,label,inlier"
np.savetxt('merge.csv', data_result, delimiter=',', header=header)

##verification