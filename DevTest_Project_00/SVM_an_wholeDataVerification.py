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
lastFrame = data_all[-1,0]
#lastFrame = 10

inlier_fedback = np.empty((0, 1))
outlier_feedback = np.empty((0, 1))

data_out = np.empty((0, 10))

# Prepare data
while (i < lastFrame):
    startTime = time.process_time()
    focusedFrame = i
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    # data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1), np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))
    data = data_all[indices[0]:indices[-1], :]

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

    data_out = np.row_stack((data_out, outlier_values, inlier_values))

    print("\nframe: %d" % i)
    print("total processed length: %d" % (inlier_values.shape[0] + outlier_values.shape[0]))
    print("length of frame data: %d" % data.shape[0])

    plt.clf()

    b1 = plt.scatter(data[:, 2], data[:, 3])
    b2 = plt.scatter(outlier_values[:, 2], outlier_values[:, 3], c="r")
    '''
    tempArray = np.zeros(outlier_values.shape[0])
    for j in range(outlier_values.shape[0]):
        tempArray[j] = focusedFrame
    tempArray.transpose()
    tempArray = np.concatenate((tempArray[:, np.newaxis], outlier_values), axis=1)
    outlineOutArray = np.concatenate((outlineOutArray, tempArray), axis=0)
    '''
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
np.savetxt('merge.csv', data_out, delimiter=',', header=header)

##verification
n = 0
TN = 0
TP = 0
FP = 0
FN = 0
while n < data_out.shape[0]:
    if (data_out[n,9] == data_out[n,8] and data_out[n,9] == 1):           # True positive
        TP = TP + 1
    elif (data_out[n,9] == data_out[n,8] and data_out[n,9] == 0):         # True negative
        TN = TN + 1
    elif (data_out[n,9] != data_out[n,8] and data_out[n,9] == 1):        # False positive
        FP = FP +1
    elif (data_out[n,9] != data_out[n,8] and data_out[n,9] == 0):        # False positive
        FN = FN +1
    n = n + 1

print("True positive:  %d" %TP )
print("True negative:  %d" %TN )
print("False positive: %d" %FP )
print("False negative: %d" %FN )