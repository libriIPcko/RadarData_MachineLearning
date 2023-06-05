# import libraries
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from numpy import where
import numpy as np
import cv2
import time

measurement = 2
mirror = False
frame_for_Analysation = 41

kernel = 'sigmoid'
gamma = 0.2 #only for rbf
coef = np.linspace(0.02,1,num=50) #only for sigmoid
nu_params = 0.8


if measurement == 1:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer1_LABELIZED.csv'
    pos_x = 0
    pos_y = 8
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 2:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer2_LABELIZED.csv'
    pos_x = 2.5
    pos_y = 7.8
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 3:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer3_LABELIZED.csv'
    pos_x = -4.5
    pos_y = 4.5
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 4:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer4_LABELIZED.csv'
    pos_x = 4
    pos_y = 5.5
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 5:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer5_LABELIZED.csv'
    pos_x = 0
    pos_y = 1
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 6:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer6_LABELIZED.csv'
    pos_x = 0
    pos_y = 2.5
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 7:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer7_LABELIZED.csv'
    pos_x = 3
    pos_y = 4
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 8:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer8_LABELIZED.csv'
    pos_x = -2
    pos_y = 3
    if mirror == True:
        pos_x = pos_x * -1
elif measurement == 9:
    path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_labelized/mer9_LABELIZED.csv'
    pos_x = 0
    pos_y = 8.8
    if mirror == True:
        pos_x = pos_x * -1

data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)
fileName = path[path.rfind('/') + 1:]

with open('data.csv', 'w') as f:
    pass

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vid_fileName = f"SVM_an_vCoefanalysis/"+path.split('/')[-1].split('.')[-2]+".mp4"
video_writer = cv2.VideoWriter(vid_fileName, fourcc, 5, (640, 480))

startTime_total = time.process_time()

outlineOutArray = np.empty((1,3))
print(outlineOutArray)

#lastFrame = data_all[-1,0]


for j in range (coef.shape[0]):
    # Prepare data
    #startFrame = i = 2
    #endFrame = lastFrame = 3
    i = frame_for_Analysation
    endFrame = lastFrame = frame_for_Analysation + 1
    while (i < lastFrame):
        startTime = time.process_time()
        focusedFrame = i
        indices = np.argwhere(data_all[:, 0] == focusedFrame)
        indices = np.squeeze(indices)
        data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1),
                          np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))

        # Modelling
        # model specification
        # model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03).fit(data)
        # mod = 1
        # model = OneClassSVM(kernel='rbf', gamma=0.2, nu=0.5).fit(data)
        mod = 3

        model = OneClassSVM(kernel=kernel, coef0=coef[j], nu=nu_params).fit(data)
        # Prediction
        data_prediction = model.predict(data)

        # Filtering anomalies
        # filter outlier index
        outlier_index = where(data_prediction == -1)  # filter outlier values
        # df = pd.DataFrame(data).iloc[outlier_index]
        # outlier_values = df.iloc[outlier_index]
        outlier_values = np.array(pd.DataFrame(data).iloc[outlier_index])
        plt.clf()
        # np.savetxt("data.csv",outlier_values,delimiter=',')
        plt.title("Frame: %d\n params kernel:%s, coef: %f ,nu: %f" %(i,kernel,coef[j],nu_params))
        b1 = plt.scatter(data[:, 0], data[:, 1])
        b2 = plt.scatter(outlier_values[:, 0], outlier_values[:, 1], c="r")
        tempArray = np.zeros(outlier_values.shape[0])
        for k in range(outlier_values.shape[0]):
            tempArray[k] = focusedFrame
        tempArray.transpose()
        tempArray = np.concatenate((tempArray[:, np.newaxis], outlier_values), axis=1)

        # save the plot as an image
        fig = plt.gcf()
        plt.ylim(0, 10)
        plt.xlim(-6, 6)
        # To vid
        # Save the figure as a PNG image
        fig.savefig(f"SVM_an_vCoefanalysis/{fileName}_pic_{j:04d}_{kernel}_coef {coef[j]}__nu {nu_params}.png")
        # Load the image and write it to the video file
        img = cv2.imread(f"SVM_an_vCoefanalysis/{fileName}_pic_{j:04d}_{kernel}_coef {coef[j]}__nu {nu_params}.png")
        video_writer.write(img)
        endTime = time.process_time()
        print("Current progress: %d/%d - Duration: %.5f" % (focusedFrame, lastFrame, (endTime - startTime)))
        outlineOutArray = np.concatenate((outlineOutArray, tempArray), axis=0)
        i = i + 1

video_writer.release()
endTime_total = time.process_time()
print("Total time: %.5f" % (endTime_total - startTime_total))
np.savetxt('data.csv',outlineOutArray,delimiter=',')

data_frame = outlineOutArray

