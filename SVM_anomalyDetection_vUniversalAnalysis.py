# import libraries
import pandas as pd
from sklearn.svm import OneClassSVM

from numpy import where
import numpy as np
import cv2
import time

#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_32_13_xwr18xx_processed_stream_2023_03_17T12_02_36_082.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_32_25_static_xwr18xx_processed_stream.csv'
path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_39_3_static_v2_xwr18xx_processed_stream.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_39_39_static_v1_xwr18xx_processed_stream.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_40_7_dynamic_xwr18xx_processed_stream.csv'
path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/statis_measurement_parsed/static_measurement_parsed/mer2.csv'
data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)
fileName = path[path.rfind('/') + 1:]

with open('data.csv', 'w') as f:
    pass

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vid_fileName = f"figure_radDat_SVM_AnomalyDet/"+"SVM_anomalyDET"+path.split('/')[-1].split('.')[-2]+".mp4"
video_writer = cv2.VideoWriter(vid_fileName, fourcc, 5, (640, 480))

startTime_total = time.process_time()

outlineOutArray = np.empty((1,3))
print(outlineOutArray)

kernel = 'sigmoid'

poly = 0
#poly_params = np.linspace(1,30,num=30)
#nu_params = np.linspace(0.5,0.994,num=100)
nu = 0.55
gamma_params = np.linspace(0.02,1,num=50)
#gamma = 0.2

for j in range (gamma_params.shape[0]):
    # Prepare data
    startFrame = i = 2
    # lastFrame = data_all[-1,0]
    endFrame = lastFrame = 3
    actual_nu = nu
    actual_gamma = gamma_params[j]
    #actual_poly = poly_params[j]
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
        model = OneClassSVM(kernel=kernel, gamma=actual_gamma, nu=actual_nu).fit(data)
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
        plt.title("Frame: %d\n params kernel:%s, gamma: %f ,nu: %f" %(i,kernel,actual_gamma,actual_nu))
        b1 = plt.scatter(data[:, 0], data[:, 1])
        b2 = plt.scatter(outlier_values[:, 0], outlier_values[:, 1], c="r")
        tempArray = np.zeros(outlier_values.shape[0])
        for k in range(outlier_values.shape[0]):
            tempArray[k] = focusedFrame
        tempArray.transpose()
        tempArray = np.concatenate((tempArray[:, np.newaxis], outlier_values), axis=1)

        # save the plot as an image
        fig = plt.gcf()
        # To vid
        # Save the figure as a PNG image
        fig.savefig(f"figure_radDat_SVM_AnomalyDet/{fileName}_pic_{j:04d}_{kernel}_gamma {actual_gamma}__nu {actual_nu}.png")
        # Load the image and write it to the video file
        img = cv2.imread(f"figure_radDat_SVM_AnomalyDet/{fileName}_pic_{j:04d}_{kernel}_gamma {actual_gamma}__nu {actual_nu}.png")
        video_writer.write(img)
        endTime = time.process_time()
        print("Current progress: %d/%d - Duration: %.5f" % (focusedFrame, lastFrame, (endTime - startTime)))
        outlineOutArray = np.concatenate((outlineOutArray, tempArray), axis=0)
        i = i + 1

video_writer.release()
endTime_total = time.process_time()
print("Total time: %.5f" % (endTime_total - startTime_total))
np.savetxt('data.csv',outlineOutArray,delimiter=',')


