import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.model_selection import GridSearchCV
from sklearn import  metrics

#from sklearn.datasets import make_blob
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import cv2
import time

#############################################
#               INIT PARAMETERS
#2-11
measurement = 2
mirror = False      #mirroring the space
turnON_rectangleLabelization = True
turnON_figuredOutput = True

quantile = np.linspace(0.17,0.4,num=30)

focusedFrame = 41
startFrame = focusedFrame
finalFrame = focusedFrame+1        #to the end of Frame is value: 999

#step Frame
frame_for_Analysation = 10
stepFrame = 1
##########################################
path_outputLABELfile = f'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/mean_shift_project/mean-shift_v1/VisualizationOutputs/m{measurement}'
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
focusedFrame = 0

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vid_fileName = f"mean-shift_v1/visualizedOutput_mean-shift_v1/m{measurement}"+path.split('/')[-1].split('.')[-2]+".mp4"
video_writer = cv2.VideoWriter(vid_fileName, fourcc, 30, (640, 480))
images = []
colors = ["#dede00", "#377eb8", "#a701bf", "#b731bf", "#c761bf", "#d791bf", "#e801bf", "#f881ff"]
#markers = ['$1$', '$2$', '$3$', '$4$', '$5$', '$6$','$7$', '$8$']
markers = ["v", "^", "<", ">", "1", "2", "3", "4"]
# colors = ["#dede00", "#377eb8", "#a701bf"]
# markers = ["x", "o", "^"]
data_out = np.empty((0,3))

if(finalFrame == 999):
 lastFrame = data_all[-1,0]
else:
 lastFrame = finalFrame

startTime_total = time.process_time()
i=startFrame
data_frame = np.empty((0,10))

for j in range (quantile.shape[0]):
    # Prepare data
    #startFrame = i = 2
    #endFrame = lastFrame = 3
    #i = frame_for_Analysation
    i = startFrame
    #endFrame = lastFrame = frame_for_Analysation + 1
    endFrame = lastFrame = finalFrame
    while (i < lastFrame):
        startTime = time.process_time()
        focusedFrame = i
        # selection on focusedFrame data
        indices = np.argwhere(data_all[:, 0] == focusedFrame)
        indices = np.squeeze(indices)
        data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1),
                          np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))
        # ***********
        centers = [[1, 1], [-1, -1], [1, -1]]
        print(quantile[j])
        bandwidth = estimate_bandwidth(data, quantile=quantile[j])
        print("bandiwdth: %f", bandwidth)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        plt.clf()
        plt.figure(1)
        plt.xlim(-5, 5)
        plt.ylim(0, 10)
        a = np.empty((0, 3))
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(data[my_members, 0], data[my_members, 1], markers[k], color=col)
            plt.plot(
                cluster_center[0],
                cluster_center[1],
                markers[k],
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=14,
            )
            output_ML = data[my_members, :]
            marks = np.full((output_ML.shape[0],), k)
            a = np.row_stack((a, (np.column_stack((output_ML, marks)))))


        # ***********
        # save the plot as an image
        fig = plt.gcf()
        plt.title("number of estimated clusters : %d \n in frame: %d  quantile: %f estBandwidth: %f" % (n_clusters_, focusedFrame, quantile[j], bandwidth))
        #fig.savefig(f"visualizedOutput_mean-shift_v1/m{measurement}/frame_{i:04d}_quantile_{quantile[j]}.png")
        fig.savefig(f"visualizedOutput_mean-shift_v1/{j}_count_frame_{i:04d}_quantile_{quantile[j]}.png")
        marker_frame = np.full((a.shape[0],), focusedFrame)
        sorted_indices_1 = np.argsort(a[:, 0])[::1]  # sorting by posX
        sorted_arr_1 = a[sorted_indices_1, :]
        data_frame_raw = data_all[indices[0]:indices[-1] + 1, :]
        sorted_indices_2 = np.argsort(data_frame_raw[:, 2])[::1]
        sorted_arr_2 = data_frame_raw[sorted_indices_2, :]

        outFrameArr = np.column_stack((sorted_arr_2, sorted_arr_1[:, 2]))
        sorted_indices_data_frames = np.argsort(outFrameArr[:, 1])[::1]  # sorting by detObj
        sorted_arr_data_frames = outFrameArr[sorted_indices_data_frames, :]
        # data_frame = np.row_stack((data_frame,outFrameArr))
        data_frame = np.row_stack((data_frame, sorted_arr_data_frames))
        endTime = time.process_time()
        print(f"Current progress: %d/%d quantile: {quantile[j]} - Duration: %.5f" % (focusedFrame, lastFrame, (endTime - startTime)))
        i = i + 1

    # Verification:
    n = 0
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    print("---- %f ----" %quantile[j])
    while n < data_frame.shape[0]:
        if (data_frame[n, 9] == data_frame[n, 8] and data_frame[n, 9] == 1):  # True positive
            TP = TP + 1
        elif (data_frame[n, 9] == data_frame[n, 8] and data_frame[n, 9] == 0 ):  # True negative
            TN = TN + 1
        elif (data_frame[n, 9] != data_frame[n, 8] and data_frame[n, 9] == 1):  # False positive
            FP = FP + 1
        elif (data_frame[n, 9] != data_frame[n, 8] and data_frame[n, 9] == 0):  # False positive
            FN = FN + 1
        n = n + 1

    print("True positive:  %d" % TP)
    print("True negative:  %d" % TN)
    print("False positive: %d" % FP)
    print("False negative: %d" % FN)
    j = j + 1

endTime_total = time.process_time()
print("\tTotal time: %.5f" %(endTime_total-startTime_total))

header = "frame,detObj,posX,poxY,posZ,v,snr,noise,label,cluster"
np.savetxt('ML_meanShift_outuput.csv',data_frame,delimiter=',',header=header)


