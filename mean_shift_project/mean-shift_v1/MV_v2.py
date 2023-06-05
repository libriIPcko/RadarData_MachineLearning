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
measurement = 9
mirror = False      #mirroring the space
turnON_rectangleLabelization = True
turnON_figuredOutput = True

bandwidth = 0.707

startFrame = 0
finalFrame = 90        #to the end of Frame is value: 999
#step Frame
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
#mean-shift_v1\visualizedOutput_mean-shift_v1\m1
#video_writer = cv2.VideoWriter("output_video.mp4", fourcc, 30, (640, 480))
video_writer = cv2.VideoWriter(vid_fileName, fourcc, 30, (640, 480))

#selection on focusedFrame data
#indices = np.argwhere(data[:, 0] == focusedFrame)
#indices = np.squeeze(indices)
#data_posX_focusedFrame = np.array(data[indices[0]:indices[-1] + 1, 2])
#data_posY_focusedFrame = np.array(data[indices[0]:indices[-1] + 1, 3])
#x = data_posX_focusedFrame
#y = data_posY_focusedFrame
#data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
#print(data)
#lenOfFocusedData = indices[-1]-indices[0]
#print("focData/indic: %d/%d" %(data.shape[0]-1,lenOfFocusedData))

#lastFrame = data_all[-1,0]
#lastFrame = 2
#startFrame = 0
#lastFrame = 10
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
while(i<lastFrame):
    startTime = time.process_time()
    focusedFrame = i
    # selection on focusedFrame data
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1), np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))
    #***********
    centers = [[1, 1], [-1, -1], [1, -1]]

    #bandwidth = estimate_bandwidth(data, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth,min_bin_freq=4)

    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    plt.clf()
    plt.figure(1)
    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    a = np.empty((0,3))
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
        output_ML = data[my_members,:]
        marks = np.full((output_ML.shape[0],),k)
        a = np.row_stack((a,(np.column_stack((output_ML,marks)))))
    plt.title("number of estimated clusters : %d \n in frame: %d " % (n_clusters_, focusedFrame))
    #***********
    # save the plot as an image
    fig = plt.gcf()
    #To GIF
    #fig.canvas.draw()
    #image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    #image = Image.frombytes(fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    #images.append(image)

    #To vid
    # Save the figure as a PNG image
    fig.savefig(f"visualizedOutput_mean-shift_v1/m{measurement}/frame_{i:04d}.png")
    # Load the image and write it to the video file
    #img = cv2.imread(f"figure_radDat_mean_shift/frame_{i:04d}.png")
    img = cv2.imread(f"visualizedOutput_mean-shift_v1/m{measurement}/frame_{i:04d}.png")
    #video_writer.write(img)

    marker_frame = np.full((a.shape[0],),focusedFrame)
    sorted_indices_1 = np.argsort(a[:, 0])[::1]     #sorting by posX
    sorted_arr_1 = a[sorted_indices_1, :]

    data_frame_raw = data_all[indices[0]:indices[-1]+1,:]
    sorted_indices_2 = np.argsort(data_frame_raw[:,2])[::1]
    sorted_arr_2 = data_frame_raw[sorted_indices_2, :]
    outFrameArr = np.column_stack((sorted_arr_2,sorted_arr_1[:,2]))
    #data_frame = np.row_stack((data_frame, np.column_stack((marker_frame,a))))

    sorted_indices_data_frames = np.argsort(outFrameArr[:, 1])[::1]  # sorting by detObj
    sorted_arr_data_frames = outFrameArr[sorted_indices_data_frames, :]

    #data_frame = np.row_stack((data_frame,outFrameArr))
    data_frame = np.row_stack((data_frame,sorted_arr_data_frames))

    endTime = time.process_time()
    print("Current progress: %d/%d - Duration: %.5f" %(focusedFrame,lastFrame,(endTime-startTime)))
    i=i+1
# create the GIF from the list of images
#imageio.mimsave('plots_new.gif', images, duration=0.5)
#video_writer.release()
endTime_total = time.process_time()
print("Total time: %.5f" %(endTime_total-startTime_total))

header = "frame,detObj,posX,poxY,posZ,v,snr,noise,label,cluster"
np.savetxt('ML_meanShift_output.csv',data_frame,delimiter=',',header=header)


#Verification:
n = 0
TN = 0
TP = 0
FP = 0
FN = 0
x = 0
while n < data_frame.shape[0]:
    if (data_frame[n,9] == data_frame[n,8] and (data_frame[n,9] == 1 or data_frame[n,9] == 1)):           # True positive
        TP = TP + 1
    elif (data_frame[n,9] == data_frame[n,8] and (data_frame[n,8] == 2 or data_frame[n,8] == 3)):         # True negative
        TN = TN + 1
    elif (data_frame[n,9] != data_frame[n,8] and data_frame[n,8] == 1):        # False positive
        if (data_frame[n, 8] == 0):
            x = x + 1
        else:
            FP = FP +1
    elif (data_frame[n,9] != data_frame[n,8] and (data_frame[n,8] == 2 or data_frame[n,8] == 3)):        # False positive
        if (data_frame[n, 8] == 0):
            x = x + 1
        else:
            FN = FN +1

    n = n + 1

print("True positive:  %d" %TP )
print("True negative:  %d" %TN )
print("False positive: %d" %FP )
print("False negative: %d" %FN )
verSub = TP + TN + FP + FN
Precision = TP / (TP+FP)
Recall = TP / (TP + FN)
F1_score = 2* (Precision*Recall)/(Precision+Recall)
print("Precision:"+str(Precision))
print("Recall:"+str(Recall))
print("F1_score:"+str(F1_score))
print(str(data_frame.shape[0]-verSub))
print("Total status: "+ str(verSub) + "/" + str(data_frame.shape[0]))