import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets import make_blob
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import cv2
import time

#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_32_13_xwr18xx_processed_stream_2023_03_17T12_02_36_082.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_32_25_static_xwr18xx_processed_stream.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_39_3_static_v2_xwr18xx_processed_stream.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_39_39_static_v1_xwr18xx_processed_stream.csv'
path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_40_7_dynamic_xwr18xx_processed_stream.csv'
data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)
focusedFrame = 0

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("output_video.mp4", fourcc, 30, (640, 480))

#selection on focusedFrame data
indices = np.argwhere(data[:,0] == focusedFrame)
indices = np.squeeze(indices)
data_posX_focusedFrame = np.array(data[indices[0]:indices[-1]+1,2])
data_posY_focusedFrame = np.array(data[indices[0]:indices[-1]+1,3])
x = data_posX_focusedFrame
y = data_posY_focusedFrame
data = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

lenOfFocusedData = indices[-1]-indices[0]
print("focData/indic: %d/%d" %(data.shape[0]-1,lenOfFocusedData))

lastFrame = data_all[-1,0]
#lastFrame = 100
images = []
colors = ["#dede00", "#377eb8", "#a701bf", "#b731bf", "#c761bf", "#d791bf", "#e801bf", "#f881ff"]
#markers = ['$1$', '$2$', '$3$', '$4$', '$5$', '$6$','$7$', '$8$']
markers = ["v", "^", "<", ">", "1", "2", "3", "4"]
# colors = ["#dede00", "#377eb8", "#a701bf"]
# markers = ["x", "o", "^"]

startTime_total = time.process_time()
i=0
while(i<lastFrame):
    startTime = time.process_time()
    focusedFrame = i
    # selection on focusedFrame data
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1), np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))
    #***********
    centers = [[1, 1], [-1, -1], [1, -1]]

    bandwidth = estimate_bandwidth(data, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)



    plt.clf()
    plt.figure(1)
    plt.xlim(-5, 5)
    plt.ylim(0, 6)
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
    fig.savefig(f"figure_radDat_mean_shift/frame_{i:04d}.png")
    # Load the image and write it to the video file
    img = cv2.imread(f"figure_radDat_mean_shift/frame_{i:04d}.png")
    video_writer.write(img)
    endTime = time.process_time()
    print("Current progress: %d/%d - Duration: %.5f" %(focusedFrame,lastFrame,(endTime-startTime)))
    i=i+1
# create the GIF from the list of images
#imageio.mimsave('plots_new.gif', images, duration=0.5)
video_writer.release()
endTime_total = time.process_time()
print("Total time: %.5f" %(endTime_total-startTime_total))
