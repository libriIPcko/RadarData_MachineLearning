import numpy as np
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets import make_blob
import matplotlib.pyplot as plt
import cv2

path_1440frames = 'C:/Users/bob\Documents/build-RadarVisualizer-Desktop_Qt_6_4_2_MinGW_64_bit-Release/release/parse_script/ParsedData/parsOut_5.4__22_12_5.csv'
path_199frames = 'C:/Users/bob\Documents/build-RadarVisualizer-Desktop_Qt_6_4_2_MinGW_64_bit-Release/release/parse_script/ParsedData/parsOut_16.3__19_5_43.csv'
data = np.genfromtxt(path_199frames,delimiter=',',skip_header=1,dtype='f8')
focusedFrame = 151
all_data = np.genfromtxt(path_199frames,delimiter=',',skip_header=1,dtype='f8')


print("array size of rows and columns: ", all_data.shape)
print("number of rows: ", all_data.shape[0])
print("number of columns: ", all_data.shape[1])

#add next column
#new_col = np.random.rand(all_data.shape[0],1)
#all_data = np.hstack((all_data, new_col))

#figure the dependencies of the chosen column
# in next interval of rows: <1;200>
#np.savetxt("foo.csv", all_data, delimiter=",")
#plt.figure(1)
#plt.scatter(all_data[1:200,1],all_data[1:200,8])
#plt.show()


# Create a figure and axes
fig, ax = plt.subplots()
# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("output_video.mp4", fourcc, 30, (640, 480))

min = 1
max = 56200
length = max-min
var = 0
var_new = 0
var_start = 0
for i in range(length):
    var = all_data[i,0]
    var_new = all_data[i+1,0]
    print(var, '/',var_new)
    if(var != var_new):
        ax.clear()
        # Plot the data
        ax.scatter(all_data[var_start:i, 2], all_data[var_start:i, 3], label="space")
        # Add legend and labels
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # Save the figure as a PNG image
        fig.savefig(f"DataOperation/frame_{i:04d}.png")
        # Load the image and write it to the video file
        img = cv2.imread(f"DataOperation/frame_{i:04d}.png")
        video_writer.write(img)
        # Print progress
        print(f"Processed frame {i + 1} of {length}")
        var_start = i
video_writer.release()

#selection on focusedFrame data
#indices = np.argwhere(data[:,0] == focusedFrame)
#indices = np.squeeze(indices)
#data_posX_focusedFrame = np.array(data[indices[0]:indices[-1]+1,2])
#data_posY_focusedFrame = np.array(data[indices[0]:indices[-1]+1,3])
#x = data_posX_focusedFrame
#y = data_posY_focusedFrame
#data = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

#for i in range(data.size()):
#    print(data[i,:])