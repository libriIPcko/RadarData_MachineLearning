import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets import make_blob
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import time

path_1440frames = 'C:/Users/bob\Documents/build-RadarVisualizer-Desktop_Qt_6_4_2_MinGW_64_bit-Release/release/parse_script/ParsedData/parsOut_5.4__22_12_5.csv'
path_1199frames = 'C:/Users/bob\Documents/build-RadarVisualizer-Desktop_Qt_6_4_2_MinGW_64_bit-Release/release/parse_script/ParsedData/parsOut_16.3__19_5_43.csv'
data = np.genfromtxt(path_1199frames,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path_1199frames,delimiter=',',skip_header=1)
focusedFrame = 0

#selection on focusedFrame data
indices = np.argwhere(data[:,0] == focusedFrame)
indices = np.squeeze(indices)
data_posX_focusedFrame = np.array(data[indices[0]:indices[-1]+1,2])
data_posY_focusedFrame = np.array(data[indices[0]:indices[-1]+1,3])
x = data_posX_focusedFrame
y = data_posY_focusedFrame
data = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

lenOfFocusedData = indices[-1]-indices[0]
#print("%d - %d" %(indices[-1],indices[0]))
print("focData/indic: %d/%d" %(data.shape[0]-1,lenOfFocusedData))


#lastFrame = data_all[-1,0]
lastFrame = 100
# Define the fixed length of the images list
# create a list of images
    # Initialize the images list with None values
#images = [None for _ in range(lastFrame+1)]
images = []
#print("size: %d" %len(images))

i=0
startTime_total = time.process_time()
while(i<lastFrame):
    startTime = time.process_time()
    focusedFrame = i
    # selection on focusedFrame data
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)

    data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1), np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))

    plt.title("Current Frame: %d/%d" %(focusedFrame, lastFrame))
    plt.scatter(data[:, 0], data[:, 1])



    # save the plot as an image
    fig = plt.gcf()
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    #image = Image.frombytes(fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    images.append(image)
    endTime = time.process_time()
    print("Current progress: %d/%d - Duration: %.5f" %(focusedFrame,lastFrame,(endTime-startTime)))
    i=i+1

# create the GIF from the list of images
imageio.mimsave('plots_new.gif', images, duration=0.1)
endTime_total = time.process_time()
print("Total time: %.5f" %(endTime_total-startTime_total))
