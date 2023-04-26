import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle, Rectangle
import time

path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer2.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_39_39_static_v1_xwr18xx_processed_stream.csv'
data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)
fileName = path[path.rfind('/') + 1:]

lastFrame = data_all[-1,0]
#lastFrame = 5
startFrame = 1
#step Frame
stepFrame = 5

#for DetObj radius
radius = .5
pos_x = 2.5   #2.5
pos_y = 8   #8

#for interfere rectangle
x_length = 3
y_length = 1
pos_x_center_interfere = 0
pos_y_center_interfere = 0

pos_x_LD = pos_x_center_interfere - x_length/2
pos_y_LD = pos_y_center_interfere

pos_x_right_interfere = pos_x_LD + x_length
pos_x_left_interfere = pos_y_LD
pos_y_UP_interfere = y_length
pos_y_DOWN_interfere = 0

print(pos_y_DOWN_interfere)
print(pos_y_UP_interfere)
print(pos_x_left_interfere)
print(pos_x_right_interfere)




mirror = False
if mirror == True:
    pos_x = pos_x * -1

outArray = np.empty((0,data.shape[1]+1))

i=startFrame
fig, ax = plt.subplots()
plt.xlim(-6,6)
plt.ylim(0,10)
# set asymmetry of x and y axes:
ax.set_aspect('equal')

ax.add_artist(Circle((pos_x,pos_y),radius,fill=False))
ax.add_artist(Rectangle((pos_x_LD,pos_y_LD),x_length,y_length))

while(i <= lastFrame):
    if(i%stepFrame == 0):
        focusedFrame = i
        indices = np.argwhere(data_all[:, 0] == focusedFrame)
        indices = np.squeeze(indices)
        procData = data[indices[0]:indices[-1], :]
        np.savetxt('test_out_0.csv', procData, delimiter=',')
        manyObj = 0
        j = 0
        objColumn = np.zeros((indices[-1] - indices[0], 1))
        startTime_total = time.process_time()
        while (j < (indices[-1] - indices[0])):
            # Define column
            # objColumn = np.empty((indices[-1] - indices[0], 1))
            data_x = procData[j, 2]
            data_y = procData[j, 3]
            # comparision in x and y:
            # if data_x < pos_x_max and data_x > pos_x_min and data_y < pos_y_max and data_y > pos_y_min:
            if ((data_x - pos_x) ** 2 + (data_y - pos_y) ** 2 < radius ** 2):
                objColumn[j, 0] = 1
                manyObj = manyObj + 1
                ax.scatter(data_x, data_y, c='#17becf', marker='x')
            elif (data_x < pos_x_LD+x_length and data_x > pos_x_LD and data_y < pos_y_UP_interfere):
                objColumn[j, 0] = 3
                ax.scatter(data_x, data_y, c='#ffbecf', marker='^')
            else:
                objColumn[j, 0] = 0
                ax.scatter(data_x, data_y, c='#1f77b4')
            j = j + 1
        result = np.column_stack((data[indices[0]:indices[-1], :], objColumn))
        # outArray = np.row_stack(outArray,result)
        outArray = np.row_stack((outArray, result))
        endTime_total = time.process_time()
        print('Frame: %d/%d ... time: %.10f' % (i, lastFrame, (endTime_total - startTime_total)))
    i = i + 1
header = 'frame,DetObj#,x,y,z,v,snr,noise,realObj'
'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer2.csv'
outFileName = path.split('/')[-1].split('.')[-2] + "_LABELIZED.csv"
np.savetxt(outFileName, outArray,header=header ,delimiter=',')



#test labelization
'''
dat_lab = outArray
lastFrame = lastFrame

n = startFrame;

while (n<=lastFrame):
    if(n%stepFrame == 0):
        focusedFrame = n
        indices = np.argwhere(data_all[:, 0] == focusedFrame)
        indices = np.squeeze(indices)
        obj = 0
        obj_counter = 0
        fig1, ax1 = plt.subplots()
        plt.xlim((-5, 5))
        plt.ylim((0, 9))
        ax1.set_aspect('equal')
        ax1.add_artist(Circle((pos_x, pos_y), radius, fill=False))
        #rectPos_x = (pos_x_center_interfere - x_length)
        #rectPos_y = (pos_y_center_interfere - y_length)
        ax1.add_artist(Rectangle((pos_x_LD, pos_y_LD), x_length, y_length, fill=False))
        while (obj < indices[-1] - indices[0]):
            if (dat_lab[obj, 8] == 1):
                # draw
                obj_counter = obj_counter + 1
                # ax1.scatter(dat_lab[obj,2],dat_lab[obj,3],c='black')
                ax1.scatter(dat_lab[obj, 2], dat_lab[obj, 3], c='red')
            elif (dat_lab[obj, 8] == 3):
                ax1.scatter(dat_lab[obj, 2], dat_lab[obj, 3], c='blue')
            else:
                ax1.scatter(dat_lab[obj, 2], dat_lab[obj, 3], c='black')
            obj = obj + 1
        fig1.suptitle("frame_%d/%d, detObj_ %d" % (n, lastFrame, obj_counter))

        # save the plot as an image
        fig1.savefig(f"FB_figures/frame_{n}.png")
        ax1.remove()
    n = n + 1
'''
