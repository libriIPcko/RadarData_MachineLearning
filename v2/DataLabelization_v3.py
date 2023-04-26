import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle
import time

path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer2.csv'
data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)
fileName = path[path.rfind('/') + 1:]

#for DetObj radius
radius = 1
pos_x = 3     #2.5
pos_y = 7.5   #8

#for interfere radius


#lastFrame = data_all[-1,0]
lastFrame = 3

mirror = False
if mirror == True:
    pos_x = pos_x * -1

#outArray = np.empty((data.shape[0],data.shape[1]+1))
outArray = np.empty((0,data.shape[1]+1))

i=0
fig, ax = plt.subplots()
plt.xlim(-6,6)
plt.ylim(0,10)
# set asymmetry of x and y axes:
ax.set_aspect('equal')

ax.add_artist(Circle((pos_x,pos_y),radius,fill=False))
ax.add_artist(Circle((pos_x,pos_y),.2))

while(i < lastFrame):
    focusedFrame = i
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    procData = data[indices[0]:indices[-1],:]
    np.savetxt('test_out_0.csv', procData, delimiter=',')
    manyObj = 0
    j = 0
    objColumn = np.zeros((indices[-1] - indices[0], 1))
    startTime_total = time.process_time()
    while(j<(indices[-1]-indices[0])):
        # Define column
        #objColumn = np.empty((indices[-1] - indices[0], 1))
        data_x = procData[j, 2]
        data_y = procData[j, 3]
        # comparision in x and y:
        #if data_x < pos_x_max and data_x > pos_x_min and data_y < pos_y_max and data_y > pos_y_min:
        if ((data_x - pos_x)**2 + (data_y - pos_y)**2 < radius**2):
            objColumn[j, 0] = 1
            manyObj = manyObj + 1
            ax.scatter(data_x, data_y, c='#17becf', marker='x')
        else:
            objColumn[j,0] = 0
            ax.scatter(data_x,data_y,c='#1f77b4')
        j = j + 1
    result = np.column_stack((data[indices[0]:indices[-1], :], objColumn))
    #outArray = np.row_stack(outArray,result)
    outArray = np.row_stack((outArray, result))
    endTime_total = time.process_time()
    print('Frame: %d/%d ... time: %.10f' %(i,lastFrame,(endTime_total-startTime_total)))
    i = i + 1
header = 'frame,DetObj#,x,y,z,v,snr,noise,realObj'
np.savetxt('test_out_1.csv', outArray,header=header ,delimiter=',')



#test labelization

#ax1.add_artist(Circle((pos_x,pos_y),.2))

dat_lab = outArray
lastFrame = lastFrame

n = 0;

while (n<lastFrame):
    focusedFrame = n
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    obj = 0
    obj_counter = 0

    fig1, ax1 = plt.subplots()
    plt.xlim(0, 5)
    plt.ylim((6, 9))
    ax1.set_aspect('equal')
    ax1.add_artist(Circle((pos_x, pos_y), radius, fill=False))

    while (obj<indices[-1]-indices[0]):
        if(dat_lab[obj,8] == 1):
            #draw
            obj_counter = obj_counter + 1
            #ax1.scatter(dat_lab[obj,2],dat_lab[obj,3],c='black')
            ax1.scatter(dat_lab[obj, 2], dat_lab[obj, 3])
        else:
            ax1.scatter(dat_lab[obj, 2], dat_lab[obj, 3], c='black')
        obj = obj + 1
    fig1.suptitle("frame_%d/%d, detObj_ %d" %(n,lastFrame,obj_counter))
    # save the plot as an image
    #fig1 = plt.gcf()
    fig1.savefig(f"FB_figures/frame_{n}.png")
    ax1.cla()
    #ax.clear()
    #ax.cla()
    n = n + 1
