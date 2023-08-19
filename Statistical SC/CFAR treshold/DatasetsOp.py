import numpy as np
import os
class Datasets:
    path = None
    measurement = None
    pos_x = None
    pos_y = None
    data = None
    def __init__(self):
        self.scriptPath = os.path.abspath(__file__)
        self.datasetsPath_Labelized = self.FindDatasetsPath_Labelized()
        self.datasetsPath_NonLabelized = self.FindDatasetsPath_NonLabelized()
        self.path_labelized_m1 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
        self.m1_pos_x = 0
        self.m1_pos_y = 8
        self.path_labelized_m2 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[1])
        self.m2_pos_x = 2.5
        self.m2_pos_y = 7.8
        self.path_labelized_m3 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[2])
        self.m3_pos_x = -4.5
        self.m3_pos_y = 4.5
        self.path_labelized_m4 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[3])
        self.m4_pos_x = 4
        self.m4_pos_y = 5.5
        self.path_labelized_m5 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[4])
        self.m5_pos_x = 0
        self.m5_pos_y = 1
        self.path_labelized_m6 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[5])
        self.m6_pos_x = 0
        self.m6_pos_y = 2.5
        self.path_labelized_m7 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[6])
        self.m7_pos_x = 3
        self.m7_pos_y = 4
        self.path_labelized_m8 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[7])
        self.m8_pos_x = -2
        self.m8_pos_y = 3
        self.path_labelized_m9 = os.path.join(os.getcwd(), os.listdir(os.getcwd())[8])
        self.m9_pos_x = 0
        self.m9_pos_y = 8.8

    def FindDatasetsPath_NonLabelized(self):
        os.chdir("..")
        contens = os.listdir(os.getcwd())
        idx = 0
        while idx < len(contens):
            if contens[idx].find("static_measurement_parsed"):
                idx = len(contens)
                os.chdir(os.path.join(os.getcwd(), "static_measurement_parsed"))
            idx += 1
        return  os.getcwd()

    def FindDatasetsPath_Labelized(self):
        os.chdir("..")  # Go above in directory
        os.chdir("..")  # Go above in directory
        contents = os.listdir(os.getcwd())  # Get Array of content of array
        idx = 0
        while idx < len(contents):
            found = None
            if contents[idx].find("Datasets") != -1:
                os.chdir(os.path.join(os.getcwd(), "Datasets"))
                contents = os.listdir(os.getcwd())
                idx = len(contents)
                jdx = 0
                while jdx < len(contents):
                    if contents[jdx].find("static_measurement_labelized") != -1:
                        os.chdir(os.path.join(os.getcwd(), "static_measurement_labelized"))
                        contents = os.listdir(os.getcwd())
                        jdx = len(contents)
                        if contents == 0:
                            found = False
                        else:
                            found = True
                    jdx += 1
                #print("Changed " + os.getcwd())
            # else:
            # print("Nothing at " ,{idx} )
            idx += 1
            if idx == len(contents):
                found = False
        return os.getcwd()
    def getArrayOfPathMeasurement_Labelized(self):
        return os.listdir(self.datasetsPath_Labelized)  # Get Array of content of array
    def getArrayOfPathMeasurement_NonLabelized(self):
        return os.listdir(self.datasetsPath_NonLabelized)  # Get Array of content of array

    def ActualMeasurement(self):
        return Datasets.measurement
    def ActualPathOfMeasurement(self):
        return Datasets.path
    def ActualPos(self):
        return [Datasets.pos_x,Datasets.pos_y]
    def LoadDataset(self,pathOfMeasurement):
        data = np.genfromtxt(pathOfMeasurement, delimiter=',', skip_header=1)
        return np.genfromtxt(pathOfMeasurement, delimiter=',', skip_header=1)
