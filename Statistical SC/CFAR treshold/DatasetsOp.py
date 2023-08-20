import numpy as np
import os
class Datasets:
    scriptPath = None
    measurement = None
    pos_x = None
    pos_y = None
    data = None

    frame = 0
    detObj = 1
    x = 2
    y = 3
    z = 4
    v = 5
    snr = 6
    noise = 7
    label = 8

    def __init__(self):
        self.scriptPath = os.getcwd()
        self.datasetsPath_Labelized = self.FindDatasetsPath_Labelized()
        self.datasetsPath_NonLabelized = self.FindDatasetsPath_NonLabelized()
        self.listLabMeas = self.getArrayOfPathMeasurement_Labelized()
        self.listNonLabMeas = self.getArrayOfPathMeasurement_NonLabelized()
    def find_directory(self,start_dir, target_dir):
        for root, dirs, files in os.walk(start_dir):
            if target_dir in dirs:
                return os.path.join(root, target_dir)
        return None
    def find_directory_in_system(self,target_dir):
        # possible_roots = [os.path.abspath("/")]  # You can add more roots if needed
        mainDir_root = ""
        abc = ""
        os.chdir(self.scriptPath)
        n = 0               #how many times can go above project directory
        while n < 5:
            os.chdir("..")
            content = os.listdir(os.getcwd())
            m = 0
            while m < len(content):
                if content[m].find("RadarData_MachineLearning") != -1:
                    mainDir_root = os.getcwd()
                    n = 5
                m += 1
            n += 1
        #print(os.getcwd())
        possible_roots = [mainDir_root]  # You can add more roots if needed
        for root in possible_roots:
            found_path = self.find_directory(root, target_dir)  # Provide both start_dir and target_dir
            if found_path:
                #print("First shot")
                return found_path
            else:
                return None
    def FindDatasetsPath_NonLabelized(self):
            #OLD VERSION OF SEEKING DESIRED DIRECTORY
        # os.chdir("..")
        # contens = os.listdir(os.getcwd())
        # idx = 0
        # found = False
        # while idx < len(contens):
        #     if contens[idx].find("static_measurement_parsed"):
        #         idx = len(contens)
        #         os.chdir(os.path.join(os.getcwd(), "static_measurement_parsed"))
        #         pathNonLabelizedMeasurement = os.getcwd()
        #         found = True
        #         idx = len(contens)
        #     idx += 1
        # found = False
        # return  os.getcwd()
        return self.find_directory_in_system("static_measurement_parsed")
    def FindDatasetsPath_Labelized(self):
            #OLD VERSION OF SEEKING DESIRED DIRECTORY
        # self.scriptPath = os.getcwd()
        # os.chdir("..")  # Go above in directory
        # os.chdir("..")  # Go above in directory
        # contents = os.listdir(os.getcwd())  # Get Array of content of array
        # idx = 0
        # while idx < len(contents):
        #     found = None
        #     if contents[idx].find("Datasets") != -1:
        #         os.chdir(os.path.join(os.getcwd(), "Datasets"))
        #         contents = os.listdir(os.getcwd())
        #         idx = len(contents)
        #         jdx = 0
        #         while jdx < len(contents):
        #             if contents[jdx].find("static_measurement_labelized") != -1:
        #                 os.chdir(os.path.join(os.getcwd(), "static_measurement_labelized"))
        #                 contents = os.listdir(os.getcwd())
        #                 jdx = len(contents)
        #                 if contents == 0:
        #                     found = False
        #                 else:
        #                     found = True
        #             jdx += 1
        #         #print("Changed " + os.getcwd())
        #     # else:
        #     # print("Nothing at " ,{idx} )
        #     idx += 1
        #     if idx == len(contents):
        #         found = False
        # return os.getcwd()
        return self.find_directory_in_system("static_measurement_labelized")
    def getArrayOfPathMeasurement_Labelized(self):
        content = os.listdir(self.datasetsPath_Labelized)  # Get Array of content of array
        i = 0
        while i < len(content):
            content[i] = os.path.join(self.datasetsPath_Labelized ,content[i])
            i += 1

        return content
    def getArrayOfPathMeasurement_NonLabelized(self):
        content = os.listdir(self.datasetsPath_NonLabelized)  # Get Array of content of array
        i = 0
        while i < len(content):
            content[i] = os.path.join(self.datasetsPath_NonLabelized ,content[i])
            i += 1
        return content
    def LoadDataset(self,pathOfMeasurement):
        data = np.genfromtxt(pathOfMeasurement, delimiter=',', skip_header=1)
        return np.genfromtxt(pathOfMeasurement, delimiter=',', skip_header=1)
    def LoadDataset_specFrame(self,pathOfMeasurement,Frame):
        idx = 0
        data = self.LoadDataset(pathOfMeasurement)
        dataFrame = []
        checkedFrame = False
        while idx < data.shape[0]:
            if data[idx, 0] == Frame:
                dataFrame.append(data[idx, :])
                checkedFrame = True
            elif checkedFrame == True:
                idx = data.shape[0]
            idx += 1
        return np.array(dataFrame)