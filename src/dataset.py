from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader
import pandas as pd
import numpy as np
import torch
import os

import warnings
warnings.filterwarnings("ignore")

# class NYCTrafficCountDataset(Dataset):
#     def __init__(self, trafficPath, xPath, adjPath, scenePath):
#         super(NYCTrafficCountDataset, self).__init__(None)
#         self.trafficPath = trafficPath
#         self.xPath = xPath
#         self.adj = adjPath
#         self.scene = scenePath
#         self.dateLst = os.listdir(self.trafficPath)
#         self.columnLst = ['SVIID',
#                             'StreetWidt', 'Length',  
#                             'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
#                             'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
#                             'SocialServ', 'Transporta', 'Water',
#                             'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
#                             'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
#                             'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
#                             'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
#                             'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
#                             'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
#                             'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
#                             'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']
#         self.adj = pd.read_csv(self.adj)

#         self.F = self.adj["From"].values
#         self.T = self.adj["To"].values
#         self.svi = pd.read_csv(self.scene).drop(columns=['Unnamed: 0']).values

#     def len(self):
#         return 24*len(self.dateLst)
#     @property
#     def raw_file_names(self):
#         return []
    
#     def get(self, idx):
#         dateIdx = idx//24
#         timeIdx = idx%24

#         date = self.dateLst[dateIdx]
#         year = date.split("_")[0]
#         xPath = os.path.join(self.xPath, "Divided_" + year + ".csv")
#         xDf = pd.read_csv(xPath)
#         xDf = xDf[self.columnLst]

#         self.x = xDf.values
#         self.x = np.append(self.x, self.svi, 1)

#         timePath = os.path.join(self.trafficPath, date)
#         trafficDf = pd.read_csv(timePath)
#         # print(trafficDf.columns)
#         # print(trafficDf.columns[-25])
#         trafficInfo = trafficDf.values
#         mask = trafficInfo[:, -1]
#         mask = mask.reshape((mask.shape[0], 1))

#         y = trafficInfo[:, -25 + timeIdx].reshape((mask.shape[0], 1))
#         xTraffic = trafficInfo[:, timeIdx: 72+timeIdx]
#         self.x = np.append(self.x, xTraffic, 1)
#         y = np.append(y, mask, 1)

#         gY = torch.tensor(y, dtype=torch.float)
#         gX = torch.tensor(self.x, dtype=torch.float)
#         edge = torch.tensor([self.F, self.T], dtype=torch.long)
#         data = Data(x=gX, edge_index=edge, y=gY)

#         return data
    
# class NYCTrafficCountDataset_sub(Dataset):
#     def __init__(self, trafficLst, xLst, adjLst, dateLst, scenePath):
#     # def __init__(self, trafficPath, xPath, adjFolder, scenePath):
#         super(NYCTrafficCountDataset_sub, self).__init__(None)
#         # self.trafficPath = trafficPath
#         # self.xPath = xPath
#         # self.adj = adjFolder
#         self.scene = scenePath
#         # self.dateLst = os.listdir(self.trafficPath)
#         self.dateLst = dateLst
#         self.columnLst = ['SVIID',
#                             'StreetWidt', 'Length',  
#                             'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
#                             'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
#                             'SocialServ', 'Transporta', 'Water',
#                             'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
#                             'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
#                             'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
#                             'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
#                             'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
#                             'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
#                             'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
#                             'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']

#         self.sviDf = pd.read_csv(self.scene).drop(columns=['Unnamed: 0'])

#         self.trafficLst = trafficLst
#         self.adjLst = adjLst
#         self.xLst = xLst


#     def len(self):
#         return 24*len(self.dateLst)
#     @property
#     def raw_file_names(self):
#         return []
    
#     def get(self, idx):
#         dateIdx = idx//24
#         timeIdx = idx%24

#         date = self.dateLst[dateIdx]

#         # Get traffic information
#         # trafficPth = os.path.join(self.trafficPath, date)
#         # trafficDf = pd.read_csv(trafficPth)
#         trafficDf = self.trafficLst[dateIdx]
#         originIDs = trafficDf["originID"].values.tolist()

#         # Get adj
#         # adjPth = os.path.join(self.adj, date)
#         # adjDf = pd.read_csv(adjPth)
#         adjDf = self.adjLst[dateIdx]
#         F = adjDf["From"].values
#         T = adjDf["To"].values

#         # Get svi scene
#         svi = self.sviDf.iloc[originIDs]

#         # Get se features
#         year = int(date.split("_")[0])-2016
#         xDf = self.xLst[year]
#         # xPath = os.path.join(self.xPath, "Divided_" + year + ".csv")
#         # xDf = pd.read_csv(xPath)
#         # xDf = xDf[self.columnLst]
#         xDf = xDf.iloc[originIDs]

#         x = xDf.values
#         x = np.append(x, svi, 1)


#         # timePath = os.path.join(self.trafficPath, date)
#         # trafficDf = pd.read_csv(timePath)
#         # print(trafficDf.columns)
#         # print(trafficDf.columns[-25])

#         # Get Y
#         trafficInfo = trafficDf.values
#         mask = trafficInfo[:, -2]
#         mask = mask.reshape((mask.shape[0], 1))

#         y = trafficInfo[:, -26 + timeIdx].reshape((mask.shape[0], 1))
#         xTraffic = trafficInfo[:, timeIdx: 72+timeIdx]

#         maxv = xTraffic.max(axis=1).reshape([xTraffic.shape[0], 1]) + 1
#         minv = xTraffic.min(axis=1).reshape([xTraffic.shape[0], 1])

#         # y = (y - minv) / (maxv - minv)
#         xTraffic = (xTraffic - minv) / (maxv - minv)

#         x = np.append(x, xTraffic, 1)
#         y = np.append(y, mask, 1)
#         y = np.append(y, minv, 1)
#         y = np.append(y, maxv, 1)
        
#         gY = torch.tensor(y, dtype=torch.float)
#         gX = torch.tensor(x, dtype=torch.float)
#         edge = torch.tensor([F, T], dtype=torch.long)
#         data = Data(x=gX, edge_index=edge, y=gY)

#         return data

class NYCTrafficCountDataset_long(Dataset):
    def __init__(self, trafficLst, xLst, adjLst, dateLst, scenePath, input_length):
    # def __init__(self, trafficPath, xPath, adjFolder, scenePath):
        super(NYCTrafficCountDataset_long, self).__init__(None)
        # self.trafficPath = trafficPath
        # self.xPath = xPath
        # self.adj = adjFolder
        self.scene = scenePath
        # self.dateLst = os.listdir(self.trafficPath)
        self.dateLst = dateLst
        self.columnLst = ['SVIID',
                            'StreetWidt', 'Length',  
                            'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
                            'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
                            'SocialServ', 'Transporta', 'Water',
                            'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
                            'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
                            'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
                            'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
                            'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
                            'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
                            'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
                            'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']

        self.sviDf = pd.read_csv(self.scene).drop(columns=['Unnamed: 0'])

        self.trafficLst = trafficLst
        self.adjLst = adjLst
        self.xLst = xLst
        self.inputLength = input_length


    def len(self):
        return 24*len(self.dateLst)
    @property
    def raw_file_names(self):
        return []
    
    def get(self, idx):
        dateIdx = idx//24
        timeIdx = idx%24

        date = self.dateLst[dateIdx]

        trafficDf = self.trafficLst[dateIdx]
        originIDs = trafficDf["originID"].values.tolist()

        adjDf = self.adjLst[dateIdx]
        F = adjDf["From"].values
        T = adjDf["To"].values

        # Get svi scene
        svi = self.sviDf.iloc[originIDs]

        # Get se features
        year = int(date.split("_")[0])-2016
        xDf = self.xLst[year]

        xDf = xDf.iloc[originIDs]

        x = xDf.values
        x = np.append(x, svi, 1)

        trafficInfo = trafficDf.values
        mask = trafficInfo[:, -2]
        mask = mask.reshape((mask.shape[0], 1))

        y = trafficInfo[:, timeIdx+68:timeIdx+72]


        xTraffic = trafficInfo[:, timeIdx+68-self.inputLength: 68+timeIdx]

        maxv = xTraffic.max(axis=1).reshape([xTraffic.shape[0], 1]) + 1
        minv = xTraffic.min(axis=1).reshape([xTraffic.shape[0], 1])

        # y = (y - minv) / (maxv - minv)
        xTraffic = (xTraffic - minv) / (maxv - minv)

        x = np.append(x, xTraffic, 1)
        y = np.append(y, mask, 1)
        y = np.append(y, minv, 1)
        y = np.append(y, maxv, 1)
        
        gY = torch.tensor(y, dtype=torch.float)
        gX = torch.tensor(x, dtype=torch.float)
        edge = torch.tensor([F, T], dtype=torch.long)
        data = Data(x=gX, edge_index=edge, y=gY)

        return data

class NYCTrafficCountDataset_short(Dataset):
    def __init__(self, trafficLst, xLst, adjLst, dateLst, scenePath, input_length):
    # def __init__(self, trafficPath, xPath, adjFolder, scenePath):
    # trafficLst: list of panda dataframes of all traffic information under data folder
    # xLst: list of panda dataframes of all x values (except scene features), including POI, SE, Phy
    # adjLst: 
    # dateLst: filenames in data folder
    # scenePath: path to scene features.
    # input length: input window (6, 12, 24)
        super(NYCTrafficCountDataset_short, self).__init__(None)
        # self.trafficPath = trafficPath
        # self.xPath = xPath
        # self.adj = adjFolder
        self.scene = scenePath
        # self.dateLst = os.listdir(self.trafficPath)
        self.dateLst = dateLst
        self.columnLst = ['SVIID',
                            'StreetWidt', 'Length',  
                            'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
                            'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
                            'SocialServ', 'Transporta', 'Water',
                            'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
                            'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
                            'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
                            'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
                            'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
                            'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
                            'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
                            'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']

        self.sviDf = pd.read_csv(self.scene).drop(columns=['Unnamed: 0'])

        self.trafficLst = trafficLst
        self.adjLst = adjLst
        self.xLst = xLst
        self.inputLength = input_length


    def len(self):
        return 24*len(self.dateLst)
    @property
    def raw_file_names(self):
        return []
    
    def get(self, idx):
        dateIdx = idx//24
        timeIdx = idx%24

        date = self.dateLst[dateIdx]

        # Get traffic information
        # trafficPth = os.path.join(self.trafficPath, date)
        # trafficDf = pd.read_csv(trafficPth)
        trafficDf = self.trafficLst[dateIdx]
        originIDs = trafficDf["originID"].values.tolist()

        # Get adj
        # adjPth = os.path.join(self.adj, date)
        # adjDf = pd.read_csv(adjPth)
        adjDf = self.adjLst[dateIdx]
        F = adjDf["From"].values
        T = adjDf["To"].values

        # Get svi scene
        svi = self.sviDf.iloc[originIDs]

        # Get se features
        year = int(date.split("_")[0])-2016
        xDf = self.xLst[year]
        # xPath = os.path.join(self.xPath, "Divided_" + year + ".csv")
        # xDf = pd.read_csv(xPath)
        # xDf = xDf[self.columnLst]
        xDf = xDf.iloc[originIDs]

        x = xDf.values
        x = np.append(x, svi, 1)


        # timePath = os.path.join(self.trafficPath, date)
        # trafficDf = pd.read_csv(timePath)
        # print(trafficDf.columns)
        # print(trafficDf.columns[-25])

        # Get Y
        trafficInfo = trafficDf.values
        mask = trafficInfo[:, -2]
        mask = mask.reshape((mask.shape[0], 1))

        y = trafficInfo[:, -26 + timeIdx].reshape((mask.shape[0], 1))
        xTraffic = trafficInfo[:, timeIdx+72-self.inputLength: timeIdx+72]

        maxv = xTraffic.max(axis=1).reshape([xTraffic.shape[0], 1]) + 1
        minv = xTraffic.min(axis=1).reshape([xTraffic.shape[0], 1])

        # y = (y - minv) / (maxv - minv)
        xTraffic = (xTraffic - minv) / (maxv - minv)

        x = np.append(x, xTraffic, 1)
        y = np.append(y, mask, 1)
        y = np.append(y, minv, 1)
        y = np.append(y, maxv, 1)
        
        gY = torch.tensor(y, dtype=torch.float)
        gX = torch.tensor(x, dtype=torch.float)
        edge = torch.tensor([F, T], dtype=torch.long)
        data = Data(x=gX, edge_index=edge, y=gY)

        return data
    
def getDatasetLst(trafficPth, adjFolder):
    trafficLst = []
    adjLst = []
    dateLst = os.listdir(trafficPth)
    for date in dateLst:
        datePth = os.path.join(trafficPth, date)
        adjPth = os.path.join(adjFolder, date)
        trafficLst.append(pd.read_csv(datePth))
        adjLst.append(pd.read_csv(adjPth))
    return dateLst, trafficLst, adjLst
# class NYCTrafficCountDataset_short_12(Dataset):
#     def __init__(self, trafficLst, xLst, adjLst, dateLst, scenePath):
#     # def __init__(self, trafficPath, xPath, adjFolder, scenePath):
#         super(NYCTrafficCountDataset_short_12, self).__init__(None)
#         # self.trafficPath = trafficPath
#         # self.xPath = xPath
#         # self.adj = adjFolder
#         self.scene = scenePath
#         # self.dateLst = os.listdir(self.trafficPath)
#         self.dateLst = dateLst
#         self.columnLst = ['SVIID',
#                             'StreetWidt', 'Length',  
#                             'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
#                             'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
#                             'SocialServ', 'Transporta', 'Water',
#                             'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
#                             'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
#                             'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
#                             'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
#                             'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
#                             'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
#                             'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
#                             'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']

#         self.sviDf = pd.read_csv(self.scene).drop(columns=['Unnamed: 0'])

#         self.trafficLst = trafficLst
#         self.adjLst = adjLst
#         self.xLst = xLst


#     def len(self):
#         return 24*len(self.dateLst)
#     @property
#     def raw_file_names(self):
#         return []
    
#     def get(self, idx):
#         dateIdx = idx//24
#         timeIdx = idx%24

#         date = self.dateLst[dateIdx]

#         # Get traffic information
#         # trafficPth = os.path.join(self.trafficPath, date)
#         # trafficDf = pd.read_csv(trafficPth)
#         trafficDf = self.trafficLst[dateIdx]
#         originIDs = trafficDf["originID"].values.tolist()

#         # Get adj
#         # adjPth = os.path.join(self.adj, date)
#         # adjDf = pd.read_csv(adjPth)
#         adjDf = self.adjLst[dateIdx]
#         F = adjDf["From"].values
#         T = adjDf["To"].values

#         # Get svi scene
#         svi = self.sviDf.iloc[originIDs]

#         # Get se features
#         year = int(date.split("_")[0])-2016
#         xDf = self.xLst[year]
#         # xPath = os.path.join(self.xPath, "Divided_" + year + ".csv")
#         # xDf = pd.read_csv(xPath)
#         # xDf = xDf[self.columnLst]
#         xDf = xDf.iloc[originIDs]

#         x = xDf.values
#         x = np.append(x, svi, 1)


#         # timePath = os.path.join(self.trafficPath, date)
#         # trafficDf = pd.read_csv(timePath)
#         # print(trafficDf.columns)
#         # print(trafficDf.columns[-25])

#         # Get Y
#         trafficInfo = trafficDf.values
#         mask = trafficInfo[:, -2]
#         mask = mask.reshape((mask.shape[0], 1))

#         y = trafficInfo[:, -26 + timeIdx].reshape((mask.shape[0], 1))
#         xTraffic = trafficInfo[:, timeIdx+60: timeIdx+72]

#         maxv = xTraffic.max(axis=1).reshape([xTraffic.shape[0], 1]) + 1
#         minv = xTraffic.min(axis=1).reshape([xTraffic.shape[0], 1])

#         # y = (y - minv) / (maxv - minv)
#         xTraffic = (xTraffic - minv) / (maxv - minv)

#         x = np.append(x, xTraffic, 1)
#         y = np.append(y, mask, 1)
#         y = np.append(y, minv, 1)
#         y = np.append(y, maxv, 1)
        
#         gY = torch.tensor(y, dtype=torch.float)
#         gX = torch.tensor(x, dtype=torch.float)
#         edge = torch.tensor([F, T], dtype=torch.long)
#         data = Data(x=gX, edge_index=edge, y=gY)

#         return data
    
# class NYCTrafficCountDataset_short_6(Dataset):
#     def __init__(self, trafficLst, xLst, adjLst, dateLst, scenePath):
#     # def __init__(self, trafficPath, xPath, adjFolder, scenePath):
#         super(NYCTrafficCountDataset_short_6, self).__init__(None)
#         # self.trafficPath = trafficPath
#         # self.xPath = xPath
#         # self.adj = adjFolder
#         self.scene = scenePath
#         # self.dateLst = os.listdir(self.trafficPath)
#         self.dateLst = dateLst
#         self.columnLst = ['SVIID',
#                             'StreetWidt', 'Length',  
#                             'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
#                             'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
#                             'SocialServ', 'Transporta', 'Water',
#                             'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
#                             'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
#                             'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
#                             'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
#                             'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
#                             'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
#                             'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
#                             'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']

#         self.sviDf = pd.read_csv(self.scene).drop(columns=['Unnamed: 0'])

#         self.trafficLst = trafficLst
#         self.adjLst = adjLst
#         self.xLst = xLst


#     def len(self):
#         return 24*len(self.dateLst)
#     @property
#     def raw_file_names(self):
#         return []
    
#     def get(self, idx):
#         dateIdx = idx//24
#         timeIdx = idx%24

#         date = self.dateLst[dateIdx]

#         # Get traffic information
#         # trafficPth = os.path.join(self.trafficPath, date)
#         # trafficDf = pd.read_csv(trafficPth)
#         trafficDf = self.trafficLst[dateIdx]
#         originIDs = trafficDf["originID"].values.tolist()

#         # Get adj
#         # adjPth = os.path.join(self.adj, date)
#         # adjDf = pd.read_csv(adjPth)
#         adjDf = self.adjLst[dateIdx]
#         F = adjDf["From"].values
#         T = adjDf["To"].values

#         # Get svi scene
#         svi = self.sviDf.iloc[originIDs]

#         # Get se features
#         year = int(date.split("_")[0])-2016
#         xDf = self.xLst[year]
#         # xPath = os.path.join(self.xPath, "Divided_" + year + ".csv")
#         # xDf = pd.read_csv(xPath)
#         # xDf = xDf[self.columnLst]
#         xDf = xDf.iloc[originIDs]

#         x = xDf.values
#         x = np.append(x, svi, 1)


#         # timePath = os.path.join(self.trafficPath, date)
#         # trafficDf = pd.read_csv(timePath)
#         # print(trafficDf.columns)
#         # print(trafficDf.columns[-25])

#         # Get Y
#         trafficInfo = trafficDf.values
#         mask = trafficInfo[:, -2]
#         mask = mask.reshape((mask.shape[0], 1))

#         y = trafficInfo[:, -26 + timeIdx].reshape((mask.shape[0], 1))
#         xTraffic = trafficInfo[:, timeIdx+66: timeIdx+72]

#         maxv = xTraffic.max(axis=1).reshape([xTraffic.shape[0], 1]) + 1
#         minv = xTraffic.min(axis=1).reshape([xTraffic.shape[0], 1])

#         # y = (y - minv) / (maxv - minv)
#         xTraffic = (xTraffic - minv) / (maxv - minv)

#         x = np.append(x, xTraffic, 1)
#         y = np.append(y, mask, 1)
#         y = np.append(y, minv, 1)
#         y = np.append(y, maxv, 1)
        
#         gY = torch.tensor(y, dtype=torch.float)
#         gX = torch.tensor(x, dtype=torch.float)
#         edge = torch.tensor([F, T], dtype=torch.long)
#         data = Data(x=gX, edge_index=edge, y=gY)

#         return data
    
# class NYCTrafficCountDataset_test_short(Dataset):
#     def __init__(self, trafficLst, xLst, adjLst, dateLst, scenePath, input_length):
#         super(NYCTrafficCountDataset_test_short, self).__init__(None)

#         self.scene = scenePath
#         self.dateLst = dateLst
#         self.columnLst = ['SVIID',
#                             'StreetWidt', 'Length',  
#                             'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
#                             'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
#                             'SocialServ', 'Transporta', 'Water',
#                             'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
#                             'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
#                             'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
#                             'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
#                             'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
#                             'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
#                             'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
#                             'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']

#         self.sviDf = pd.read_csv(self.scene).drop(columns=['Unnamed: 0'])

#         self.trafficLst = trafficLst
#         self.adjLst = adjLst
#         self.xLst = xLst
#         self.inputLength = input_length

#     def len(self):
#         return 72*len(self.dateLst)
#     @property
#     def raw_file_names(self):
#         return []
    
#     def get(self, idx):
#         dateIdx = idx//72
#         timeIdx = idx%72

#         date = self.dateLst[dateIdx]

#         # Get traffic information
#         # trafficPth = os.path.join(self.trafficPath, date)
#         # trafficDf = pd.read_csv(trafficPth)
#         trafficDf = self.trafficLst[dateIdx]
#         originIDs = trafficDf["originID"].values.tolist()

#         # Get adj
#         # adjPth = os.path.join(self.adj, date)
#         # adjDf = pd.read_csv(adjPth)
#         adjDf = self.adjLst[dateIdx]
#         F = adjDf["From"].values
#         T = adjDf["To"].values

#         # Get svi scene
#         svi = self.sviDf.iloc[originIDs]

#         # Get se features
#         year = int(date.split("_")[0])-2016
#         xDf = self.xLst[year]

#         xDf = xDf.iloc[originIDs]

#         x = xDf.values        # if i%5 == 0:
#         #     model_path = os.path.join(normal_path, str(i) + ".pt")
#         #     torch.save(model.state_dict(), model_path)
#         x = np.append(x, svi, 1)

#         # Get Y
#         trafficInfo = trafficDf.values
#         mask = trafficInfo[:, -2]
#         mask = mask.reshape((mask.shape[0], 1))

#         y = trafficInfo[:, 24 + timeIdx].reshape((mask.shape[0], 1))
#         xTraffic = trafficInfo[:, timeIdx+24-self.inputLength: timeIdx+24]

#         maxv = xTraffic.max(axis=1).reshape([xTraffic.shape[0], 1]) + 1
#         minv = xTraffic.min(axis=1).reshape([xTraffic.shape[0], 1])

#         # y = (y - minv) / (maxv - minv)
#         xTraffic = (xTraffic - minv) / (maxv - minv)

#         x = np.append(x, xTraffic, 1)
#         y = np.append(y, mask, 1)
#         y = np.append(y, minv, 1)
#         y = np.append(y, maxv, 1)
        
#         gY = torch.tensor(y, dtype=torch.float)
#         gX = torch.tensor(x, dtype=torch.float)
#         edge = torch.tensor([F, T], dtype=torch.long)
#         data = Data(x=gX, edge_index=edge, y=gY)

#         return data

# class NYCTrafficCountDataset_test_long(Dataset):
#     def __init__(self, trafficLst, xLst, adjLst, dateLst, scenePath, input_length):
#     # def __init__(self, trafficPath, xPath, adjFolder, scenePath):
#         super(NYCTrafficCountDataset_test_long, self).__init__(None)
#         # self.trafficPath = trafficPath
#         # self.xPath = xPath
#         # self.adj = adjFolder
#         self.scene = scenePath
#         # self.dateLst = os.listdir(self.trafficPath)
#         self.dateLst = dateLst
#         self.columnLst = ['SVIID',
#                             'StreetWidt', 'Length',  
#                             'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
#                             'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
#                             'SocialServ', 'Transporta', 'Water',
#                             'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
#                             'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
#                             'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
#                             'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
#                             'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
#                             'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
#                             'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
#                             'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']

#         self.sviDf = pd.read_csv(self.scene).drop(columns=['Unnamed: 0'])

#         self.trafficLst = trafficLst
#         self.adjLst = adjLst
#         self.xLst = xLst
#         self.inputLength = input_length


#     def len(self):
#         return 72*len(self.dateLst)
#     @property
#     def raw_file_names(self):
#         return []
    
#     def get(self, idx):
#         dateIdx = idx//72
#         timeIdx = idx%72

#         date = self.dateLst[dateIdx]

#         trafficDf = self.trafficLst[dateIdx]
#         originIDs = trafficDf["originID"].values.tolist()

#         adjDf = self.adjLst[dateIdx]
#         F = adjDf["From"].values
#         T = adjDf["To"].values

#         # Get svi scene
#         svi = self.sviDf.iloc[originIDs]

#         # Get se features
#         year = int(date.split("_")[0])-2016
#         xDf = self.xLst[year]

#         xDf = xDf.iloc[originIDs]

#         x = xDf.values
#         x = np.append(x, svi, 1)

#         trafficInfo = trafficDf.values
#         mask = trafficInfo[:, -2]
#         mask = mask.reshape((mask.shape[0], 1))

#         y = trafficInfo[:, timeIdx+24:timeIdx+28]
#         xTraffic = trafficInfo[:, timeIdx+24-self.inputLength: 24+timeIdx]

#         maxv = xTraffic.max(axis=1).reshape([xTraffic.shape[0], 1]) + 1
#         minv = xTraffic.min(axis=1).reshape([xTraffic.shape[0], 1])

#         # y = (y - minv) / (maxv - minv)
#         xTraffic = (xTraffic - minv) / (maxv - minv)

#         x = np.append(x, xTraffic, 1)
#         y = np.append(y, mask, 1)
#         y = np.append(y, minv, 1)
#         y = np.append(y, maxv, 1)
        
#         gY = torch.tensor(y, dtype=torch.float)
#         gX = torch.tensor(x, dtype=torch.float)
#         edge = torch.tensor([F, T], dtype=torch.long)
#         data = Data(x=gX, edge_index=edge, y=gY)

#         return data

if __name__ == "__main__":
    #input_length = [6, 12, 24]
    input_length = 6
    train_trafficPath = r"Data_subGraph/train/data"
    val_trafficPath = r"Data_subGraph/val/data"
    test_trafficPath = r"Data_subGraph/test/data"

    train_adjPath = r"Data_subGraph/train/adj"
    val_adjPath = r"Data_subGraph/val/adj"
    test_adjPath = r"Data_subGraph/test/adj"
    xPath = r"X"
        
    scenePath = r"X/scene.csv"

    columnLst = ['SVIID',
                'StreetWidt', 'Length',  
                'Commercial', 'CulturalFa', 'EducationF','Government', 'HealthServ', 
                'Miscellane', 'PublicSafe', 'Recreation', 'ReligiousI', 'Residentia', 
                'SocialServ', 'Transporta', 'Water',
                'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
                'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
                'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
                'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
                'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
                'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
                'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
                'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']
    xLst = []
    xLst.append(pd.read_csv(os.path.join(xPath, "Divided_2016.csv"))[columnLst])
    xLst.append(pd.read_csv(os.path.join(xPath, "Divided_2017.csv"))[columnLst])
    xLst.append(pd.read_csv(os.path.join(xPath, "Divided_2018.csv"))[columnLst])
    xLst.append(pd.read_csv(os.path.join(xPath, "Divided_2019.csv"))[columnLst])

    trainDateLst, trainTrafficLst, trainAdjLst = getDatasetLst(train_trafficPath, train_adjPath)
    valDateLst, valTrafficLst, valAdjLst = getDatasetLst(val_trafficPath, val_adjPath)
    # testDateLst, testTrafficLst, testAdjLst = getDatasetLst(test_trafficPath, test_adjPath)
    
    train_dataset = NYCTrafficCountDataset_short(trainTrafficLst, xLst, trainAdjLst, trainDateLst, scenePath, inLength)
    # test_dataset = NYCTrafficCountDataset_short(testTrafficLst, xLst, testAdjLst, testDateLst, scenePath, inLength)
    val_dataset = NYCTrafficCountDataset_short(valTrafficLst, xLst, valAdjLst, valDateLst, scenePath, inLength)
    
    training_dataloader = DataLoader(train_dataset, 
                                batch_size=5, 
                                shuffle=False, 
                                num_workers=5)
    validation_dataloader = DataLoader(val_dataset, 
                                    batch_size=5, 
                                    shuffle=False, 
                                    num_workers=5)