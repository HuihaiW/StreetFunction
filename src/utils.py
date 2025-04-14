import pandas as pd
import numpy as np
from random import shuffle
import torch

import os
import datetime
import torch
# import shuil
from datetime import timedelta, date
import warnings
warnings.filterwarnings("ignore")

def getHourlyData(inputFile, outputFile):
    newRowLst = []
    df = pd.read_csv(inputFile).drop(columns="Unnamed: 0")
    for i in range(df.shape[0]):
        row = df.iloc[i].values
        newRow = []
        for j in range(24):
            data = row[j*4: (j+1)*4]
            if -1 in data:
                newRow.append(-1)
            else:
                m = 106.66053597731798 
                s = 156.377030788485
                # print(type(data))
                data = data.astype(float)
                data = data * s + m
                newRow.append(data.sum())
        newRowLst.append(newRow)
    newColumns = [str(time) for time in range(24)]
    newDF = pd.DataFrame(newRowLst, columns=newColumns)
    newDF.to_csv(outputFile)


def getDate(date, length):
    # return the date before the lenght of days of the input date
    days = []
    for i in range(1, length+1):
        date_i = date - timedelta(i)
        no = date_i.weekday()
        if no < 5:
            y = str(date_i.year)
            m = str(date_i.month)
            d = str(date_i.day)
            fileName = y + "_" + m + "_" + d + ".csv"
            days.append(fileName)
    days.reverse()
    return days


def prepareDates(folder, targetFolder):
    # get the hourly data with 14 days' traffic information + 1 target date traffic information
    dateList = os.listdir(folder)
    i = 0
    for date in dateList:
        print(date, i)
        sDate = date.split("_")
        year = int(sDate[0])
        month = int(sDate[1])
        day = int(sDate[2].split(".")[0])
        d = datetime.date(year, month, day)
        x_days = getDate(d, 14)
        timeDfList = []
        if set(x_days).intersection(set(dateList)) == set(x_days):
            for x_day in x_days:
                timeDf = pd.read_csv(os.path.join(folder, x_day)).drop(columns="Unnamed: 0")
                timeDfList.append(timeDf)
            originDf = pd.read_csv(os.path.join(folder, date)).drop(columns="Unnamed: 0")
            timeDfList.append(originDf)
            targetDf = pd.concat(timeDfList, axis=1)
            targetDf.to_csv(os.path.join(targetFolder, date))
            i+=1

        else:
            continue

# def cleanDataset(folder):


def getMask(folder, targetfolder):
    dataLst = os.listdir(folder)
    for data in dataLst:
        print(data)
        mask = []
        dataPath = os.path.join(folder, data)
        targetPath = os.path.join(targetfolder, data)
        df = pd.read_csv(dataPath)
        columns = df.columns.values[-96:]
        df = df[columns]

        for i in range(df.shape[0]):
            row = df.iloc[i].values
            m = 1
            for d in row:
                if d == -1:
                    m = 0
                    break
            mask.append(m)
        df["mask"] = mask
        print(sum(mask))
        if sum(mask) > 10:
            df.to_csv(targetPath, index=False)
            print("saved")
        
        # return 0
# getMask(r"P3//PreparedData//Y_time//Hourly//WeekdayPrepared_Test", 
#         r"P3//PreparedData//Y_time//Hourly//WeekdayPrepared_Ready")

# ****************************************************************
# Get hourly data from 15-minute data
# dataFolder = r"PreparedData/Y_time/Weekday/"
# newDataFolder = r"PreparedData/Y_time/Hourly/Weekday"
# dataLst = os.listdir(dataFolder)
# for d in dataLst:
#     inputFile = os.path.join(dataFolder, d)
#     outputFile = os.path.join(newDataFolder, d)
#     if os.path.exists(outputFile):
#         continue
#     print(d)
#     getHourlyData(inputFile, outputFile)

# ***************************************************************
# prepare the dataset
# folder = r"P3//PreparedData//Y_time//Hourly//Weekday"
# target = r"P3//PreparedData//Y_time//Hourly//WeekdayPrepared"
# prepareDates(folder,target)
def upstream(SID, adjDf, numLyr):
    downIDs = [SID]
    newFrom = []
    newTo = []
    for i in range(numLyr):
        downIDsTemp = []
        for downID in downIDs:
            subDf = adjDf[adjDf["To"] == downID]
            newFrom += subDf["From"].values.tolist()
            newTo += subDf["To"].values.tolist()
            downIDsTemp += subDf["From"].values.tolist()
        downIDs = list(set(downIDsTemp))
    newDf = pd.DataFrame({"From": newFrom, "To": newTo})
    newDf = newDf.drop_duplicates(ignore_index=True)
    return newDf
    

def smallGraph(originDf, adjDf, numLyr):
    # targetDf: generated new traffic data
    # newAdj: generated corresponding adjacent matrix
    masks = originDf["mask"].values
    idx = np.where(masks == 1)[0]
    newDfLst = []
    for i in idx:
        newDfLst.append(upstream(i, adjDf, numLyr))
    
    newDf = pd.concat(newDfLst, axis=0, ignore_index=True)

    rowIDs = newDf["From"].values.tolist() + newDf["To"].values.tolist()
    rowIDs = list(set(rowIDs))
    rowIDs.sort()
    targetDf = originDf.iloc[rowIDs]
    targetDf["originID"] = rowIDs

    # renew ids of generated new adjacent matrix
    fromLst = newDf["From"].values.tolist()
    toLst = newDf["To"].values.tolist()
    renewed = []
    for i in range(len(fromLst)):
        f = fromLst[i]
        t = toLst[i]
        newF = rowIDs.index(f)
        newT = rowIDs.index(t)
        renewed.append([newF, newT])
    
    newAdj = pd.DataFrame(renewed, columns=["From", "To"])
    return targetDf, newAdj


def MAPE(pred, real):
    # pred = pred * (max - min) + min
    
    # real = real * (max - min) + min
    return torch.mean(torch.abs((pred - real)/(real + 1)))
def RMSE(pred, real):
    loss = torch.nn.MSELoss()
    loss_value = loss(pred, real)
    loss_value = torch.sqrt(loss_value)
    return loss_value

if __name__ == "__main__":
    adjPth = r"C:\\Users\\bigti\\research\\StreetFunction\\adjacentMatrix.csv"
    adjDf = pd.read_csv(adjPth)
    # print(adjDf[adjDf["To"] == 10926])
    origin = r"C:\\Users\\bigti\\research\\StreetFunction\\train"
    target = r"C:\\Users\\bigti\\research\\StreetFunction\\Data\\train"
    nameLst = os.listdir(origin)
    for name in nameLst:
        print(name)
        originPth = os.path.join(origin, name)
        originDf = pd.read_csv(originPth)
        targetDf, newAdj = smallGraph(originDf, adjDf, 10)

        trafficSavePth = os.path.join(target, "data", name)
        adjSavePth = os.path.join(target, "adj", name)

        targetDf.to_csv(trafficSavePth, index=False)
        newAdj.to_csv(adjSavePth, index=False)
    