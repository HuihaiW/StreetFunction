from network import SFGAT
from network import SFGAT_LSTM
from network import SFGAT_LSTM_LONG
from network import SFGAT_SE
from network import SFGAT_SE_LONG
from network import SFGAT_POI_LONG
from network import SFGAT_POI
from network import SFGAT_SCE_LONG
from network import SFGAT_POI_SVI_LONG
from network import SFGAT_LONG
from network import SFGAT_SCE_POI
from dataset import NYCTrafficCountDataset_sub
from dataset import NYCTrafficCountDataset_long
from dataset import NYCTrafficCountDataset_short
from dataset import NYCTrafficCountDataset_short_12
from dataset import NYCTrafficCountDataset_short_6
from network import *
from torch_geometric.data import DataLoader
from tqdm import tqdm
from utils import RMSE
import torch
import os
import numpy as np
import pandas as pd

def train(model, trainDataloader, valDataloader, epoch, loss, optimizer, best_path):
    best = 1000000000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ave_train_loss = []
    ave_val_loss = []
    for i in range(epoch):
        print("-----------" + str(i) + "--------------")
        train_loss = []
        val_loss = []
        with tqdm(trainDataloader) as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                batch.to(device)
                mask = batch.y[:, 1]
                mask = mask == 1

                out = model(batch)
                out = out[mask]

                y = batch.y[:, 0]
                y = y[mask]
                
                minv = batch.y[:, 2][mask]
                maxv = batch.y[:, 3][mask]
                minv = minv.reshape((minv.shape[0], 1))
                maxv = maxv.reshape((maxv.shape[0], 1))

                out = out * (maxv - minv) + minv
                # print(maxv - minv)
                # print(out)
                out = out.reshape(out.shape[0])

                l = loss(out, y)
                # print(l)
                l.backward()
                optimizer.step()

                train_loss.append(l.tolist())
                tepoch.set_postfix(loss=train_loss[-1])

        with torch.no_grad():
            with tqdm(valDataloader) as tepoch:
                for batch in tepoch:
                # for batch in valDataloader:
                    batch.to(device)

                    mask = batch.y[:, 1]
                    mask = mask == 1

                    minv = batch.y[:, 2][mask]
                    maxv = batch.y[:, 3][mask]                    

                    out_val = model(batch)
                    out_val = out_val[mask]
                    out_val = out_val.reshape(out_val.shape[0])
    
                    y = batch.y[:, 0]


                    out_val = out_val * (maxv - minv) + minv
                    y = y[mask]

                    l_val = loss(out_val, y)
                    val_loss.append(l_val.tolist())
                    tepoch.set_postfix(loss=val_loss[-1])
            
        ave_train_loss.append(sum(train_loss)/len(train_loss))
        ave_val_loss.append(sum(val_loss)/len(val_loss))

        # print('Trainning loss is: ' + str(ave_train_loss[-1]))
        print('Trainning loss is: ' + str(ave_train_loss[-1]) + 
              '   , validation loss is: ' + str(ave_val_loss[-1]))
        if ave_val_loss[-1] < best:
            best = ave_val_loss[-1]
            torch.save(model.state_dict(), best_path)
        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
    return ave_train_loss, ave_val_loss

def train_long(model, trainDataloader, valDataloader, epoch, loss, optimizer, best_path):
    best = 1000000000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ave_train_loss = []
    ave_val_loss = []
    for i in range(epoch):
        print("-----------" + str(i) + "--------------")
        train_loss = []
        val_loss = []
        
        with tqdm(trainDataloader) as tepoch:
            # 
            for batch in tepoch:
                # torch.gradient = True
                model.train()
                optimizer.zero_grad()
            # for batch in tqdm(trainDataloader):
                batch.to(device)
                # print(out.shape)
                mask = batch.y[:, -3]
                mask = mask == 1

                out = model(batch)
                out = out[mask]

                y = batch.y[:, 0:4]
                y = y[mask]
                
                minv = batch.y[:, -2][mask]
                maxv = batch.y[:, -1][mask]
                minv = minv.reshape((minv.shape[0], 1))
                maxv = maxv.reshape((maxv.shape[0], 1))

                out = out * (maxv - minv) + minv
                mask2 = y>0
                y = y[mask2]
                out = out[mask2]

                l = loss(out, y)
                l.backward()
                optimizer.step()

                train_loss.append(l.tolist())
                tepoch.set_postfix(loss=train_loss[-1])

        with torch.no_grad():
            with tqdm(valDataloader) as tepoch:
                for batch in tepoch:
                # for batch in valDataloader:
                    batch.to(device)

                    mask = batch.y[:, -3]
                    mask = mask == 1

                    minv = batch.y[:, -2][mask]
                    maxv = batch.y[:, -1][mask]
                    minv = minv.reshape((minv.shape[0], 1))
                    maxv = maxv.reshape((maxv.shape[0], 1))                    

                    out_val = model(batch)
                    out_val = out_val[mask]
                    out_val = out_val
    
                    y = batch.y[:, 0:4]


                    out_val = out_val * (maxv - minv) + minv
                    y = y[mask]
                    
                    mask2 = y>0
                    y = y[mask2]
                    out_val = out_val[mask2]

                    l_val = loss(out_val, y)
                    val_loss.append(l_val.tolist())
                    tepoch.set_postfix(loss=val_loss[-1])
            
        ave_train_loss.append(sum(train_loss)/len(train_loss))
        ave_val_loss.append(sum(val_loss)/len(val_loss))

        # print('Trainning loss is: ' + str(ave_train_loss[-1]))
        print('Trainning loss is: ' + str(ave_train_loss[-1]) + 
              '   , validation loss is: ' + str(ave_val_loss[-1]))
        if ave_val_loss[-1] < best:
            best = ave_val_loss[-1]
            torch.save(model.state_dict(), best_path)

    return ave_train_loss, ave_val_loss

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

if __name__ == "__main__":
    for inLength in [24, 12, 6]:
        outputHorizon = 1
        best_path = os.path.join(r"NewTrain", "Best_SCE_LSTM_"+str(inLength)+"_"+str(outputHorizon)+".pt")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        # net = SFGAT_LONG(inLength).to(device)
        net = SFGAT_SCE(inLength).to(device)
        torch.manual_seed(3407)

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
        # test_dataloader = DataLoader(test_dataset, 
        #                                 batch_size=6, 
        #                                 shuffle=False, 
        #                                 num_workers=5)

        
        epoch=100
        # loss = MAPE
        loss = torch.nn.L1Loss()
        # loss = torch.nn.MSELoss()
        # loss = RMSE
        learning_rate = 0.001
        # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)

        train_loss, val_loss = train(net, 
                                    training_dataloader,
                                    validation_dataloader,
                                    epoch,
                                    loss,
                                    optimizer, 
                                    best_path)
        # loss = {'training': train_loss, 'val':val_loss}
        # loss = pd.DataFrame(loss)
        # loss.to_csv(r"C:\\Users\bigti\\research\\StreetFunction\\weights_new_all_long_24\\loss.csv")