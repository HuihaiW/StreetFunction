from network import SFGAT_LSTM_GAN
from network import SFGAT
from dataset import NYCTrafficCountDataset_sub
from torch_geometric.data import DataLoader
from tqdm import tqdm
from utils import MAPE
import torch
import os
import numpy as np
import pandas as pd
from train import getDatasetLst
import matplotlib.pyplot as plt

if __name__ == "__main__":
    test_trafficPath = r"C:\\Users\bigti\\research\\StreetFunction\\Data\\test\\data"

    test_adjPath = r"C:\\Users\\bigti\\research\\StreetFunction\\Data\\test\\adj"

    xPath = r"C:\\Users\\bigti\\research\\StreetFunction\\X"
    
    scenePath = r"C:\\Users\\bigti\\research\\StreetFunction\\X\\scene.csv"

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
    print(os.listdir(test_trafficPath))

    testDateLst, testTrafficLst, testAdjLst = getDatasetLst(test_trafficPath, test_adjPath)

    test_dataset = NYCTrafficCountDataset_sub(testTrafficLst, xLst, testAdjLst, testDateLst, scenePath)

    # test_dataloader = DataLoader(test_dataset, 
    #                                 batch_size=6, 
    #                                 shuffle=False, 
    #                                 num_workers=5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    best_path = r"C:\\Users\bigti\\research\\StreetFunction\\weights_LSTM_GAN\\best\\best.pt"
    best_path1 = r"C:\\Users\bigti\\research\\StreetFunction\\weights\\best\\best.pt"
    net = SFGAT().to(device)
    net1 = SFGAT_LSTM_GAN().to(device)
    net.load_state_dict(torch.load(best_path))
    net1.load_state_dict(torch.load(best_path1))

    net.eval()
    net1.eval()
    pLst = []
    rLst = []
    pLst1 = []
    idx = 0
    for i in range(0, 24):
        with torch.no_grad():
            data = test_dataset[i].to(device)
            
            mask = data.y[:, 1]
            mask = mask == 1

            out = net(data)
            out = out[mask]
            out = out.reshape(out.shape[0])

            y = data.y[:, 0]
            y = y[mask]
            
            minv = data.y[:, 2][mask]
            maxv = data.y[:, 3][mask]


            out = out * (maxv - minv) + minv
            out1 = out1

            # p = out.detach().numpy()
            # r = y.detach().numpy()
            p = out.to('cpu').numpy()[idx]
            r = y.to('cpu').numpy()[idx]

            pLst.append(p)
            rLst.append(r)

            # print(p)
            # print(r)
    prLst = test_dataset[0].x[:, 421:]
    print(prLst)

    mask = test_dataset[0].y[:, 1]
    mask = mask == 1

    minv = test_dataset[0].y[:, 2][mask][idx]
    maxv = test_dataset[0].y[:, 3][mask][idx]

    # print(maxv)
    # print(minv)
    mask1 = mask.reshape(mask.shape[0])
    mask1 = mask1.repeat(72).reshape(mask.shape[0], 72).numpy()
    mask1.transpose()

    prLst = prLst[mask][idx]
    # print(prLst)

    prLst = prLst * (maxv - minv) + minv
    prLst = prLst.tolist()

    # print(prLst)
    nullLst = [None] * 72
    plt.plot(prLst + pLst)
    plt.plot(nullLst + rLst)
    plt.show()
    

    # net = SFGAT().to(device)
    # epoch=200
    # loss = MAPE
    # # loss = torch.nn.L1Loss()
    # learning_rate = 0.001
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # best_path = r"C:\\Users\bigti\\research\\StreetFunction\\weights\\best\\best.pt"
    # normal_path = r"C:\\Users\bigti\\research\\StreetFunction\\weights\\normal"
    # # net.load_state_dict(torch.load(best_path))

    #     # print(batch.x)
    #     # break
    

    # train_loss, val_loss = train(net, 
    #                              training_dataloader,
    #                              validation_dataloader,
    #                              epoch,
    #                              loss,
    #                              optimizer, 
    #                              best_path, 
    #                              normal_path)
    # loss = {'training': train_loss, 'val':val_loss}
    # loss = pd.DataFrame(loss)
    # loss.to_csv(r"C:\\Users\bigti\\research\\StreetFunction\\weights\\loss.csv")