import torch
import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

import warnings
warnings.filterwarnings("ignore")

class SFGAT_LSTM_GAN(torch.nn.Module):
    def __init__(self):
        super(SFGAT_LSTM_GAN, self).__init__()
        self.num_layers = 3

        self.phy1 = nn.Linear(2, 64)
        self.phy2 = nn.Linear(64, 64)

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.gat1 = GATConv(384, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(72, 64, num_layers=self.num_layers)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], 72))
        # t_last = t[:, -1]

        x_phy = self.phy1(x_phy)
        x_phy = self.act2(x_phy)
        x_phy = self.phy2(x_phy)
        x_phy = self.act3(x_phy)   

        x_poi = self.poi1(x_poi)
        x_poi = self.act2(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act3(x_sec) 

        x_svi = self.svi1(x_svi)
        x_svi = self.act2(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = torch.cat((x_phy, x_poi, x_sec, x_svi), 1)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act1(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x

class SFGAT(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.phy1 = nn.Linear(2, 64)
        self.phy2 = nn.Linear(64, 64)

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(384, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_phy = self.phy1(x_phy)
        x_phy = self.act3(x_phy)
        x_phy = self.phy2(x_phy)
        x_phy = self.act3(x_phy)   

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act2(x_sec) 

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = torch.cat((x_phy, x_poi, x_sec, x_svi), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_LSTM(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_LSTM, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()
    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        # # x = torch.cat((x, t), 1)

        x = self.linear1(t)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
class SFGAT_POI_SVI(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_POI_SVI, self).__init__()
        self.num_layers =  input_length     
        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path) 3
        self.inputLength = input_length

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(192, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)
        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)a):
        x_all, edge_index = data.x, data.edge_index
        x_poi = x_all[:,3:16]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1] 

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = torch.cat((x_poi, x_svi), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_POI_SVI_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_POI_SVI_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(192, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_poi = x_all[:,3:16]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1] 

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act1(x_poi)

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act1(x_svi)    
        
        x = torch.cat((x_poi, x_svi), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act1(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_LSTM_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_LSTM_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        # x = torch.cat((x, t), 1)

        x = self.linear1(t)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.phy1 = nn.Linear(2, 64)
        self.phy2 = nn.Linear(64, 64)

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(384, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_phy = self.phy1(x_phy)
        x_phy = self.act3(x_phy)
        x_phy = self.phy2(x_phy)
        x_phy = self.act3(x_phy)   

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act2(x_sec) 

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = torch.cat((x_phy, x_poi, x_sec, x_svi), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
                # self.phy1 = nn.Linear(2, 64)
        # self.phy2 = nn.Linear(64, 64)

        # self.poi1 = nn.Linear(13, 64)
        # self.poi2 = nn.Linear(64, 64)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
        x = self.linear3(x)

        return x
    
class SFGAT_SE_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_SE_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        # self.phy1 = nn.Linear(2, 64)
        # self.phy2 = nn.Linear(64, 64)

        # self.poi1 = nn.Linear(13, 64)
        # self.poi2 = nn.Linear(64, 64)

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        # self.svi1 = nn.Linear(365, 128)
        # self.svi2 = nn.Linear(128, 128)

        # self.all1 = nn.Linear(384, 128)
        # self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act2(x_sec)     
        
        
        x = self.gat1(x_sec, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
        x = self.linear3(x)

        return x
    
class SFGAT_SE(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_SE, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act2(x_sec)    
        
        x = self.gat1(x_sec, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_POI_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_POI_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        # self.phy1 = nn.Linear(2, 64)
        # self.phy2 = nn.Linear(64, 64)

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        # self.sec1 = nn.Linear(40, 128)
        # self.sec2 = nn.Linear(128, 128)

        # self.svi1 = nn.Linear(365, 128)
        # self.svi2 = nn.Linear(128, 128)

        # self.all1 = nn.Linear(384, 128)
        # self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(64, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        # x_phy = self.phy1(x_phy)
        # x_phy = self.act3(x_phy)
        # x_phy = self.phy2(x_phy)
        # x_phy = self.act3(x_phy)   

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        # x_sec = self.sec1(x_sec)
        # x_sec = self.act2(x_sec)
        # x_sec = self.sec2(x_sec)
        # x_sec = self.act2(x_sec) 

        # x_svi = self.svi1(x_svi)
        # x_svi = self.act3(x_svi)
        # x_svi = self.svi2(x_svi)
        # x_svi = self.act3(x_svi)    
        
        # x = torch.cat((x_phy, x_poi, x_sec, x_svi), 1)
        # x = self.all1(x)
        # x = self.act3(x)
        # x = self.all2(x)
        # x = self.act3(x)
        
        x = self.gat1(x_poi, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
                # self.phy1 = nn.Linear(2, 64)
        # self.phy2 = nn.Linear(64, 64)

        # self.poi1 = nn.Linear(13, 64)
        # self.poi2 = nn.Linear(64, 64)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
        x = self.linear3(x)

        return x
    
class SFGAT_POI(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_POI, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.gat1 = GATConv(64, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]  

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)  
        
        x = self.gat1(x_poi, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_SCE(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_SCE, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = self.gat1(x_svi, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_SCE_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_SCE_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = self.gat1(x_svi, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
        x = self.linear3(x)

        return x

class SFGAT_SCE_POI(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_SCE_POI, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        # self.phy1 = nn.Linear(2, 64)
        # self.phy2 = nn.Linear(64, 64)

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        # self.sec1 = nn.Linear(40, 128)
        # self.sec2 = nn.Linear(128, 128)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(128+64, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        # x_phy = self.phy1(x_phy)
        # x_phy = self.act3(x_phy)
        # x_phy = self.phy2(x_phy)
        # x_phy = self.act3(x_phy)   

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        # x_sec = self.sec1(x_sec)
        # x_sec = self.act2(x_sec)
        # x_sec = self.sec2(x_sec)
        # x_sec = self.act2(x_sec) 

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = torch.cat((x_poi, x_svi), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_SCE_POI_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_SCE_POI_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(128+64, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = torch.cat((x_poi, x_svi), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
        x = self.linear3(x)

        return x

class SFGAT_SCE_SE_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_SCE_SE_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(128+128, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act2(x_sec) 

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = torch.cat((x_sec, x_svi), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
                # self.phy1 = nn.Linear(2, 64)
        # self.phy2 = nn.Linear(64, 64)

        # self.poi1 = nn.Linear(13, 64)
        # self.poi2 = nn.Linear(64, 64)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
        x = self.linear3(x)

        return x
    
class SFGAT_SCE_SE(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_SCE_SE, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        self.svi1 = nn.Linear(365, 128)
        self.svi2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(128*2, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act2(x_sec) 

        x_svi = self.svi1(x_svi)
        x_svi = self.act3(x_svi)
        x_svi = self.svi2(x_svi)
        x_svi = self.act3(x_svi)    
        
        x = torch.cat((x_sec, x_svi), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x

class SFGAT_POI_SE(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_POI_SE, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(64+128, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act2(x_sec) 

        x = torch.cat((x_poi, x_sec), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)

        return x
    
class SFGAT_POI_SE_LONG(torch.nn.Module):
    def __init__(self, input_length):
        super(SFGAT_POI_SE_LONG, self).__init__()
        self.num_layers = 3
        self.inputLength = input_length

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)

        self.sec1 = nn.Linear(40, 128)
        self.sec2 = nn.Linear(128, 128)

        self.all1 = nn.Linear(64+128, 128)
        self.all2 = nn.Linear(128, 128)

        self.gat1 = GATConv(128, 128)
        self.gat2 = GATConv(128, 128)
        self.gat3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(self.inputLength, 64, num_layers=self.num_layers)
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 4)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        x_phy = x_all[:, 1:3]
        x_poi = x_all[:,3:16]
        x_sec = x_all[:, 16:56]
        x_svi = x_all[:, 56:421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], self.inputLength))
        # t_last = t[:, -1]

        x_poi = self.poi1(x_poi)
        x_poi = self.act3(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act3(x_poi)

        x_sec = self.sec1(x_sec)
        x_sec = self.act2(x_sec)
        x_sec = self.sec2(x_sec)
        x_sec = self.act2(x_sec) 
        
        x = torch.cat((x_poi, x_sec), 1)
        x = self.all1(x)
        x = self.act3(x)
        x = self.all2(x)
        x = self.act3(x)
        
        x = self.gat1(x, edge_index)
        x = self.act3(x)
        x = self.gat2(x, edge_index)
        x = self.act3(x)
        x = self.gat3(x, edge_index)
        x = self.act3(x)
                # self.phy1 = nn.Linear(2, 64)
        # self.phy2 = nn.Linear(64, 64)

        # self.poi1 = nn.Linear(13, 64)
        # self.poi2 = nn.Linear(64, 64)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)        
        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[2]
        t = self.time1(t)
        t = self.act3(t)
        t = self.time2(t)
        t = self.act3(t)

        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)        # if i%5 == 0:
        #     model_path = os.path.join(normal_path, str(i) + ".pt")
        #     torch.save(model.state_dict(), model_path)
        x = self.linear3(x)

        return x