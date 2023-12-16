#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Hong Wang

@Note: The code are copyed from https://github.com/flora619/GADRP, with a minor repair

https://www.ncbi.nlm.nih.gov/pubmed/29556758


"""



import os
import random
import csv
from torch import nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.utils.data as Data
import datetime
import numpy as np

class Auto_Encoder(nn.Module):
    def __init__(self, device, indim, outdim = 512):
        super(Auto_Encoder, self).__init__()
        self.encoder = Encoder(device=device, indim=indim, outdim=outdim)
        self.decoder = Decoder(device=device, outdim=indim, indim=outdim)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    def output(self, x):
        return self.encoder(x)

class Encoder(nn.Module):
    def __init__(self, device, indim, outdim=512):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(indim, 2048, device=device)
        self.linear2 = nn.Linear(2048, 1024, device=device)
        self.linear3 = nn.Linear(1024, outdim, device=device)
    def forward(self, x):
        x = nn.SELU()(self.linear1(x))
        x = nn.SELU()(self.linear2(x))
        x = nn.Sigmoid()(self.linear3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, device, outdim, indim=512):
        super(Decoder, self).__init__()
        self.linear3 = nn.Linear(indim, 1024, device=device)
        self.linear2 = nn.Linear(1024, 2048, device=device)
        self.linear1 = nn.Linear(2048, outdim, device=device)
    def forward(self, x):
        x = nn.SELU()(self.linear3(x))
        x = nn.SELU()(self.linear2(x))
        x = nn.Sigmoid()(self.linear1(x))
        return x
    
# cell
cell_index_file = 'data/cell/cell_30_ge_cn_id.csv'
cell_ge_file = 'data/cell/ONeil_31_cell_47838_dim_genex.csv'
cell_cn_file = 'data/cell/ONeil_31_cell_23316_dim_CNV.csv'
# cell_rna_file = 'ONeil\ONeil_31_cell_52522_dim_transcripts.csv'

cell_ge_ae = "data\cell\ONeil_30cell_ge_512dim.pt"
cell_cn_ae = "data\cell\ONeil_30cell_cn_512dim.pt"
# cell_rna_ae = "ONeil\ONeil_30cell_rna_512dim.pt"
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_ae(model, trainLoader, test_feature):
    start = datetime.datetime.now()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    best_model = model
    best_loss = 100
    for epoch in range(1, 2500+1):
        for x in trainLoader:
            y = x
            encoded, decoded = model(x)
            train_loss = loss_func(decoded, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        with torch.no_grad():
            y = test_feature
            encoded, decoded = model(test_feature)
            test_loss = loss_func(decoded, y)
        if (test_loss.item() < best_loss):
            best_loss = test_loss
            best_model = model
        if epoch % 100 == 0:
            end = datetime.datetime.now()
            print('epoch:', epoch, 'train loss =  ', train_loss.item(), 'test loss:', test_loss.item(), 'time:',(end - start).seconds)
    return best_model

cell_dict = dict(zip(pd.read_csv(cell_index_file)['cell_line'], pd.read_csv(cell_index_file)['ID']))

def get_cell_features(filename, cell_dict):
    with open(filename) as f:
        reader = csv.reader(f)
        reader.__next__()
        cell_features = [list() for i in range(len(cell_dict))] 
        for line in reader:
            if line[0] in cell_dict:
                cell_features[cell_dict[line[0]]] = list(map(float, line[1:]))
        return cell_features

lr = 0.0001
batch_size = 30
def main():
    random.seed(42)
    # load gene expression data, and DNA copy number data of cell line
    # rna_features = get_cell_features(cell_rna_file, cell_dict)
    ge_features = get_cell_features(cell_ge_file, cell_dict)
    cn_features = get_cell_features(cell_cn_file, cell_dict)

    # rna_features = np.array(rna_features)
    ge_features = np.array(ge_features)
    cn_features = np.array(cn_features)

    #normalization
    min_max = MinMaxScaler()
    # rna_features = torch.tensor(min_max.fit_transform(rna_features)).float().to(device)
    ge_features = torch.tensor(min_max.fit_transform(ge_features)).float().to(device)
    cn_features = torch.tensor(min_max.fit_transform(cn_features)).float().to(device)

    # rna_dim = rna_features.shape[-1]
    ge_dim = ge_features.shape[-1]
    cn_dim = cn_features.shape[-1]
    # print(rna_dim)
    print(ge_dim)
    print(cn_dim)

    # dimension reduction(gene transcirpts)
    # rna_ae = Auto_Encoder(device, rna_dim, 512)
    # train_list = random.sample((rna_features).tolist(), int(0.9 * len(rna_features)))
    # test_list = [item for item in (rna_features).tolist() if item not in train_list]
    # train = torch.tensor(train_list).float().to(device)
    # test = torch.tensor(test_list).float().to(device)
    # data_iter = Data.DataLoader(train, batch_size, shuffle=True)
    # best_model = train_ae(rna_ae, data_iter, test)
    # torch.save(best_model.output(rna_features), cell_rna_ae)

    # dimension reduction(gene expression data)
    ge_ae = Auto_Encoder(device, ge_dim, 512)
    train_list = random.sample((ge_features).tolist(), int(0.9 * len(ge_features)))
    test_list = [item for item in (ge_features).tolist() if item not in train_list]
    train = torch.tensor(train_list).float().to(device)
    test = torch.tensor(test_list).float().to(device)
    data_iter = Data.DataLoader(train, batch_size, shuffle=True)
    best_model = train_ae(ge_ae, data_iter, test)
    torch.save(best_model.output(ge_features), cell_ge_ae)

    # dimension reduction(DNA copy number data)
    cn_ae = Auto_Encoder(device, cn_dim, 512)
    train_list = random.sample((cn_features).tolist(), int(0.9 * len(cn_features)))
    test_list = [item for item in (cn_features).tolist() if item not in train_list]
    train = torch.tensor(train_list).float().to(device)
    test = torch.tensor(test_list).float().to(device)
    data_iter = Data.DataLoader(train, batch_size, shuffle=True)
    best_model = train_ae(cn_ae, data_iter, test)    
    torch.save(best_model.output(cn_features), cell_cn_ae)


if __name__ == '__main__':
    main()

    