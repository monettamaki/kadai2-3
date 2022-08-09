#!/usr/bin/python3
import numpy as np
import os, glob
import pathlib
import cv2
import csv
import pandas as pd
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
  #Definition of how to make information smaller
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        #Definition of how to classify
        self.classifier = nn.Linear(in_features=4 * 4 * 128, out_features=num_classes)
    def forward(self, x):
        x = self.features(x) #small
        x = x.view(x.size(0), -1)
        x = self.classifier(x) #Classification
        return x
    
class Create_Datasets(Dataset):
    def __init__(self, img, position):
        self.img = torch.stack(img)
        self.position = torch.stack(position)         
    def __getitem__(self, index):
        path = self.img_paths[index]
        pos = self.position[index]
        return path, pos
    def __len__(self):         
        return len(self.img)  


def dataload():
    lst_x = pd.read_csv('./Point_data.csv', header=None, dtype=int).iloc[:,0].tolist()
    lst_y = pd.read_csv('./Point_data.csv', header=None, dtype=int).iloc[:,1].tolist()    #print(lst)
    #csvファイルのx座標とy座標の情報をリストに
    x_pos = []
    y_pos = []
    x_pos = lst_x
    y_pos = lst_y
    #x_pos = lst[::2]
    #y_pos = lst[1::2]
    output_data = [] 
    for i in range(len(x_pos)):
        output_data.append([x_pos[i], y_pos[i]])
    input_data = []
    image_dir = './Image/'
    search_pattern = '*.png'
    for image_path in glob.glob(os.path.join(image_dir,search_pattern)):
        # (height,width,channels)
        data = cv2.imread(image_path)
        # (1,height,width,channels)
        data_expanded = np.expand_dims(data,axis=0)
        input_data.append(data_expanded)
        # (n_samples,height,width,channels)
    image_datas = np.concatenate(input_data,axis=0)
    #print(image_datas)
    #print(input_data)
    #バッチデータの準備儀式
    input_data = torch.FloatTensor(input_data) 
    output_data = torch.FloatTensor(output_data)
    dataset = TensorDataset(input_data, output_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    return input_data, output_data, train_loader
    print(input_data.shape)
    

def train(EPOCHS,input_data, train_loader, output_data):
    record_loss_train = []
    record_epoch_train = []
    test_losses = []
    test_x = []
    test_y = []
    model = NeuralNetwork(10).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(EPOCHS):
        model.train()
        loss_train = 0
        for j, xy in enumerate(train_loader):
            model = model.to(device)
            train_input = xy[0].to(device)   
            train_output = xy[1].to(device)#ここj[0]じゃないの？
            optimizer.zero_grad()
            output = model(train_input)
            loss = criterion(model(train_input), train_output)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()
        loss_train /= j+1
        record_loss_train.append(loss_train)
        record_epoch_train.append(epoch)
    
        loss_test = 0.0
        for j, xy in enumerate(train_loader):
            input_data.to(device) #.to(device)転送作業、CPU、GPUどっち使うか
            model = model.to(device)
            test = model(input_data).to(device) #open
            test_x.append(test[0])
            test_y.append(test[1])
            test_input = xy[0].to(device)
            test_output = xy[1].to(device)
            val_output = model(test_input)
            val_loss = criterion(model(test_input), test_output)
            loss_test += val_loss.item()
        loss_test /= j+1
        record_loss_test.append(loss_test)    
        if epoch%10 == 0:
            print("epoch: {}, loss: {},  " \
            "val_epoch: {}, val_loss: {}".format(epoch, loss_train, epoch, loss_test))

    return test, test_x, test_y, output, val_output, record_loss_train, record_loss_test

def main():
    EPOCHS = 100
    input_data, output_data, train_loader = dataload()
    test, test_x, test_y, output, val_output, record_loss_train, record_loss_test = train(EPOCHS,input_data, train_loader, output_data) 
    test = test.detach().numpy()
    #output_x = []
    #output_y = []
    #for i in range(50):
    #    output_x.append([open_xy[0][i][0]])
    #    output_y.append([open_xy[0][i][1]])
    #plt.plot(output_x, output_y, linestyle="None", linewidth=0, marker='o')
    plt.plot(test_x, test_y, linestyle="None", linewidth=0, marker='o')
    plt.show()
    plt.style.use('ggplot')
    plt.plot(record_loss_train, label='train loss')
    plt.plot(record_loss_test, label='validation loss')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
