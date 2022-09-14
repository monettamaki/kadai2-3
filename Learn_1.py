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
#from sklearn import preprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NeuralNetwork(nn.Module):
  #Definition of how to make information smaller    
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),#filter
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=960000, out_features=num_classes)
    def forward(self, x):
        x = self.features(x) #small
        x = x.view(x.size(0), -1)
        x = self.classifier(x) #Classification
        x = torch.nn.functional.rrelu(x)
        return x

def dataload():
    lst_x = pd.read_csv('./Point_data.csv', header=None, dtype=int).iloc[:,0].tolist()
    lst_y = pd.read_csv('./Point_data.csv', header=None, dtype=int).iloc[:,1].tolist()    #print(lst)
    x_pos = []
    y_pos = []
    x_pos = lst_x
    y_pos = lst_y
    #normalization
    x_pos = [n/800 for n in x_pos]
    y_pos = [n/600 for n in y_pos]    
    output_data = [] 
    for i in range(len(x_pos)):
        output_data.append([x_pos[i], y_pos[i]])
    input_data = []
    image_dir = './Image/'
    search_pattern = '*.png'
    image_pathes = glob.glob(os.path.join(image_dir,search_pattern))
    image_pathes.sort()
    for image_path in image_pathes:
        # (height,width,channels)
        data = cv2.imread(image_path)
        # (1,height,width,channels)
        data_expanded = np.expand_dims(data,axis=0)
        input_data.append(data_expanded)
        # (n_samples,height,width,channels)
    image_datas = np.concatenate(input_data,axis=0)
    image_datas = np.transpose(image_datas, (0, 3, 1, 2))
    #print(image_datas)
    #print(image_datas.shape)
    #print(output_data.shape)
    input_data = torch.FloatTensor(image_datas) 
    output_data = torch.FloatTensor(output_data)
    dataset = TensorDataset(input_data, output_data)
    #train用とval用で分ける。
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
    #print("val_dataset{}".format(val_dataset))
    return input_data, output_data, train_loader, val_loader

def train(EPOCHS,input_data, train_loader, val_loader, output_data):
    record_loss_train = []
    record_epoch_train = []
    record_loss_test = []
    test_losses = []
    test_x = []
    test_y = []
    different = []
    model = NeuralNetwork(2)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        model.train()
        loss_train = 0
        for j, xy in enumerate(train_loader):
            train_input = xy[0].to(device)#image
            #train_output = xy[1].to(device)
            train_output = xy[1].to(device)
            #print("train_output.shape{}".format(train_output.shape))
            optimizer.zero_grad()
            output = model(train_input)#point
            #print("output{}".format(output))
            loss = criterion(output, train_output)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()
        loss_train /= j+1
        record_loss_train.append(loss_train)
        record_epoch_train.append(epoch)
                
        model.eval()#更新しないから勾配いらない
        loss_test = 0.0
        for j, xy in enumerate(val_loader):
            #print("j{}".format(j))
            #print("xy{}".format(xy))
            test_input = xy[0].to(device)#image
            test_output = xy[1].to(device)
            test = model(test_input)
            diff = test_output - test
            diff = diff.detach().to('cpu')
            #print(diff.device)
            #print(defe)
            #test_x.append(test[:,0])
            #test_y.append(test[:,1])
            #val_loss = criterion(model(test_input), test_output)
            val_loss = criterion(test, test_output)
            loss_test += val_loss.item()
            different.append(diff)
        loss_test /= j+1
        record_loss_test.append(loss_test)
        if epoch%1 == 0:
            print("epoch: {}, loss: {},val_loss: {},loss_train: {}".format(epoch, loss_train, loss_test,loss_train))      
    return test, output, record_loss_train, record_loss_test, different #test_x, test_y
    
def main():
    EPOCHS = 1000
    input_data, output_data, train_loader,val_loader = dataload()
    test, output, record_loss_train, record_loss_test, different = train(EPOCHS,input_data, train_loader, val_loader, output_data)
    #print(len(different))
    different = torch.cat(different,0)
    #print(different)
    #print(different[:,0])
    different[:,0] = different[:,0]*800
    different[:,1] = different[:,1]*600
    print(different)
    plt.plot(record_loss_train, label='train loss')
    plt.plot(record_loss_test, label='validation loss')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
