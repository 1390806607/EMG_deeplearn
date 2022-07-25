import torch.nn
from torch import nn
import torch.nn.functional as F
import numpy as np
class DNN(nn.Module):
    def __init__(self,in_features,classes_num):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=classes_num)
    def forward(self, x):
        fc1 = self.fc1(x)
        dropout1 = nn.Dropout(0.2)(fc1)
        fc2 = self.fc2(dropout1)
        dropout2 = nn.Dropout(0.4)(fc2)
        output = self.fc3(dropout2)
        return output

class CNN(nn.Module):
    def __init__(self,input_channels,classes):
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.classes = classes
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(4, 3))
        self.bactchN1 = nn.BatchNorm2d(32)
        self.maxpooling1= nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.bactchN2 = nn.BatchNorm2d(64)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.bactchN3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(in_features=512,out_features=128)
        self.output = nn.Linear(in_features=128,out_features=classes)


    def forward(self,x):
        conv1 = self.conv1(x)
        batchN1 = self.bactchN1(conv1)
        #relu1= nn.ReLU()(conv1)
        relu1 = nn.ReLU()(batchN1)
        pooling1 = self.maxpooling1(relu1)
        conv2 = self.conv2(pooling1)
        batchN2 = self.bactchN2(conv2)
        #relu2 = nn.ReLU()(conv2)
        relu2 = nn.ReLU()(batchN2)
        pooling2 = self.maxpooling1(relu2)
        conv3 = self.conv3(pooling2)
        batchN3 = self. bactchN3(conv3)
        #relu3 = nn.ReLU()(conv3)
        relu3 = nn.ReLU()(batchN3)
        flatten = nn.Flatten()(relu3)

        dropout1 = nn.Dropout(0.5)(flatten)
        fc1 = self.fc1(dropout1)
        dropout2 = nn.Dropout(0.5)(fc1)
        output = self.output(dropout2)




        return output



class MLCNN(nn.Module):
    def __init__(self,input_channels,classes):
        super(MLCNN, self).__init__()
        self.input_channels = input_channels
        self.classes = classes
        self.f1 = [4, 3, 2, 1]
        self.f2 = [1, 2, 2, 3]
        self.conv = []
        for i in range(4):
            self.sequential = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(self.f1[i], 3)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,1)),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.f2[i], 3)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1))
            )
            self.conv.append(self.sequential)
        self.fc1 = nn.Linear(in_features=13824,out_features=128)
        self.output = nn.Linear(in_features=128,out_features=classes)




    def forward(self,x):
        conv =[]
        length = len(self.conv)
        for i in range(length):
            conv.append(nn.Flatten()(self.conv[i](x)))
        concat_out = torch.concat(conv,dim=1)       # 安装维度进行拼接
        dropout1 = nn.Dropout(0.5)(concat_out)
        fc1 = self.fc1(dropout1)
        dropout2 = nn.Dropout(0.5)(fc1)
        output = self.output(dropout2)

        return output







if __name__=='__main__':
    input = torch.randn(1,1,40,10)   # 40,10
    # model = DNN(400,6)
    model = MLCNN(1,6)
    output = model(input)
    print(output)