import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.william_dataset import dataSet
from pytorch_network.williamDNN import DNN
import scipy.io as scio
import numpy as np


def get_data(path):
    """

    :param path: 原始数据的路径
    :return:
        data: 训练所需要的数据（batch,time,channels）
        label: 标签(batch,targets)
        label_to_id:  标签对应的索引
        id_to_label:  索引对应的标签
    """
    label_to_id = {}
    id_to_label = {}
    num = 0
    data = []
    label = []

    origin_data = scio.loadmat(path)
    for key,value in origin_data.items():
        if key in ['__header__', '__version__', '__globals__']:
            continue
        # print(key)
        if key[:-1] not in label_to_id.keys():
            label_to_id[key[:-1]] = num
            id_to_label[num] = key[:-1]
            num += 1
        data.append(value)
        label.append([label_to_id[key[:-1]]]*value.shape[0])
    data = np.array(data).reshape((-1,3000,2))
    label = np.max(np.array(label).reshape((-1,2)),axis=1)
    return data, label, label_to_id, id_to_label

def predict(data, label, id_to_label, infeature, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DNN(input = infeature, outClassNumber = num_classes)
    checkpoint = torch.load('../models/epoch-1999.pth.tar',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])  # 模型参数
    # optimizer.load_state_dict(checkpoint['opt_dict'])#优化参数
    model.to(device)
    model.eval()

    index_list = np.random.permutation(data.shape[0]) #data.shape=(180,3000,2)
    input_ = data[index_list[0]]
    label = label[index_list[0]]

    clip=[]

    for value in input_:
        clip.append(value)
        if len(clip)==200:
            inputs = np.array(clip).astype(np.float32)     # (200,2)
            inputs = np.expand_dims(inputs, axis=0).reshape(-1,infeature)   # 升维(1,200,2)  -> 降维(1,400)
            inputs =  torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad= False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)
            probability = torch.nn.Sigmoid()(outputs)
            predict_label = torch.max(probability, 1)[1].detach().cpu()[0]
            print(id_to_label[predict_label.item()])
            clip.pop()


if __name__ == "__main__":
    input_size = 400
    hidden_size = 256
    num_classes = 6
    batch_size = 64
    num_epochs = 2000
    learning_rate = 0.001  # 0.0001
    num_workers = 0
    momentum = 0.9
    path = '../data/sEMG_for_Basic_Hand_movements/Database 1/female_3.mat'
    data, label, label_to_id, id_to_label = get_data(path)
    predict(data, label, id_to_label, input_size, num_classes)



