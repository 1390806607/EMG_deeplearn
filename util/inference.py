import torch
import os
from pytorch_network.model import DNN,CNN
import scipy.io as scio
import numpy as np
from feature_utils import featureRMS,featureMAV,featureWL,featureZC,featureSSC
import math
import time,datetime
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
    # 30,3000
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
    # data = np.array(data).reshape((-1,3000,2))
    # data = np.array(data).reshape(-1, 2, 30, 3000).transpose(0, 2, 3, 1).reshape(-1, 3000, 2)
    # label = np.max(np.array(label).reshape(-1, 2, 30).transpose(0, 2, 1).reshape(-1, 2), axis=1)
    data = np.array(data).reshape(-1, 2, 1, 3000).transpose(0, 2, 3, 1).reshape(-1, 3000, 2)
    label = np.max(np.array(label).reshape(-1, 2, 1).transpose(0, 2, 1).reshape(-1, 2), axis=1)
    # print(label_to_id)
    # print(data.shape)
    # print(label.shape)
    return data, label, label_to_id, id_to_label

# 从txt中读取二通道文件
def getdata_from_txt(path,labels_path):

    id2label = {}
    tmp_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        if len(line.strip().split(',')) == 2 and line.strip() != '':
            res = line.strip().split(',')
            try:
                res = list(map(lambda x: int(x), res))
            except ValueError as e:
                continue
            tmp_data.append(res)
    tmp_data = np.array(tmp_data, dtype=np.int32)
    tmp_data = np.array(list(map(lambda a: a / np.max(a), tmp_data)))

    with open(labels_path, 'r', encoding='utf-8') as f1:
        for index,value in enumerate(f1.readlines()):
            id2label[index] = value
        f1.close()
    return tmp_data, id2label


def predict(data,id2label,infeature,input_channels,num_classes,flag=True,timeWindow = 40,strideWindow = 40):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    # model = DNN(in_features=infeature,classes_num=num_classes)
    model = CNN(input_channels, num_classes)
    checkpoint = torch.load('../models/epoch-199.pth.tar',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])  # 模型参数
    # optimizer.load_state_dict(checkpoint['opt_dict'])#优化参数

    model.to(device)
    model.eval()

    # index_list = np.random.permutation(data.shape[0])   # (batch_size,3000,2)
    #
    #
    # input_  = data[index_list[0]]           # (3000,2)
    input_ = data
    # label = target[index_list[0]]
    if flag:
        length = math.floor((input_.shape[0] - timeWindow) / strideWindow)
        temp = []   # 临时变量
        for j in range(length):
            rms = featureRMS(input_[strideWindow * j:strideWindow * j + timeWindow, :])
            mav = featureMAV(input_[strideWindow * j:strideWindow * j + timeWindow, :])
            wl = featureWL(input_[strideWindow * j:strideWindow * j + timeWindow, :])
            zc = featureZC(input_[strideWindow * j:strideWindow * j + timeWindow, :])
            ssc = featureSSC(input_[strideWindow * j:strideWindow * j + timeWindow, :])
            featureStack = np.hstack((rms, mav, wl, zc, ssc))
            temp.append(featureStack)
        input_ = np.array(temp)



    clip = []
    count = {}
    for value in input_:
        clip.append(value)
        if len(clip) == 20:
            inputs = np.array(clip).astype(np.float32)
            # inputs = np.expand_dims(inputs, axis=0).reshape(-1,infeature)
            inputs = np.expand_dims(np.expand_dims(inputs, axis=0), axis=0)
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            start_time = time.time()
            with torch.no_grad():
                outputs = model.forward(inputs)
            # print(f'time:{time.time()-start_time}')
            probs = torch.nn.Sigmoid()(outputs)
            predict_label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            print(id2label[predict_label])
            if id2label[predict_label] not in count.keys():
                count[id2label[predict_label]] = 0
            else:
                count[id2label[predict_label]] += 1
            clip.pop(0)
    print(count)
    print(max(count, key=count.get))



# 15501473127

# 8986 0621 2700 6773 192
# 8614 0017 8008 611

if __name__=='__main__':
    input_size = 400
    input_channels = 1
    hidden_size = 256
    num_classes = 7
    path = '../data/mydataset/1_2/palmar5.txt'
    labels_path = '../data/mydataset/labels.txt'
    data, id_to_label = getdata_from_txt(path, labels_path)
    predict(data,id_to_label,input_size,input_channels,num_classes)
