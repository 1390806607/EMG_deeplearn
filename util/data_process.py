import os
import scipy.io as scio
import numpy as np
from feature_utils import featureRMS,featureMAV,featureWL,featureZC,featureSSC
import math
import h5py
from sklearn.model_selection import train_test_split
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
    filenames = os.listdir(path)
    for filename in filenames:
        matpath = os.path.join(path,filename)
        origin_data = scio.loadmat(matpath)
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
    data = np.array(data).reshape(-1, 2, 1, 3000).transpose(0, 2, 3, 1).reshape(-1, 3000, 2)
    # label = np.max(np.array(label).reshape(-1, 2, 30).transpose(0, 2, 1).reshape(-1, 2), axis=1)
    label = np.max(np.array(label).reshape(-1, 2, 1).transpose(0, 2, 1).reshape(-1, 2), axis=1)
    # print(label_to_id)
    # print(data.shape)
    # print(label.shape)
    return data, label, label_to_id, id_to_label

# 从txt中读取二通道文件
def getdata_from_txt(path,label_path):
    label2id = {}
    id2label = {}
    num = 0
    data = np.array([[0,0]])
    label = []
    for name in os.listdir(path):
        tmp = name.split('.')[0][:-1]
        tmp_data = []
        if tmp not in label2id.keys():
            label2id[tmp] = num
            id2label[num] = tmp
            num += 1
        filename_path = os.path.join(path, name)
        f = open(filename_path, 'r', encoding='utf-8')
        for line in f.readlines():
            if len(line.strip().split(',')) == 2 and line.strip() != '':
                res = line.strip().split(',')
                try:
                    res = list(map(lambda x: int(x), res))
                except ValueError as e:
                    continue
                tmp_data.append(res)
                label.append(label2id[tmp])
        tmp_data = np.array(tmp_data, dtype=np.int32)
        tmp_data = np.array(list(map(lambda a: a / np.max(a), tmp_data)))
        data = np.concatenate([data, tmp_data], axis=0)
    data = data[1:]
    label = np.array(label, dtype=np.int32)
    with open(label_path, 'a', encoding='utf-8') as f:
        for index, value in enumerate(label2id.keys()):
            f.write(value + '\n')
        f.close()
    return data, label, label2id, id2label

# 特征获取
def feature_process(data,
                    label,
                    classes = 6,
                    timeWindow = 40,
                    strideWindow = 40):
    featureData = []
    featureLabel = []
    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if (label[j] == i):
                index.append(j)
        iemg = data[index, :]
        iemg = iemg.reshape(-1,2)
        length = math.floor((iemg.shape[0] - timeWindow) / strideWindow)
        print("class ", i, ",number of sample: ", iemg.shape[0], length)



        for j in range(length):
            rms = featureRMS(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            mav = featureMAV(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            wl = featureWL(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            zc = featureZC(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            ssc = featureSSC(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            featureStack = np.hstack((rms, mav, wl, zc, ssc))

            featureData.append(featureStack)
            featureLabel.append(i)
    featureData = np.array(featureData)
    featureLabel = np.array(featureLabel)
    return featureData,featureLabel




# 构造成图像数据
def get_iamge(data, label, classes=6, flag=True):
    imageData = []
    imageLabel = []
    # imageheight = 40
    imageheight = 20
                            # data (batch_size,10)   label (batch_size,)
    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if (label[j] == i):
                index.append(j)

        iemg = data[index, ...]
        if flag:
            length = math.floor((iemg.shape[0] - imageheight) / imageheight)
        else:
            length = math.floor((iemg.shape[1] - imageheight) / imageheight)    # data (batch_size,3000,2)

        print("class ", i, " number of sample: ", iemg.shape[0], length)


        for j in range(length):
            if flag:
                subImage = iemg[imageheight * j:imageheight * (j + 1), :]
                imageLabel.append(i)
            else:
                subImage = iemg[:,imageheight * j:imageheight * (j + 1), :]
                imageLabel.append([i]*len(index))
            imageData.append(subImage)


    imageData = np.array(imageData).reshape((-1, imageheight, data.shape[-1]))
    imageLabel = np.array(imageLabel).reshape(-1)
    print(imageData.shape)
    print(imageLabel.shape)
    return imageData, imageLabel


def save_h5(save_path,data,label,label_to_id,id_to_label):
    train_X,val_X,train_Y,val_Y = train_test_split(data,label,test_size=0.2)
    file = h5py.File(save_path, 'w')
    file.create_dataset('train_data', data=train_X)
    file.create_dataset('train_label', data=train_Y)
    file.create_dataset('val_data', data=val_X)
    file.create_dataset('val_label', data=val_Y)
    # file.create_dataset('label2id', data=label_to_id)
    # file.create_dataset('id2label', data=id_to_label)
    file.close()


if __name__=='__main__':
    path = '../data/mydataset/1_2'
    save_path = '../data/mydataset/1_2.h5'
    label_path = '../data/mydataset/labels.txt'

    # data, label, label_to_id, id_to_label = get_data(path)
    data, label, label_to_id, id_to_label = getdata_from_txt(path,label_path)
    featureData, featureLabel = feature_process(data,label,classes=7)
    imageData, imageLabel = get_iamge(featureData, featureLabel,classes=7, flag=True)
    save_h5(save_path, imageData, imageLabel, label_to_id, id_to_label)
    # imageData, imageLabel = get_iamge(data, label,flag=False)
    # save_h5(save_path,imageData,imageLabel,label_to_id,id_to_label)
