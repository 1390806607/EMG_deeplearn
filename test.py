# # import scipy.io as scio
# # import numpy as np
# #
# # dataFemale_1 = scio.loadmat('data/sEMG_for_Basic_Hand_movements/Database 1/female_1.mat')
# #
# #
# # dataLabel={}
# # PoseV=0
# # IdV=0
# # countP=0
# # countI=0
# # list=[]
# # posture=[]
# # postureData=[]
# # dictPoseToID={}
# # dictIdToPose={}
# #
# #
# #
# #
# # for keys,_ in dataFemale_1.items():
# #     z= keys
# #     if z=='__header__':
# #         continue
# #     elif z=='__version__':
# #         continue
# #     elif z=='__globals__':
# #         continue
# #     list.append(z)
# #
# # o=0
# #
# # for keys,values in dataFemale_1.items():#data array
# #     x= values
# #     if x== b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Nov 18 12:16:25 2014':
# #         continue
# #
# #     elif x== '1.0':
# #         continue
# #     elif x== []:
# #         continue
# #
# #     postureData.append(x)
# #
# # s=np.array(postureData)
# # w=s.reshape(-1,2,3000)
# # while o<180:
# #     if o<30:
# #         dataLabel.update({str(w[o]):0})
# #     elif o<60:
# #         dataLabel.update({str(w[o]):1})
# #     elif o<90:
# #         dataLabel.update({str(w[o]):2})
# #     elif o<120:
# #         dataLabel.update({str(w[o]):3})
# #     elif o<150:
# #         dataLabel.update({str(w[o]):4})
# #     elif o<180:
# #         dataLabel.update({str(w[o]):5})
# #     print(o)
# #     o+=1
# #
# #
# #
# # nameCounter=0
# # while nameCounter<12:
# #     posture.append(list[nameCounter][:-1])
# #     nameCounter+=2
# #
# # while PoseV<12:
# #     dictPoseToID.update({list[PoseV][:-1]:countP})
# #     PoseV+=2
# #     countP+=1
# # while IdV<12:
# #     dictIdToPose.update({countI:list[IdV][:-1]})
# #     IdV+=2
# #     countI+=1
# #
# # #print(dataLabel)
# # #print(dictIdToPose)
# # #print(dictPoseToID)
# # #print(posture)
# # #print(postureData[0])
# # #print(w.shape)  #  (batch,channels,height,wight)
# # print(dataLabel)
#
import os
import scipy.io as scio
import numpy as np
import math
import h5py
from sklearn.model_selection import train_test_split     # pip install sklearn
def get_data(path):
    """

    :param path: 原始数据的路径
    :return:
        data: 训练所需要的数据（batch,time,channels）
        label: 标签(batch,targets)
        label_to_id:  标签对应的索引
        id_to_label:  索引对应的标签
    """
    origin_data = scio.loadmat(path)
    label_to_id = {}
    id_to_label = {}
    num = 0
    data = []
    label = []
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
    data = np.array(data).reshape((180,3000,2))
    label = np.max(np.array(label).reshape((180,2)),axis=1)
    return data, label, label_to_id, id_to_label

# # 特征获取
# def feature_process(data,
#                     label,
#                     classes = 6,
#                     timeWindow = 200,
#                     strideWindow = 200):
#     featureData = []
#     featureLabel = []
#     for i in range(classes):
#         index = []
#         for j in range(label.shape[0]):
#             if (label[j] == i):
#                 index.append(j)
#         iemg = data[index, :]         # (30,3000,2)
#         length = math.floor((iemg.shape[1] - timeWindow) / strideWindow)
#         print("class ", i, ",number of sample: ", iemg.shape[0], length)
#
#         for j in range(length):
#             rms = featureRMS(iemg[:, strideWindow * j:strideWindow * j + timeWindow, :])
#             mav = featureMAV(iemg[:, strideWindow * j:strideWindow * j + timeWindow, :])
#             wl = featureWL(iemg[:, strideWindow * j:strideWindow * j + timeWindow, :])
#             # zc = featureZC(iemg[:, strideWindow * j:strideWindow * j + timeWindow, :])
#             # ssc = featureSSC(iemg[:, strideWindow * j:strideWindow * j + timeWindow, :])
#             featureStack = np.hstack((rms, mav, wl))
#
#             featureData.append(featureStack)
#             featureLabel.append(i)
#     featureData = np.array(featureData)
#     return featureData,featureLabel

#
# 构造成图像数据

def get_iamge(data,label,classes=6):
    imageLength = 200
    set = []
    idSet = []
    # label=(180,)
    for i in range(classes):
        indexCheck=[]
        for x in range(label.shape[0]):
            if label[x]==i:
                indexCheck.append(x)


        iemg = data[indexCheck,...]
        setCount = math.floor(iemg.shape[1]/imageLength)

        for x in range(setCount):
            setImage = iemg[:,x*imageLength:(x+1)*imageLength,:]
            set.append(setImage)
            y= [i]*30

            idSet.append(y)
    set = np.array(set)
    idSet = np.array(idSet)
    set= set.reshape(-1,200,2)
    idSet = idSet.reshape(-1,)
    return set, idSet

#
#
#
#
#
#
#
#
#
#
#
#
#
def save_h5(save_path,data,label):
    train_V, test_V, train_target, test_target = train_test_split(data, label, test_size=0.3)
    file = h5py.File(save_path, 'w')
    file.create_dataset('train_Value', data=train_V)
    file.create_dataset('test_Value', data=test_V)
    file.create_dataset('train_Id', data=train_target)
    file.create_dataset('test_Id', data=test_target)
    file.close()



if __name__=='__main__':
    path = './data/sEMG_for_Basic_Hand_movements/Database 1/female_1.mat'
    save_path = "D:\\ai_project\\sEMG_DeepLearning-master\\data\\sEMG_for_Basic_Hand_movements\\storeDataFile.h5"
    data, label, label_to_id, id_to_label = get_data(path)
    setStored,idSetStored = get_iamge(data, label)
    save_h5(save_path,setStored,idSetStored)

#
# from torch.utils.data import Dataset,DataLoader
# import h5py
# import torch
#
# class dataSet(Dataset):
#     def __init__(self, path, train = True):
#         self.path = path
#         self.data, self.label = self.loadData(train)
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         data = self.data[index]
#         label = self.label[index]
#         return torch.from_numpy(data), torch.tensor(label)
#
#     def loadData(self, train = True):
#         file = h5py.File(path, 'r')
#         if train:
#             data, label = file["train_Value"][:], file["train_Id"][:]
#
#         else:
#             data, label = file["test_Value"][:], file["test_Id"][:]
#
#         file.close()
#         return data, label
#
# if __name__ == "__main__":
#     path= 'D:\\ai_project\sEMG_DeepLearning-master\data\sEMG_for_Basic_Hand_movements\\storeDataFile.h5'
#     train_data = dataSet(path=path, train=False)
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
#     for index,sample in enumerate(train_loader):
#         print(index)
#         print(sample[0])
#         print(sample[1])
#
#
#
#
#


