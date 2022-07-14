import scipy.io as scio

# 读取数据
E1 = scio.loadmat('data\\S1_E1.mat')
E2 = scio.loadmat('data\\S1_E2.mat')
E3 = scio.loadmat('data\\S1_E3.mat')
E4 = scio.loadmat('data\\S1_E4.mat')

print(E1.keys())
print(E2.keys())
print(E3.keys())
print(E4.keys())
# # 合并E1，E2，E3数据
# E1_emg = E1['emg']
# E2_emg = E2['emg']
# E3_emg = E3['emg']
# E4_emg = E4['emg']
#
# E1_label = E1['label']
# E2_label = E2['label']
# E3_label = E3['label']
# E4_label = E4['label']
#
# E1_label = E1['restimulus']
# index1 =[]
# for i in range(len(E1_label)):
#     if E1_label[i]!=0:
#         index1.append(i)
# label1 = E1_label[index1,:]
# emg1 = E1_emg[index1,:]
#
# index2 =[]
# for i in range(len(E2_label)):
#     if E2_label[i]!=0:
#         index2.append(i)
# label2 = E2_label[index2,:]
# emg2 = E2_emg[index2,:]
#
# index3 =[]
# for i in range(len(E3_label)):
#     if E3_label[i]!=0:
#         index3.append(i)
# label3 = E3_label[index3,:]
# emg3 = E3_emg[index3,:]
#
# emg = np.vstack((emg1,emg2,emg3))
# label = np.vstack((label1,label2,label3))
# label = label-1
#
# print(emg.shape)
# print(label.shape)
# print(label)

# plt.plot(label)

# featureData = []
# featureLabel = []
# classes = 48
# timeWindow = 200
# strideWindow = 200
#
# for i in range(classes):
#     index = [];
#     for j in range(label.shape[0]):
#         if (label[j, :] == i):
#             index.append(j)
#     iemg = emg[index, :]
#     length = math.floor((iemg.shape[0] - timeWindow) / strideWindow)
#     print("class ", i, ",number of sample: ", iemg.shape[0], length)
#
#     for j in range(length):
#         rms = featureRMS(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
#         mav = featureMAV(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
#         wl = featureWL(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
#         zc = featureZC(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
#         ssc = featureSSC(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
#         featureStack = np.hstack((rms, mav, wl, zc, ssc))
#
#         featureData.append(featureStack)
#         featureLabel.append(i)
# featureData = np.array(featureData)
#
# print(featureData.shape)
# print(len(featureLabel))
#
# emg = emg * 20000
#
# imageData = []
# imageLabel = []
# imageLength = 200
# classes = 49
#
# for i in range(classes):
#     index = [];
#     for j in range(label.shape[0]):
#         if (label[j, :] == i):
#             index.append(j)
#
#     iemg = emg[index, :]
#     length = math.floor((iemg.shape[0] - imageLength) / imageLength)
#     print("class ", i, " number of sample: ", iemg.shape[0], length)
#
#     for j in range(length):
#         subImage = iemg[imageLength * j:imageLength * (j + 1), :]
#         imageData.append(subImage)
#         imageLabel.append(i)
#
# imageData = np.array(imageData)
# print(imageData.shape)
# print(len(imageLabel))


# ############文件保存
# file = h5py.File('DB2//DB2_S1_feature_200_0.h5','w')
# file.create_dataset('featureData', data = featureData)
# file.create_dataset('featureLabel', data = featureLabel)
# file.close()
#
# file = h5py.File('DB2//DB2_S1_image_200_0.h5','w')
# file.create_dataset('imageData', data = imageData)
# file.create_dataset('imageLabel', data = imageLabel)
# file.close()