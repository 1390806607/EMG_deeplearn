import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.william_dataset import dataSet
from pytorch_network.williamDNN import DNN

input_size = 400
hidden_size = 256
num_classes = 6
batch_size = 64
num_epochs = 2000
learning_rate = 0.001   # 0.0001
num_workers = 0
momentum = 0.9

def acc(input, target):
    if(input.size() != target.size()):
        input = torch.argmax(input, dim=1)
    accuracy= (targets == input).sum()/len(target)
    return  accuracy

path= 'D:\\ai_project\sEMG_DeepLearning-master\data\sEMG_for_Basic_Hand_movements\\storeDataFile.h5'
save_dir="../models"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_data= dataSet(path, train=True)
train_loader=DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_data= dataSet(path, train=False)
test_loader= DataLoader(test_data, batch_size=16, shuffle = True)

# 创建模型实例化
model = DNN(input_size,num_classes)
# 优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)
# 损失函数
loss_func = nn.BCEWithLogitsLoss()
#使用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in tqdm(range(num_epochs)):
    epoch_loss=0
    epoch_acc=0
    model.train()
    for _,(data,targets) in enumerate(train_loader):
        data= data.view(-1, input_size).float()
        targets =targets.view(-1).to(device).long()


        outputs = model(data).to(device)

        label = F.one_hot(targets,num_classes)
        loss = loss_func(outputs, label.float())


        optimizer.zero_grad()    # 每个小批量的权重进行优化前，进行归零
        loss.backward()          # 对权重梯度进行反向传播
        optimizer.step()         # 更新权重值

        epoch_loss+=loss.item()

        prob = F.sigmoid(outputs)   # 计算输出值的可能性
        epoch_acc += acc(prob,targets)
    if epoch%5==0:
        test_epoch_loss = 0
        test_epoch_acc = 0
        model.eval()
        with torch.no_grad():
            for test_data,test_targets in test_loader:
                test_data = test_data.view(-1,input_size).float()
                test_targets = test_targets.view(-1).to(device).long()
                outputs = model(test_data).to(device)
                label = F.one_hot(test_targets,num_classes)
                loss = loss_func(outputs, label.float())
                test_epoch_loss+=loss.item()

                test_prob = F.sigmoid(outputs)   # 计算输出值的可能性
                test_epoch_acc += acc(prob,targets)
    print(f'epoch:{epoch},train_loss:{epoch_loss/len(train_loader)},train_acc:{epoch_acc/len(train_loader)},'
          f'val_loss:{test_epoch_loss/len(test_loader)},val_acc:{test_epoch_acc/len(test_loader)}')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
    }, os.path.join(save_dir, 'epoch-' + str(epoch) + '.pth.tar'))
    print("Save model at {}\n".format(os.path.join(save_dir, 'epoch-' + str(epoch) + '.pth.tar')))