import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.dataset import EmgDataset
from pytorch_network.model import DNN, CNN, MLCNN
import matplotlib.pyplot as plt
input_channels = 1
input_size = 400
hidden_size = 256
num_classes = 6
batch_size = 16
num_epochs = 200
learning_rate = 0.0001   # 0.0001                # 学习率可以调参
num_workers = 0
momentum = 0.9                                # 动量也可以仅限在Sgd


def acc(input, target):
    if input.size() != target.size():
        input = torch.argmax(input,dim=1)
    accuray = (targets == input).sum()/len(target)
    return accuray


def loss_acc_plot(train_loss,train_acc,val_acc,val_loss):
    iters = range(len(train_loss))
    plt.figure()
    # acc
    plt.plot(iters, train_acc, 'r', label='train acc')
    # loss
    plt.plot(iters, train_loss, 'g', label='train loss')

    # val_acc
    plt.plot(iters, val_acc, 'b', label='val acc')
    # val_loss
    plt.plot(iters, val_loss, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.show()


##############  读取数据
path = '../data/sEMG_for_Basic_Hand_movements/feautre_cache.h5'
save_dir = '../models'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
train_data = EmgDataset(path,train=True)
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_data = EmgDataset(path,train=False)
val_loader = DataLoader(val_data,batch_size=12,shuffle=True)

# 创建模型实例化
#model = DNN(input_size,num_classes)
#model = CNN(input_channels,num_classes)
model = MLCNN(input_channels,num_classes)
# 优化器
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)       # 优化器可以调 Adam   优化器里面的权重参数
optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-4)
# 损失函数
loss_func = nn.BCEWithLogitsLoss()        # 损失函数可以换
#loss_func = nn.CrossEntropyLoss()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_acc = []
train_loss = []
val_acc = []
val_loss = []
# train the model using minibatch
for epoch in tqdm(range(num_epochs), total=num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for i, (data, targets) in enumerate(train_loader):
        #data = data.view(-1,input_size).float() #DNN
        data = torch.unsqueeze(data,dim=1).float() #CNN,MLCNN
        targets = targets.view(-1).to(device).long()

        # forward
        outputs = model(data).to(device)
       # 计算输出值的可能性

        label = F.one_hot(targets,num_classes)
        loss = loss_func(outputs, label.float())

        # backward and optimize
        optimizer.zero_grad()    # 每个小批量的权重进行优化前，进行归零
        loss.backward()          # 对权重梯度进行反向传播
        optimizer.step()         # 更新权重值

        epoch_loss += loss.item()

        # 求准确率
        prob = F.sigmoid(outputs)   # 计算输出值的可能性
        epoch_acc += acc(prob,targets)
        # every 100 iteration, print loss
        if (i + 1) % 100 == 0:
            print("Step [{}/{}] Train Loss: {:.4f}"
                  .format(i + 1, len(train_loader), loss.item()))
    if epoch % 5 == 0:
        test_epoch_loss = 0
        test_epoch_acc = 0
        model.eval()
        with torch.no_grad():
            for (val_data,val_target) in val_loader:
                #val_data = val_data.view(-1,input_size).float() #DNN
                val_data = torch.unsqueeze(val_data,dim=1).float() #CNN,ML
                val_target = val_target.view(-1).to(device).long()
                # forward
                outputs = model(val_data).to(device)
                label = F.one_hot(val_target, num_classes)
                loss = loss_func(outputs, label.float())
                test_epoch_loss += loss.item()
                val_prop = F.sigmoid(outputs)
                test_epoch_acc += acc(prob,targets)
    train_loss.append(epoch_loss/len(train_loader))
    train_acc.append(epoch_acc/len(train_loader))
    val_acc.append(test_epoch_acc/len(val_loader))
    val_loss.append(test_epoch_loss/len(val_loader))
    print(f'epoch:{epoch},train_loss:{epoch_loss/len(train_loader)},train_acc:{epoch_acc/len(train_loader)},'
          f'val_loss:{test_epoch_loss/len(val_loader)},val_acc:{test_epoch_acc/len(val_loader)}')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
    }, os.path.join(save_dir, 'epoch-' + str(epoch) + '.pth.tar'))
    print("Save model at {}\n".format(os.path.join(save_dir, 'epoch-' + str(epoch) + '.pth.tar')))

# acc 和loss 可视化
loss_acc_plot(train_loss,train_acc,val_acc,val_loss)





