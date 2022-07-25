from torch.utils.data import Dataset,DataLoader
import h5py
import torch
class EmgDataset(Dataset):
    def __init__(self,path,train=True):
        self.path = path
        self.data, self.label = self.load_data(train)
    def __len__(self):
        return len(self.data)

    #需要重写__getitem__方法
    def __getitem__(self, index):
        # Loading and preprocessing.
        data = self.data[index]
        label = self.label[index]
        return torch.from_numpy(data), torch.tensor([label])

    def load_data(self,train=True):
        file = h5py.File(self.path, 'r')
        if train:
            data, label= file['train_data'][:], file['train_label'][:]
        else:
            data, label = file['val_data'][:], file['val_label'][:]
        file.close()
        return data, label



if __name__=='__main__':
    path = '../data/mydataset/1_2.h5'
    train_data = EmgDataset(path=path, train=True)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=False, num_workers=1)
    for index,sample in enumerate(train_loader):
        print(index)
        print(sample[0])
        print(sample[1])


