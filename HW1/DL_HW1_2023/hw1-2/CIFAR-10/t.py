# %% [markdown]
# # 2 Convolutional Neural Network

# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torchtoolbox.tools import summary
import matplotlib.pyplot as plt
import zipfile
import numpy as np

def zip_list(file_path):
    zf = zipfile.ZipFile(file_path, 'r')
    zf.extractall()

if __name__ == '__main__':
    file_path = 'train.zip'
    zip_list(file_path)

# %%
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()
print(device)
# !nvidia-smi

# %%
import cv2
from PIL import Image

train_img = np.zeros((10,200,32,32,3))
test_img = np.zeros((10,40,32,32,3))

path = os.listdir("./train/")
path.sort()
path_train_transport = []
path_test_transport = []

for i in path:
    path_train_transport.append(os.listdir("./train/" + i + "/"))
    path_test_transport.append(os.listdir("./test/" + i + "/"))

# print(len(path_train_transport[0]))
# print(path_train_transport)
# print(len(path_test_transport[0]))
# print(path_test_transport)
print(path)

for i in range(len(train_img)):
    for j in range(len(train_img[0])):
        train_img[i,j] = cv2.imread("./train/" + path[i] + "/" + path_train_transport[i][j])
    for k in range(len(test_img[0])):    
        test_img[i,k] = cv2.imread("./test/" + path[i] + "/" + path_test_transport[i][k])

""" Data Processing """
# Data transforms (normalization & data augmentation)
train_img = train_img.reshape(2000, 32, 32, 3)
test_img = test_img.reshape(400, 32, 32, 3)

stats = ((train_img[:, :, :, 0].mean(), train_img[:, :, :, 1].mean(), train_img[:, :, :, 2].mean()), 
         (train_img[:, :, :, 0].std(), train_img[:, :, :, 0].std(), train_img[:, :, :, 0].std()))

train_tfms = tt.Compose([
    tt.transforms.RandomCrop(32, padding = 4),
    tt.transforms.RandomHorizontalFlip(),
    tt.transforms.ToTensor(),
    tt.transforms.Normalize(mean = stats[0], std = stats[1])])

test_tfms = tt.Compose([
    tt.transforms.ToTensor(),
    tt.transforms.Normalize(mean = stats[0], std = stats[1])])

# train_img = Image.fromarray(np.uint8(train_img[0]))
# test_img = Image.fromarray(np.uint8(test_img[:])) 
# train_img = train_tfms(train_img)
# test_img = train_tfms(test_img)
onehot_train = np.arange(0,2000,1) // 200
onehot_test = np.arange(0,400,1) // 40
train_label = np.eye(10)[onehot_train]
test_label = np.eye(10)[onehot_test]
train_label = torch.Tensor(train_label)
test_label = torch.Tensor(test_label)
print(train_label[0].shape)

# %%
class DeviceDataLoader():
    """ Move data to a device """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """ Yield a batch of data after moving it to device """
        for data in self.dl:
            yield to_device(data, self.device)

    def __len__(self):
        """ Number of Batches """
        return len(self.dl)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label) -> None:
        super().__init__()
        self.dataset = dataset
        self.label = label

    def __getitem__(self, idx):
        return torch.Tensor(self.dataset[idx]), torch.Tensor(self.label[idx])

    def __len__(self):
            return len(self.dataset)

# train_dl = DeviceDataLoader(train_img, device)
# test_dl  = DeviceDataLoader(test_img, device)
train_img = torch.Tensor(train_img)
test_img = torch.Tensor(test_img)
train_dl = train_img.permute(0, 3, 1, 2)
test_dl = test_img.permute(0, 3, 1, 2)
train_dl = DataLoader(dataset = Dataset(train_dl,train_label), batch_size = 128, shuffle = True)
test_dl = DataLoader(dataset = Dataset(test_dl,test_label), batch_size = 128, shuffle = True)
print(type(train_dl))

# %%
""" Model """
def accuracy(pred, label):
    _, preds = torch.max(pred, dim = 1)
    _, labels = torch.max(label, dim = 1)
    return torch.sum(preds == labels) / len(preds)

def convolution_block(in_channel, out_channel, kernel_size, stride, padding):
    layers = [nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, stride = stride, padding = padding),
              nn.BatchNorm2d(out_channel),
              nn.ReLU(inplace = True)]
    return nn.Sequential(*layers)

class ImageClassificationBase(nn.Module):
    def train_step(self, batch):
        data, label = batch
        output = self.forward(data)
        # print("train",output)
        # print("train_label",label)
        loss = F.cross_entropy(output, label)
        acc = accuracy(output, label)
        return {"train_loss": loss, "train_acc": acc}

    def test_step(self, batch):
        data, label = batch
        output = self.forward(data)
        # print("test", output)
        # print("test_label",label)
        loss = F.cross_entropy(output, label)
        acc = accuracy(output, label)
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, train_output, test_output):
        train_losses = [x["train_loss"] for x in train_output]
        epoch_loss = torch.stack(train_losses).mean()           # Combine Train Loss
        train_accs = [x["train_acc"] for x in train_output]
        epoch_acc = torch.stack(train_accs).mean()            # Combine Train Accuracy

        test_losses = [x["test_loss"] for x in test_output]
        test_epoch_loss = torch.stack(test_losses).mean()      # Combine Test Loss
        test_accs = [x["test_acc"] for x in test_output]
        test_epoch_acc = torch.stack(test_accs).mean()       # Combine Test Accuracy

        return {"train_loss": epoch_loss.item(), "train_acc": epoch_acc.item()}, {"test_loss": test_epoch_loss.item(), "test_acc": test_epoch_acc.item()}

    def epoch_end(self, epoch, train_result, test_result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}".format(
              epoch + 1, train_result["train_loss"], train_result["train_acc"]))
        print("Epoch [{}], test_loss: {:.4f}, test_acc: {:.4f}".format(
              epoch + 1, test_result["test_loss"], test_result["test_acc"]))
        
class Inception(nn.Module):
    def __init__(self, input_channel, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        self.b1 = convolution_block(input_channel, n1x1, kernel_size = 1, stride = 1, padding = 0)

        self.b2 = nn.Sequential(convolution_block(input_channel, n3x3_reduce, kernel_size = 1, stride = 1, padding = 0),
                                convolution_block(n3x3_reduce, n3x3, kernel_size = 3, stride = 1, padding = 1))

        self.b3 = nn.Sequential(convolution_block(input_channel, n5x5_reduce, kernel_size = 1, stride = 1, padding = 0),
                                convolution_block(n5x5_reduce, n5x5, kernel_size = 3, stride = 1, padding = 1),
                                convolution_block(n5x5, n5x5, kernel_size = 3, stride = 1, padding = 1))

        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride = 1, padding = 1),
                                convolution_block(input_channel, pool_proj, kernel_size = 1, stride = 1, padding = 0))

    def forward(self, x):
        return torch.cat([self.b1(x),  self.b2(x), self.b3(x), self.b4(x)], dim = 1)

class GoogleNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace = True)
        self.conv_1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1, bias = False)
        self.norm_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False)
        self.norm_2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 192, kernel_size = 3, padding = 1, bias = False)
        self.norm_3 = nn.BatchNorm2d(192)
        self.A1 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.A2 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.A3 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.A4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.A5 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.A6 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.A7 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.A8 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.A9 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.maxpool = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p = 0.2)
        self.fc = nn.Linear(192,10)

    def forward(self, data):
        data = self.conv_1(data)
        data = self.norm_1(data)
        data = self.relu(data)
        data = self.maxpool(data)
        data = self.conv_2(data)
        data = self.norm_2(data)
        data = self.relu(data)
        data = self.maxpool(data)
        data = self.conv_3(data)
        data = self.norm_3(data)
        data = self.relu(data)

        # data = self.maxpool(data)
        # data = self.A1(data)
        # data = self.A2(data)
        # data = self.maxpool(data)

        # data = self.A3(data)
        # data = self.A4(data)
        # data = self.A5(data)
        # data = self.A6(data)
        # data = self.A7(data)
        # data = self.maxpool(data)

        # data = self.A8(data)
        # data = self.A9(data)
        data = self.avgpool(data)
        data = self.dropout(data)
        data = data.view(data.size()[0], -1)
        data = self.fc(data)
        return data

# %%
""" Set Confict """
Model = to_device(GoogleNet(), device)
epoch = 100
max_lr = 1e-2
train_best_acc = 0.
test_best_acc = 0.
optimizer = torch.optim.SGD(params = Model.parameters(), lr = max_lr, momentum = 0.9, weight_decay = 5e-4)
train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epoch // 2)

def evaluate(model, train, test):
    with torch.no_grad():
        model.eval()
        train_outputs = [model.train_step(batch) for batch in train]
        print(train_outputs)
        test_outputs = [model.test_step(batch) for batch in test]
        return model.test_epoch_end(train_outputs, test_outputs)

""" Training　"""
def Train(epoch, max_lr, Model, train_dl, test_dl, opt, epoch_array = [], train_loss_array = [], test_loss_array = [], train_best_acc = 0., test_best_acc = 0.):
    torch.cuda.empty_cache() 
    train_history = []
    test_history = []

    for epoch in range(epoch):
        """ Training """
        Model.train()
        train_losses = []
        batch_idx = 0
        loss_sum = 0.
        train_num = 0
        for batch in train_dl:
            opt.zero_grad()
            loss = Model.train_step(batch)
            print(loss.items())
            loss_sum += loss["train_loss"]
            train_num += 1
            loss["train_loss"].backward()
            opt.step()
            # print("Train Epoch: {}, Loss: {:.6f}".format(epoch,  loss.item()))

        """ Test """
        print("Train loss: ", (loss_sum / train_num))
        train_loss_array.append(loss_sum / train_num)
        epoch_array.append(epoch + 1)
        train_result, test_result = evaluate(Model, train_dl, test_dl)
        train_loss_array.append(train_result["train_loss"])
        test_loss_array.append(test_result["test_loss"])
        Model.epoch_end(epoch, train_result, test_result)
        if train_best_acc < train_result["train_acc"]:
            train_best_acc = train_result["train_acc"]
        if test_best_acc < test_result["test_acc"]:
            test_best_acc = test_result["test_acc"]
        train_history.append(train_result)
        test_history.append(test_result)
        train_scheduler.step()
    return train_history, test_history

# %%
# %%time
epoch_array = []
train_loss_array = []
test_loss_array = []
train_history = []
test_history = []
history1, history2 = Train(epoch, max_lr, Model, train_dl, test_dl, optimizer, epoch_array , train_loss_array, test_loss_array, train_best_acc, test_best_acc)
train_history += history1
test_history += history2


