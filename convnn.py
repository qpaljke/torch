import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import cv2

import numpy as np
import os
from tqdm import tqdm


class Dataset2Class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)

    def __getitem__(self, index):
        if index < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[index])
        else:
            class_id = 1
            index -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[index])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        img = img.transpose((2, 0, 1))
        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)

        return {'img': t_img, 'label': t_class_id}


class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.LeakyReLU(0.01)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(3, 32, 3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=0)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):

        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.adaptive_pool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out


def count_parameters(model: ConvNN):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


dogs_train_path = "./train/dogs"
cats_train_path = "./train/cats"

dogs_test_path = "./test/dogs"
cats_test_path = "./test/cats"

train_dataset = Dataset2Class(dogs_train_path, cats_train_path)
test_dataset = Dataset2Class(dogs_test_path, cats_test_path)

batch_size = 16

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1,
                                          drop_last=False)

net = ConvNN()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    loss_value = 0
    accuracy_value = 0
    for sample in (pbar := tqdm(train_loader)):
        img, label = sample['img'], sample['label']
        optimizer.zero_grad()

        label = F.one_hot(label, 2).float()
        pred = net(img)

        loss = loss_func(pred, label)

        loss.backward()
        loss_item = loss.item()
        loss_value += loss_item

        optimizer.step()

        curr_accuracy = accuracy(pred, label)
        accuracy_value += curr_accuracy

    pbar.set_description(f'loss: {loss_item}\taccuracy: {curr_accuracy}')
    print(loss_value/len(test_loader))
    print(accuracy_value/len(test_loader))